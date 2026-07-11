"""Phase-driven training.

  Phase 0  — SASRec alone, sequential CTR with BCE.          -> sasrec.pt
  Phase 1  — LoRA warm-up on the TEXT-ONLY prompt.           -> lora_phase1.pt
             The LLM first learns the task format (answer "Yes"/"No" from
             titles) WITHOUT collaborative tokens, so Phase 2 gradients reflect
             collaborative signal, not prompt-format learning.
  Phase 2  — Full hybrid prompt. LLM AND LoRA frozen (Phase-1 weights loaded);
             only the QFormer + projection heads train. SASRec frozen (Phase 0).
             -> qformer.pt (soup)
  Phase 2b — Phase 2 with `unfreeze_sasrec=True`: SASRec fine-tunes jointly.

  We deliberately do NOT offer a joint one-step LoRA+QFormer path: per CoLLM's
  ablations it underperforms, especially on cold users — the LoRA gradient
  dominates early and the mapping module never learns to carry collaborative
  information.

Freeze/unfreeze is handled per phase by building the optimizer over EXACTLY the
intended parameter set and additionally flipping requires_grad, so a bug in one
mechanism cannot silently train the wrong weights.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import load_data, collate, UserGroupedBatchSampler
from losses import combined_loss, bce_loss
from models import SASRec, QFormerBridge, LLMRec
from selection import CheckpointSelector
from evaluate import qualifying_users, score_dataset, auc as auc_fn, uauc as uauc_fn


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(ds, cfg, batch_size, grouped=True, seed=0):
    if grouped:
        sampler = UserGroupedBatchSampler(ds.uid, batch_size=batch_size,
                                          users_per_batch=cfg.users_per_batch, seed=seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                          num_workers=cfg.num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                      num_workers=cfg.num_workers)


# --------------------------------------------------------------------------- #
# Phase 0: SASRec CTR pre-training
# --------------------------------------------------------------------------- #

def train_phase0(cfg: Config, train_ds, val_ds, n_items, device) -> tuple[SASRec, dict]:
    sasrec = SASRec(n_items, cfg.emb_dim, cfg.max_his_len,
                    cfg.sasrec_blocks, cfg.sasrec_heads, cfg.sasrec_dropout).to(device)
    opt = torch.optim.AdamW(sasrec.parameters(), lr=cfg.phase0_lr,
                            weight_decay=cfg.phase0_weight_decay)
    dl = make_loader(train_ds, cfg, cfg.phase0_batch_size, grouped=False, seed=cfg.seed)

    history = {"loss": [], "val_auc": [], "val_uauc": []}
    best_auc, best_state = -1.0, None
    for epoch in range(cfg.phase0_epochs):
        sasrec.train()
        losses = []
        for b in dl:
            b = b.to(device)
            logit = sasrec.ctr_logit(b.his, b.his_mask, b.iid)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, b.label)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        v_auc, v_uauc = eval_sasrec(sasrec, val_ds, device)
        history["loss"].append(float(np.mean(losses)))
        history["val_auc"].append(v_auc); history["val_uauc"].append(v_uauc)
        print(f"[phase0] epoch {epoch + 1}/{cfg.phase0_epochs} "
              f"loss {history['loss'][-1]:.4f} val AUC {v_auc:.4f} val UAUC {v_uauc:.4f}")
        if v_auc > best_auc:
            best_auc = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in sasrec.state_dict().items()}

    sasrec.load_state_dict(best_state)
    out = Path(cfg.out_dir); out.mkdir(exist_ok=True, parents=True)
    torch.save(best_state, out / "sasrec.pt")
    print(f"[phase0] saved best (val AUC {best_auc:.4f}) -> {out / 'sasrec.pt'}")
    return sasrec, history


@torch.no_grad()
def eval_sasrec(sasrec, ds, device, batch_size=512):
    sasrec.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    uids, labels, scores = [], [], []
    for b in dl:
        b = b.to(device)
        s = torch.sigmoid(sasrec.ctr_logit(b.his, b.his_mask, b.iid))
        uids.append(b.uid.cpu().numpy()); labels.append(b.label.cpu().numpy())
        scores.append(s.cpu().numpy())
    uids, labels, scores = map(np.concatenate, (uids, labels, scores))
    return auc_fn(labels, scores), uauc_fn(uids, labels, scores)


# --------------------------------------------------------------------------- #
# Shared LLM-stage helpers
# --------------------------------------------------------------------------- #

def _val_scores(cfg, llm, sasrec, qformer, val_ds, device, hybrid):
    """Val scores for checkpoint selection; optionally on a fixed user subsample."""
    return score_dataset(llm, sasrec, qformer, val_ds,
                         batch_size=cfg.phase2_batch_size * 2,
                         device=device, hybrid=hybrid,
                         progress=not cfg.smoke_test)  # 7B over full val is minutes-long


def _selection_val_set(cfg, val_ds):
    """The validation set used for per-step checkpoint selection.

    `val_subsample_users > 0` caps eval cost for the full LLM run: a FIXED
    (seeded) subset of qualifying users, chosen once per stage so every
    checkpoint is scored on the identical set. Final reported numbers always
    come from the full test split — this only affects selection.
    """
    users = qualifying_users(val_ds.uid, val_ds.label)
    if cfg.val_subsample_users and cfg.val_subsample_users < len(users):
        rng = np.random.default_rng(cfg.seed)
        users = np.sort(rng.choice(users, size=cfg.val_subsample_users, replace=False))
        idx = np.flatnonzero(np.isin(val_ds.uid, users))
        val_ds = torch.utils.data.Subset(val_ds, idx.tolist())
    return val_ds, users


def _llm_stage_loop(cfg, llm, sasrec, qformer, train_ds, val_ds, device, *,
                    hybrid: bool, params: list, lr: float, epochs: int,
                    batch_size: int, grad_accum: int, tag: str,
                    tracked_state_fn) -> tuple[CheckpointSelector, dict]:
    """One training loop shared by Phases 1/2/2b — they differ only in which
    parameters train and whether soft tokens are injected."""
    opt = torch.optim.AdamW(params, lr=lr)
    dl = make_loader(train_ds, cfg, batch_size, grouped=True, seed=cfg.seed)
    val_ds, val_users = _selection_val_set(cfg, val_ds)
    print(f"[{tag}] {sum(p.numel() for p in params):,} trainable params, "
          f"{len(val_users)} qualifying val users for selection")

    selector = CheckpointSelector(sel_window=cfg.sel_window, top_k=cfg.top_k_soup,
                                  patience=cfg.patience, n_boot=cfg.n_boot, seed=cfg.seed)
    history = {"loss": [], "bce": [], "pair": []}
    step, stop = 0, False
    t_last, step_last = time.time(), 0
    for epoch in range(epochs):
        if stop:
            break
        llm.train(); qformer.train()
        sasrec.train(cfg.unfreeze_sasrec and hybrid)
        opt.zero_grad()
        for b in dl:
            b = b.to(device)
            if hybrid:
                H = sasrec.encode_history(b.his, b.his_mask)
                e_i = sasrec.item_embedding(b.iid)
                u_tok, i_tok = qformer(H, b.his_mask, e_i)
                p = llm(b.his_titles, b.target_titles, u_tok, i_tok)
            else:
                p = llm(b.his_titles, b.target_titles)
            loss, bce, pair = combined_loss(p, b.label, b.uid,
                                            lambda_pair=cfg.lambda_pair,
                                            margin=cfg.pair_margin)
            (loss / grad_accum).backward()
            history["loss"].append(loss.item())
            history["bce"].append(bce.item()); history["pair"].append(pair.item())
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
            if step % cfg.log_every_steps == 0:
                w = cfg.log_every_steps
                rate = (step - step_last) / max(time.time() - t_last, 1e-9)
                t_last, step_last = time.time(), step
                print(f"[{tag}] epoch {epoch + 1} step {step}/{len(dl) * epochs} "
                      f"loss {np.mean(history['loss'][-w:]):.4f} "
                      f"(bce {np.mean(history['bce'][-w:]):.4f} "
                      f"pair {np.mean(history['pair'][-w:]):.4f}) "
                      f"{rate:.2f} step/s", flush=True)
            if step % cfg.eval_every_steps == 0:
                uids, labels, scores = _val_scores(cfg, llm, sasrec, qformer, val_ds,
                                                   device, hybrid)
                stop = selector.update(step, uids, labels, scores, val_users,
                                       tracked_state_fn())
                llm.train(); qformer.train()
                sasrec.train(cfg.unfreeze_sasrec and hybrid)
                t_last, step_last = time.time(), step  # don't count eval in step/s
                if stop:
                    print(f"[{tag}] early stop at step {step} (patience)")
                    break
        print(f"[{tag}] epoch {epoch + 1}/{epochs} mean loss "
              f"{np.mean(history['loss'][-len(dl):]):.4f}")

    if not selector.history:  # ensure at least one checkpoint exists
        uids, labels, scores = _val_scores(cfg, llm, sasrec, qformer, val_ds, device, hybrid)
        selector.update(step, uids, labels, scores, val_users, tracked_state_fn())
    return selector, history


# --------------------------------------------------------------------------- #
# Phase 1: LoRA warm-up (text-only prompt)
# --------------------------------------------------------------------------- #

def train_phase1(cfg: Config, llm, sasrec, qformer, train_ds, val_ds, device):
    llm.set_lora_trainable(True)
    for m in (sasrec, qformer):
        m.requires_grad_(False)
    params = llm.trainable_lora_parameters()

    selector, history = _llm_stage_loop(
        cfg, llm, sasrec, qformer, train_ds, val_ds, device,
        hybrid=False, params=params, lr=cfg.phase1_lr, epochs=cfg.phase1_epochs,
        batch_size=cfg.phase1_batch_size, grad_accum=cfg.phase1_grad_accum,
        tag="phase1", tracked_state_fn=llm.lora_state_dict)

    # Phase 1 selects a single best checkpoint (souping happens in Phase 2,
    # where the final model is assembled)
    best = selector.best_state()
    llm.load_lora_state_dict(best)
    out = Path(cfg.out_dir); out.mkdir(exist_ok=True, parents=True)
    torch.save(best, out / "lora_phase1.pt")
    print(f"[phase1] saved LoRA -> {out / 'lora_phase1.pt'}")
    return selector, history


# --------------------------------------------------------------------------- #
# Optional: QFormer alignment pre-training (contrastive)
# --------------------------------------------------------------------------- #

def align_pretrain_qformer(cfg: Config, sasrec, qformer, train_ds, device):
    """Match user-query readouts to target item embeddings with in-batch
    softmax contrast — a cheap initialization so Phase 2 starts from queries
    that already attend to informative history positions."""
    qformer.requires_grad_(True)   # Phase 1 froze it; alignment trains it
    opt = torch.optim.AdamW(qformer.parameters(), lr=cfg.align_lr)
    dl = make_loader(train_ds, cfg, min(32, cfg.phase2_batch_size * 4),
                     grouped=False, seed=cfg.seed)
    # eval mode on purpose: (a) a deterministic, dropout-free contrastive init,
    # and (b) MPS's SDPA kernel raises NotImplementedError for attention dropout
    # once a transformers forward has flipped the MHA dispatch path. Gradients
    # still flow in eval mode — only dropout is disabled.
    sasrec.eval(); qformer.eval()
    for epoch in range(cfg.align_epochs):
        losses = []
        for b in dl:
            b = b.to(device)
            pos = b.label > 0.5           # align only on positives (user liked item)
            if pos.sum() < 2:
                continue
            with torch.no_grad():
                H = sasrec.encode_history(b.his[pos], b.his_mask[pos])
                e = sasrec.item_embedding(b.iid[pos])
            logits = qformer.align_scores(H, b.his_mask[pos], e)
            target = torch.arange(logits.size(0), device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[align] epoch {epoch + 1}/{cfg.align_epochs} loss {np.mean(losses):.4f}")


# --------------------------------------------------------------------------- #
# Phase 2 / 2b: QFormer (+ optionally SASRec) on the full hybrid prompt
# --------------------------------------------------------------------------- #

def train_phase2(cfg: Config, llm, sasrec, qformer, train_ds, val_ds, device):
    # freeze LLM + LoRA (Phase-1 weights already loaded); train QFormer (+SASRec in 2b)
    llm.set_lora_trainable(False)
    llm.model.requires_grad_(False)
    qformer.requires_grad_(True)
    sasrec.requires_grad_(cfg.unfreeze_sasrec)
    params = list(qformer.parameters())
    if cfg.unfreeze_sasrec:
        params += [p for p in sasrec.parameters() if p.requires_grad]

    def tracked_state():
        # track everything the soup must average: QFormer (+SASRec in 2b)
        sd = {f"qformer.{k}": v for k, v in qformer.state_dict().items()}
        if cfg.unfreeze_sasrec:
            sd.update({f"sasrec.{k}": v for k, v in sasrec.state_dict().items()})
        return sd

    selector, history = _llm_stage_loop(
        cfg, llm, sasrec, qformer, train_ds, val_ds, device,
        hybrid=True, params=params, lr=cfg.phase2_lr, epochs=cfg.phase2_epochs,
        batch_size=cfg.phase2_batch_size, grad_accum=cfg.phase2_grad_accum,
        tag="phase2b" if cfg.unfreeze_sasrec else "phase2",
        tracked_state_fn=tracked_state)

    # final model = weight-average soup of the top-k checkpoints
    soup = selector.soup(cfg.top_k_soup)
    load_tracked_state(soup, qformer, sasrec if cfg.unfreeze_sasrec else None)
    out = Path(cfg.out_dir); out.mkdir(exist_ok=True, parents=True)
    torch.save(soup, out / "qformer.pt")
    print(f"[phase2] saved souped QFormer -> {out / 'qformer.pt'}")
    return selector, history


def load_tracked_state(sd: dict, qformer, sasrec=None):
    qformer.load_state_dict({k[len("qformer."):]: v for k, v in sd.items()
                             if k.startswith("qformer.")})
    if sasrec is not None and any(k.startswith("sasrec.") for k in sd):
        sasrec.load_state_dict({k[len("sasrec."):]: v for k, v in sd.items()
                                if k.startswith("sasrec.")})


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_models(cfg: Config, n_items: int, device: str):
    llm = LLMRec(cfg.backbone, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout,
                 cfg.lora_targets, cfg.load_4bit, device)
    sasrec = SASRec(n_items, cfg.emb_dim, cfg.max_his_len, cfg.sasrec_blocks,
                    cfg.sasrec_heads, cfg.sasrec_dropout).to(device)
    qformer = QFormerBridge(cfg.emb_dim, llm.llm_dim, cfg.n_queries,
                            cfg.qformer_layers, cfg.qformer_heads,
                            cfg.qformer_dropout, cfg.target_aware).to(device)
    return llm, sasrec, qformer


def run_all_phases(cfg: Config, seed: int | None = None):
    """Full pipeline for one seed. Returns dict of models, selectors, histories."""
    seed = cfg.seed if seed is None else seed
    seed_everything(seed)
    device = cfg.resolve_device()
    print(f"[run] seed {seed}, device {device}, smoke_test={cfg.smoke_test}")

    train_ds, val_ds, test_ds, n_users, n_items, id2title = load_data(cfg)

    # Phase 0
    sasrec, hist0 = train_phase0(cfg, train_ds, val_ds, n_items, device)

    # LLM + QFormer
    llm, _, qformer = build_models(cfg, n_items, device)
    sasrec = sasrec.to(device)

    # Phase 1
    sel1, hist1 = train_phase1(cfg, llm, sasrec, qformer, train_ds, val_ds, device)

    # Optional alignment pre-training for the QFormer
    if cfg.qformer_align_pretrain:
        align_pretrain_qformer(cfg, sasrec, qformer, train_ds, device)

    # Phase 2 (or 2b when cfg.unfreeze_sasrec)
    sel2, hist2 = train_phase2(cfg, llm, sasrec, qformer, train_ds, val_ds, device)

    return {"cfg": cfg, "seed": seed, "device": device,
            "datasets": (train_ds, val_ds, test_ds),
            "models": (llm, sasrec, qformer),
            "selectors": (sel1, sel2),
            "histories": (hist0, hist1, hist2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--unfreeze_sasrec", action="store_true")
    ap.add_argument("--qformer_align_pretrain", action="store_true")
    ap.add_argument("--target_aware", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.smoke() if args.smoke_test else Config()
    cfg.unfreeze_sasrec = args.unfreeze_sasrec
    cfg.qformer_align_pretrain = args.qformer_align_pretrain
    cfg.target_aware = bool(args.target_aware)
    if args.seed is not None:
        cfg.seed = args.seed

    run_all_phases(cfg)
