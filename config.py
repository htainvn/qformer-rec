"""Central configuration for the SASRec -> target-conditioned QFormer -> LLM recommender.

Everything tunable lives here so experiments are reproducible from a single object.
Use `Config()` for the full run and `Config.smoke()` for the CPU-only smoke test.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    data_dir: str = "ml-1m"          # CoLLM's preprocessed ML-1M pickles
    # SASRec/QFormer history length (left-padded, pad 0). Decoupled from the
    # 10-title prompt cap: sweep on real ML-1M val AUC peaked at L=100
    # (0.6991 vs 0.6885 at L=10; L=150/200 decline) — the collaborative encoder
    # needs a longer view than the prompt text can carry.
    max_his_len: int = 100
    # titles listed in the prompt text. MEASURED: raising to 20 (with the
    # bigger LoRA + lambda 0.8) made Phase 1 WORSE (~0.69 AUC / 0.679 UAUC vs
    # 0.717/0.695 at 10) — the extra titles are older, weaker-signal history
    # that dilutes the prompt. 10 is the proven value.
    prompt_titles: int = 10
    smoke_test: bool = False         # tiny synthetic data + tiny backbone, CPU-friendly

    # ---- SASRec (collaborative backbone) -----------------------------------
    # regularization matters here: with ~34k train rows SASRec overfits within
    # a few epochs (val AUC peaks ~3 then declines while train loss -> 0.1),
    # and a weak Phase-0 model caps the whole pipeline's UAUC
    emb_dim: int = 64                # item embedding dim
    sasrec_blocks: int = 2           # causal self-attention blocks
    sasrec_heads: int = 2
    sasrec_dropout: float = 0.3
    phase0_lr: float = 1e-3
    phase0_epochs: int = 30          # upper bound; early stopping usually ends sooner
    phase0_patience: int = 5         # stop after this many epochs w/o val-AUC gain
    phase0_batch_size: int = 256
    phase0_weight_decay: float = 1e-4
    phase0_neg_ratio: int = 0        # 0 = use labeled samples as-is (CTR-style BCE)

    # ---- QFormer bridge -----------------------------------------------------
    # bridge="mlp" swaps in CoLLM's original mapping (MLP of SASRec's last
    # state -> 1 soft token) with everything else identical — the controlled
    # baseline arm for the paper's core MLP-vs-QFormer claim.
    bridge: str = "qformer"          # "qformer" | "mlp"
    n_queries: int = 4               # N learnable queries -> N <UserID> soft tokens
    qformer_layers: int = 2
    qformer_heads: int = 4
    qformer_dropout: float = 0.2     # the 4.4M bridge memorizes 839 train users
                                     # within ~1k steps; regularize hard
    target_aware: bool = True        # FiLM-condition queries on target item embedding
    # Design 2 (models/README.md): pre-train a DIN encoder alongside SASRec and
    # feed the QFormer fused keys/values concat([H_sasrec, D_din]) of width
    # 2*emb_dim, with target-agnostic queries (the target already lives inside
    # DIN's per-position weighting). The escalation path when Design 1's UAUC
    # plateaus — which it did, at ~0.695 val.
    design2: bool = False
    qformer_align_pretrain: bool = False  # optional contrastive alignment before Phase 2
    # ---- SeLLa-Rec adaptation (arXiv:2504.10107), recast around the QFormer --
    # SeLLa-Rec's transferable ideas, with the QFormer playing the role of its
    # projection layer: (1) distill per-item SEMANTIC vectors from the Phase-1
    # LoRA-tuned LLM (last hidden state over the quoted title — the OUTPUT
    # space the reader actually reasons in, unlike build_title_vectors' frozen
    # INPUT-embedding mean); (2) contrastively pre-align the bridge's user/item
    # tokens to those vectors BEFORE Phase 2 (their Stage 2 + warm-started
    # projection); (3) a third <WarmID> soft token carrying the projected
    # semantic vector of the TARGET item (their <Warm_ID>). All off by default;
    # the full SeLLa arm = sella_prealign + sella_warm_token.
    sella_prealign: bool = False     # Stage-2 InfoNCE alignment of the bridge
    sella_prealign_epochs: int = 3
    sella_prealign_lr: float = 1e-3
    # cosine InfoNCE is norm-blind, but token NORM is load-bearing here (the
    # zero-init note below/in qformer.py: out-of-scale tokens wreck the Phase-1
    # prompt). The MSE term anchors aligned tokens to the scale of real
    # semantic embeddings, so the warm-started Phase 2 starts readable.
    sella_prealign_mse: float = 0.25
    sella_tau: float = 0.07          # InfoNCE temperature
    sella_warm_token: bool = False   # third <WarmID> soft token (target item)
    sella_anchor: bool = False       # align_titles anchors to DISTILLED vectors
                                     # instead of input-embedding title means
                                     # (requires align_titles=True to matter)
    sella_distill_batch: int = 64    # titles per LLM forward during distillation
    # Title-anchored alignment (auxiliary Phase-2/3 loss): pull the MEAN of the
    # user soft tokens toward the mean LLM title-embedding of the FULL history
    # (100 items; the prompt shows only 10) — tokens become readable by
    # construction AND carry the 90 unshown titles. Item token is NOT anchored
    # to its own title (that would be redundant with the prompt).
    align_titles: bool = False
    align_titles_weight: float = 0.1
    align_epochs: int = 3
    align_lr: float = 1e-3

    # ---- LLM ----------------------------------------------------------------
    # The collaborative UAUC signal saturates ~0.68 (SASRec == DIN, blends
    # worse); the reader is the lever — but MEASURED: r=16 on all 7 projections
    # (40M params) UNDERPERFORMED r=8 on q/v (4M) by ~2pts on 34k samples
    # (capacity >> data). The gentle adapter is the proven recipe; the
    # stronger-reader lever that remains open is the BACKBONE (Qwen2.5-7B).
    backbone: str = "lmsys/vicuna-7b-v1.5"   # or "Qwen/Qwen2.5-7B" (base; tokenizer verified)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "v_proj")
    load_4bit: bool = False          # set True to QLoRA-quantize the frozen backbone

    # ---- LR schedule (LLM stages) --------------------------------------------
    # "" = constant lr (the house default). "cosine" = CoLLM's schedule:
    # linear warmup from warmup_lr over warmup_steps, then cosine decay from
    # the stage lr down to min_lr — verified in CoLLM's released yaml configs
    # (init_lr 1e-3, min_lr 8e-5, warmup_lr 1e-5, warmup_steps 200).
    lr_schedule: str = ""            # "" | "cosine"
    min_lr: float = 8e-5
    warmup_steps: int = 200
    warmup_lr: float = 1e-5

    # ---- loss ---------------------------------------------------------------
    # Within-user pairwise (BPR) weight. MEASURED: 0.8 traded ~1pt val AUC for
    # no UAUC gain (BPR is invariant to per-user score shifts, so upweighting
    # it decalibrates the cross-user ordering pooled AUC measures). 0.5 is the
    # value the best observed run used.
    lambda_pair: float = 0.5
    pair_margin: float = 0.0         # 0.0 -> plain BPR softplus; >0 -> margin hinge

    # ---- Phase 1 (LoRA warm-up, text-only prompt) ---------------------------
    # epochs are an UPPER BOUND — patience early-stopping on smoothed val UAUC
    # is the real terminator. The observed 1-epoch run ended with the val curve
    # still rising (UAUC 0.6887->peak 0.6916, AUC still climbing), i.e. the
    # epoch limit cut training short, not the data.
    phase1_lr: float = 1e-4
    phase1_epochs: int = 5           # upper bound; patience early-stops at the plateau
    phase1_batch_size: int = 8
    phase1_grad_accum: int = 4
    phase1_weight_decay: float = 0.01

    # ---- Phase 2 (QFormer + projections on full prompt) ----------------------
    # Observed on the full run: lr 1e-3 reached +1.3 val AUC over the zero-token
    # reference by step 500, then declined monotonically (bridge overfit). Slow
    # it down, regularize, and sample the early peak densely.
    phase2_lr: float = 5e-4
    phase2_weight_decay: float = 0.05
    phase2_eval_every_steps: int = 250   # phase 2's peak is early and narrow
    # dense early sampling: eval every `dense_every` steps until `dense_until`,
    # then fall back to phase2_eval_every_steps. The peak is early (observed at
    # steps ~50-500), but 50-step evals cost ~3.4min each for ~15s of training
    # (~93% eval wall-clock) AND shrink patience to 350 steps (patience counts
    # EVALS). 150 keeps 6 early looks while patience covers >=1050 steps.
    phase2_dense_eval_until: int = 900
    phase2_dense_eval_every: int = 150
    phase2_epochs: int = 3
    phase2_batch_size: int = 8
    phase2_grad_accum: int = 4
    unfreeze_sasrec: bool = False    # Phase 2b: also fine-tune SASRec
    sasrec_lr_2b: float = 1e-4       # 2b: SASRec fine-tunes 5x slower than the
                                     # bridge so it adapts, not forgets, Phase 0

    # ---- Phase 3 (optional joint co-adaptation) ------------------------------
    # In Phase 2 the LoRA is frozen, so it never learns to READ the (now
    # informative) soft tokens — it only ever saw zero tokens in Phase 1. This
    # short final phase unfreezes LoRA AND QFormer together, from their trained
    # inits, at a low lr. NOTE this is a *co-adaptation from good inits*, which
    # is standard and safe — NOT the from-scratch joint LoRA+QFormer path the
    # spec warns against (that fails because LoRA gradient dominates a random
    # mapping early). Default off.
    phase3_joint_finetune: bool = False
    phase3_lr: float = 5e-5          # low: both modules already near-good
    phase3_epochs: int = 1
    users_per_batch: int = 4         # group sampler: users per batch (so same-user pairs exist)

    # ---- checkpoint selection ------------------------------------------------
    # Eval cadence is a wall-clock lever at 7B scale: at eval_every=200 with the
    # full 10.4k-row val split, evaluation was ~85% of Phase-1/2 runtime (10.4k
    # eval forwards after every 1.6k training samples). 500 steps x a 300-user
    # val subsample keeps ~25 selection points over 3 epochs (plenty for the
    # smoothing window, top-k soup, and patience) at ~1/3 of the wall-clock.
    # Per-epoch evals would be too coarse: 3 points can't feed any of them.
    # Selection primary metric. CoLLM/BinLLM select checkpoints by val AUC
    # (verified in their code: agg_metrics = auc; uauc only logged). We select
    # on AUC too but add a GUARD: within the noise band ties break by the guard
    # metric, and greedy-soup members are admitted only if the primary improves
    # AND the guard does not drop by more than sel_guard_tol.
    sel_metric: str = "auc"          # "auc" | "uauc"; the other becomes the guard
    sel_guard_tol: float = 0.003
    log_every_steps: int = 50        # print running train loss every this many steps
    eval_every_steps: int = 500      # evaluate val AUC/UAUC every this many steps
    sel_window: int = 5              # moving-average window for smoothed val UAUC —
                                     # val UAUC noise is ~±0.025 on ML-1M's 282
                                     # qualifying users; window 3 under-smoothed
    top_k_soup: int = 3              # weight-average the top-k checkpoints (model soup)
    # patience is in EVALS on the smoothed primary metric. History: 6 was too
    # tight (killed Phase 2 after ~300 optimizer updates, before the bridge
    # learned anything at the old 200-step cadence); 12 let declining runs
    # burn hours past their peak. 7 at the current cadences = ~3.5k Phase-1
    # steps / ~1.75k Phase-2 steps of no-improvement tolerance.
    patience: int = 7                # early stop after this many evals w/o smoothed gain
    n_boot: int = 1000               # bootstrap resamples (over users) for CIs

    # ---- evaluation -----------------------------------------------------------
    n_seeds: int = 3                 # seeds per reported config (mean +/- std)
    seed: int = 42
    ndcg_ks: tuple = (5, 10)

    # ---- engineering -----------------------------------------------------------
    out_dir: str = "checkpoints"
    device: str = ""                 # "" = auto-detect (cuda > mps > cpu)
    num_workers: int = 0
    val_subsample_users: int = 300   # selection evals score this fixed user subset
                                     # (0 = full val); test evaluation is never subsampled

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def collm(cls) -> "Config":
        """CoLLM / SeLLa-Rec reproduction protocol.

        Sources: CoLLM's released train_configs/collm_pretrain_sasrec_ood_cc.yaml
        (LoRA r8/a16 on q,v; init_lr 1e-3 cosine->8e-5, warmup 200; batch 16;
        200 epochs x 50 iters ~= 10k optimizer steps ~= 5 epochs of the 33.9k
        train rows; SASRec emb 64, 2 blocks, 4 heads, dropout 0.2, max_len 25;
        pure BCE; checkpoint = argmax raw val AUC) and the SeLLa-Rec paper
        (backbone re-based to Qwen2-7B BASE; same ood2 split and CoLLM eval).

        Documented deviations (things we cannot or should not match):
          * CoLLM's Vicuna numbers used Vicuna-7B v0 (LLaMA-1 deltas) — not
            reproducible today; this preset targets the SeLLa Qwen2 track.
          * Phase 0 still selects SASRec by val UAUC (our measured fix; their
            AUC-selection costs ~1.7pt UAUC downstream).
          * Selection evals use the 300-user val subsample (wall-clock); test
            scoring is never subsampled.
          * patience=20 as a runaway guard; their configs run all epochs.
        The SeLLa arm itself = this preset + sella_prealign/sella_warm_token.
        """
        return cls(
            backbone="Qwen/Qwen2-7B",
            max_his_len=25,
            sasrec_heads=4,
            sasrec_dropout=0.2,
            lambda_pair=0.0,             # CoLLM trains pure BCE
            lr_schedule="cosine",
            phase1_lr=1e-3, phase2_lr=1e-3,
            phase1_epochs=5, phase2_epochs=5,
            phase1_batch_size=16, phase1_grad_accum=1,
            phase2_batch_size=16, phase2_grad_accum=1,
            phase1_weight_decay=1e-3, phase2_weight_decay=1e-3,
            sel_window=1,                # their selection: argmax RAW val AUC
            top_k_soup=1,                # no soup
            patience=20,
            eval_every_steps=250, phase2_eval_every_steps=250,
            phase2_dense_eval_until=0, phase2_dense_eval_every=0,
        )

    @classmethod
    def smoke(cls) -> "Config":
        """Tiny everything: synthetic data + tiny-gpt2 so the pipeline runs on CPU in minutes."""
        # tiny-random-gpt2 (32-dim) over sshleifer/tiny-gpt2 (2-dim): wide enough
        # that soft tokens measurably influence P("Yes"), so the smoke test can
        # verify metrics actually move, not just that shapes line up
        return cls(
            smoke_test=True,
            backbone="hf-internal-testing/tiny-random-gpt2",
            max_his_len=10,          # synthetic histories are <= 10 anyway
            emb_dim=16,
            n_queries=2,
            qformer_layers=1,
            qformer_heads=2,
            phase0_epochs=3,
            phase0_batch_size=64,
            phase1_epochs=1,
            phase1_batch_size=4,
            phase1_grad_accum=1,
            phase2_epochs=2,
            phase2_batch_size=4,
            phase2_grad_accum=1,
            log_every_steps=10,
            eval_every_steps=20,
            phase2_eval_every_steps=20,
            sel_window=2,
            top_k_soup=2,
            patience=4,
            n_boot=200,
            n_seeds=2,
            num_workers=0,
        )

    def save(self, path: str | Path):
        d = asdict(self)
        d["lora_targets"] = list(d["lora_targets"])
        d["ndcg_ks"] = list(d["ndcg_ks"])
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text())
        d["lora_targets"] = tuple(d["lora_targets"])
        d["ndcg_ks"] = tuple(d["ndcg_ks"])
        return cls(**d)
