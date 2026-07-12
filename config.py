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
    prompt_titles: int = 10          # titles listed in the prompt text (template cap)
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
    n_queries: int = 4               # N learnable queries -> N <UserID> soft tokens
    qformer_layers: int = 2
    qformer_heads: int = 4
    qformer_dropout: float = 0.1
    target_aware: bool = True        # FiLM-condition queries on target item embedding
    qformer_align_pretrain: bool = False  # optional contrastive alignment before Phase 2
    align_epochs: int = 3
    align_lr: float = 1e-3

    # ---- LLM ----------------------------------------------------------------
    backbone: str = "lmsys/vicuna-7b-v1.5"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "v_proj")
    load_4bit: bool = False          # set True to QLoRA-quantize the frozen backbone

    # ---- loss ---------------------------------------------------------------
    lambda_pair: float = 0.5         # weight on within-user pairwise (BPR) ranking term
    pair_margin: float = 0.0         # 0.0 -> plain BPR softplus; >0 -> margin hinge

    # ---- Phase 1 (LoRA warm-up, text-only prompt) ---------------------------
    # epochs are an UPPER BOUND — patience early-stopping on smoothed val UAUC
    # is the real terminator. The observed 1-epoch run ended with the val curve
    # still rising (UAUC 0.6887->peak 0.6916, AUC still climbing), i.e. the
    # epoch limit cut training short, not the data.
    phase1_lr: float = 1e-4
    phase1_epochs: int = 3
    phase1_batch_size: int = 8
    phase1_grad_accum: int = 4

    # ---- Phase 2 (QFormer + projections on full prompt) ----------------------
    phase2_lr: float = 1e-3          # the 4.4M zero-init bridge learns through a
                                     # frozen 7B; 5e-4 trained too slowly to trend
    phase2_epochs: int = 3
    phase2_batch_size: int = 8
    phase2_grad_accum: int = 4
    unfreeze_sasrec: bool = False    # Phase 2b: also fine-tune SASRec
    users_per_batch: int = 4         # group sampler: users per batch (so same-user pairs exist)

    # ---- checkpoint selection ------------------------------------------------
    # Eval cadence is a wall-clock lever at 7B scale: at eval_every=200 with the
    # full 10.4k-row val split, evaluation was ~85% of Phase-1/2 runtime (10.4k
    # eval forwards after every 1.6k training samples). 500 steps x a 300-user
    # val subsample keeps ~25 selection points over 3 epochs (plenty for the
    # smoothing window, top-k soup, and patience) at ~1/3 of the wall-clock.
    # Per-epoch evals would be too coarse: 3 points can't feed any of them.
    log_every_steps: int = 50        # print running train loss every this many steps
    eval_every_steps: int = 500      # evaluate val AUC/UAUC every this many steps
    sel_window: int = 5              # moving-average window for smoothed val UAUC —
                                     # val UAUC noise is ~±0.025 on ML-1M's 282
                                     # qualifying users; window 3 under-smoothed
    top_k_soup: int = 3              # weight-average the top-k checkpoints (model soup)
    # patience is in EVALS: with eval_every_steps=200 and grad_accum=4 that is
    # only 50 optimizer updates per eval — 6 evals killed Phase 2 after ~300
    # updates on the observed run, before the QFormer had learned anything
    patience: int = 12               # early stop after this many evals w/o smoothed-UAUC gain
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
