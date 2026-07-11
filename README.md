# SASRec → target-conditioned QFormer → LLM recommender

An LLM recommender that injects collaborative information into a frozen LLM through a
QFormer bridge — extending [CoLLM (arXiv:2310.19488)](https://arxiv.org/abs/2310.19488)
by replacing its MLP mapping module with a **target-conditioned QFormer** over SASRec's
full sequence of hidden states. Task: binary click/rating prediction ("Yes"/"No") on
CoLLM's preprocessed MovieLens-1M. **Primary metric: UAUC** (per-user AUC averaged over
users); secondary: global AUC, NDCG@k.

Targets on ML-1M: UAUC ≥ 0.71, AUC ≥ 0.75, vs published CoLLM-SASRec 0.6990/0.7235,
BinLLM 0.6956/0.7425, CoLLM-MF 0.6875. Every reported number is a multi-seed mean with a
bootstrap CI over users; baseline wins are claimed via a paired per-user bootstrap test.

## Layout

```
config.py           central dataclass config (Config() full run, Config.smoke() CPU test)
data.py             CoLLM ML-1M pickles, synthetic fallback, user-grouped batch sampler
models/sasrec.py    SASRec backbone; exposes ALL per-position hidden states H [B,L,d]
models/qformer.py   QFormer bridge; FiLM target conditioning (Q' = γ(e_i)·Q + β(e_i))
models/llm_rec.py   frozen LLM + LoRA; soft tokens spliced into inputs_embeds
models/README.md    Design 2 (DIN-fused values) — documented fallback, not implemented
losses.py           BCE + within-user pairwise BPR (targets UAUC directly)
selection.py        smoothed-UAUC selection, bootstrap noise band, top-k model soup
train.py            Phase 0 (SASRec) → 1 (LoRA warm-up) → 2/2b (QFormer)
evaluate.py         AUC/UAUC, bootstrap CIs, paired per-user test, NDCG, stratified diag
run.ipynb           end-to-end notebook (defaults to smoke_test; full-run cell marked)
build_notebook.py   regenerates run.ipynb
```

## Run on Google Colab

1. Open [run.ipynb](run.ipynb) in Colab: `File → Open notebook → GitHub → htainvn/qformer-rec`
   (or use `https://colab.research.google.com/github/htainvn/qformer-rec/blob/main/run.ipynb`).
2. Pick a GPU runtime (`Runtime → Change runtime type`). T4 works with `cfg.load_4bit=True`;
   an A100 runs Vicuna-7B in bf16 directly.
3. Run the first code cell — it clones this repo, installs `peft`/`accelerate`/`bitsandbytes`,
   and unzips `ml-1m.zip`. Then run the rest top-to-bottom.
4. For the real numbers, enable the marked **FULL RUN** cell (Vicuna-7B, multi-seed) instead
   of the default smoke config.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# CPU smoke test (synthetic data + tiny random GPT-2, a few minutes):
.venv/bin/python train.py --smoke_test
# full run (needs GPU; expects CoLLM pickles under ml-1m/):
.venv/bin/python train.py
# notebook:
.venv/bin/jupyter nbconvert --to notebook --execute --inplace run.ipynb
```

Data: place CoLLM's `train_ood2.pkl` / `valid_ood2.pkl` / `test_ood2.pkl` under `ml-1m/`
(from [github.com/zyang1580/CoLLM](https://github.com/zyang1580/CoLLM)). If absent, a
synthetic dataset with the same schema is generated so the pipeline still runs.

## Training phases

| Phase | Trains | Frozen | Prompt |
|---|---|---|---|
| 0 | SASRec (CTR, BCE) | — | — |
| 1 | LoRA only | LLM, SASRec, QFormer | text-only (no ID fields) |
| 2 | QFormer + projections | LLM, LoRA, SASRec | full hybrid |
| 2b (`unfreeze_sasrec`) | QFormer + SASRec | LLM, LoRA | full hybrid |

A joint one-step LoRA+QFormer path is deliberately not offered (underperforms, esp. cold
users). Optional `qformer_align_pretrain` contrastive init before Phase 2.
