# Models

## Design 1 (implemented): SASRec → target-conditioned QFormer → LLM

```
history ids ──> SASRec (frozen after Phase 0) ──> H ∈ [B, L, d]   (ALL positions)
                                                    │ keys/values
target id ──> e_i ──> FiLM(γ, β) ──> Q' = γ(e_i)·Q + β(e_i)
                                                    │ queries
                                    QFormer (2 layers, self+cross attn)
                                                    │
                              N <UserID> tokens + 1 <TargetItemID> token
                                                    │ spliced into inputs_embeds
                     frozen LLM + LoRA ──> P("Yes") at the answer position
```

- **Why QFormer over CoLLM's MLP:** the MLP maps one pooled vector (SASRec's
  last state) to one soft token — a hard information bottleneck. The QFormer
  cross-attends over *every* history position, so the LLM receives N tokens
  distilled from the full sequence.
- **Target conditioning (FiLM):** queries are modulated by the target item
  embedding *before* cross-attention, so the bridge reads out "how does this
  history relate to items like this candidate" rather than a generic profile.
  γ is parameterized as `1 + Δγ` with `Δγ` initialized to zero → training starts
  exactly at the target-agnostic model and learns how much conditioning to use.
  Set `target_aware=False` for the agnostic control baseline.
- **Two-level target injection (by design):** early (FiLM in the bridge) and
  late (title + projected item token in the prompt). Early shapes *what is read
  out of the history*; late lets the LLM do the final semantic matching.

## Design 2 (implemented — `cfg.design2=True`): DIN-fused values

The fallback for Design 1 plateauing below the UAUC target — which happened at
~0.695 val UAUC. Now implemented: `models/din.py` (DINEncoder + FusedEncoder),
`train_phase0_din`, and `kv_dim` support in the QFormer.

**Measured go/no-go on real ML-1M (val, UAUC-selected):** DIN standalone
0.6766 UAUC vs SASRec 0.6763 — identical within noise — and their score blend
is WORSE (0.6744) on UAUC while better (+1.4pts) on AUC. The two encoders'
within-user signals are the same signal; ML-1M's 34k interactions appear to
saturate at ~0.68 collaborative UAUC. Design 2 is therefore NOT expected to
lift UAUC on this dataset; it remains useful on datasets with richer
per-user interaction data.

Keep SASRec **and** add a DIN encoder (Deep Interest Network). DIN computes
target-aware attention weights over history items; instead of using its pooled
output, expose its **pre-pool, target-weighted per-position states**
`D ∈ [B, L, d_din]` (each history position's embedding scaled by its relevance
to the target). Concatenate per position with SASRec's states:

```
K = V = concat([H_sasrec, D_din], dim=-1) ∈ [B, L, d + d_din]
```

and let a **target-agnostic** QFormer (no FiLM — the target information is
already inside the values) attend over this fused sequence.

Why the ceiling is higher: Design 1 injects the target only through low-rank
FiLM modulation of N small queries; Design 2 injects it multiplicatively at
*every history position* before the bridge, which is strictly more expressive
(DIN-style local relevance weighting composed with SASRec's sequential
encoding). Why the cost is higher: a second encoder to pre-train (or co-train)
per forward pass, double-width keys/values, and one more model to checkpoint,
ablate, and seed-sweep.

Migration path if needed: pre-train DIN with the same Phase-0 CTR objective,
freeze both encoders, widen `QFormerLayer`'s cross-attention `kdim/vdim` to
`d + d_din`, set `target_aware=False`, and rerun Phase 2 only.

## SeLLa-Rec arm (arXiv:2504.10107) — semantic alignment, recast around the QFormer

SeLLa-Rec's three transferable ideas, adapted so the QFormer plays the role of
its projection layer (flags in `config.py`, all off by default):

1. **Semantic distillation** (`build_semantic_vectors`): after Phase 1, each
   item's semantic vector `e_i^L` = the LoRA-tuned LLM's **last hidden state**
   over the quoted title — the OUTPUT space the task-adapted reader reasons
   in, unlike `build_title_vectors`' frozen input-embedding mean.
2. **Stage-2 contrastive pre-alignment** (`sella_prealign`): before Phase 2,
   InfoNCE pulls `item_proj(e_i)` toward `sem[target]` and the mean user token
   toward the history-mean of `sem` (the user side is our QFormer-role
   extension — MF user embeddings have no text, a title history does). An MSE
   anchor keeps token norms at semantic-embedding scale (cosine InfoNCE is
   norm-blind; out-of-scale tokens are the documented Phase-2 failure mode).
   This is SeLLa's warm-started projection; it deliberately trades away the
   zero-init no-op start, and the Phase-2 zero-token floor still guarantees
   the final model never regresses below Phase 1.
3. **`<WarmID>` token** (`sella_warm_token`): a third soft token
   `warm_proj(sem[target])` spliced after `<TargetItemID>` ("and the semantic
   feature ..."), zero-init so Phase 2 starts as a no-op. Phase 1's template
   mix includes the 3-slot prompt with a zero warm token, so there is no
   template-shift penalty. `warm_proj` stays untrained until Phase 2 (SeLLa
   keeps `Proj^(W→L)` random until its Stage 3).

Stage mapping: SeLLa Stage 1 (LoRA task adaptation) = Phase 1; Stage 2
(collab-semantic alignment) = `sella_prealign`; Stage 3 (projection training,
LLM+LoRA frozen) = Phase 2. `sella_anchor` additionally switches the
`align_titles` Phase-2 auxiliary anchor to the distilled vectors. The
faithful-to-paper baseline is `bridge="mlp"` + `sella_prealign` +
`sella_warm_token`; the same flags on `bridge="qformer"` are the paper's ideas
composed with the N-token cross-attentive bridge.

## Measured negative results (ML-1M, val, UAUC-selected — so nobody re-tries them blind)

| Idea | Result | Verdict |
|---|---|---|
| BPR pairwise term in Phase 0 | 0.6709–0.6742 vs 0.6763 baseline | no gain |
| 3-seed SASRec score ensemble | 0.6769 vs 0.6763 | noise |
| DIN encoder (Design 2 go/no-go) | 0.6766 vs 0.6763; blend 0.6744 | same signal as SASRec |
| AR next-item pretrain -> CTR finetune | 0.6962/0.6673 vs 0.6966/0.6763 | transfers nothing (601 users, mean seq 27) |

Collaborative UAUC on this dataset saturates at ~0.68 across every encoder and
training recipe tried; the reader (LLM) side is where the remaining headroom lives.
