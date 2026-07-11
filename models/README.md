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

## Design 2 (documented only — NOT implemented): DIN-fused values

A fallback with a higher ceiling (and higher cost) if Design 1 plateaus below
the UAUC target.

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
