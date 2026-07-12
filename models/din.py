"""Design 2: DIN encoder + fused keys/values for the QFormer.

DIN (Deep Interest Network, Zhou et al. 2018) computes a target-aware attention
weight for every history position via a "local activation unit" — an MLP over
[h_j, e_t, h_j * e_t, h_j - e_t]. Standard DIN sum-pools the weighted history;
we instead EXPOSE THE PRE-POOL weighted per-position states D in [B, L, d], as
the design doc specifies, and concatenate them with SASRec's sequential states
as the QFormer's keys/values:

    K = V = concat([H_sasrec, D_din], dim=-1) in [B, L, 2d]

Why this has a higher ceiling than Design 1's FiLM: FiLM injects the target
through a low-rank modulation of N small queries; DIN injects it
multiplicatively at EVERY history position before the bridge — per-position
relevance weighting composed with sequential encoding. The QFormer queries can
then stay target-agnostic (`target_aware=False`): the target information is
already inside the values.
"""

import torch
import torch.nn as nn


class DINEncoder(nn.Module):
    def __init__(self, n_items: int, emb_dim: int = 64, dropout: float = 0.3,
                 att_hidden: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        # local activation unit: score(h_j, e_t) from rich pairwise features
        self.att = nn.Sequential(
            nn.Linear(4 * emb_dim, att_hidden), nn.PReLU(),
            nn.Linear(att_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        # CTR head for standalone pre-training: [pooled, e_t, pooled*e_t] -> logit
        self.head = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim), nn.PReLU(),
            nn.Dropout(dropout), nn.Linear(emb_dim, 1),
        )

    def _weights(self, h: torch.Tensor, e_t: torch.Tensor, mask: torch.Tensor):
        """Per-position relevance of each history item to the target.
        h: [B, L, d], e_t: [B, d], mask: [B, L] -> [B, L, 1]"""
        e = e_t.unsqueeze(1).expand_as(h)
        feats = torch.cat([h, e, h * e, h - e], dim=-1)
        w = self.att(feats)                                   # [B, L, 1]
        # DIN convention: mask pads, no softmax — absolute relevance intensity
        # (softmax would force weights to sum to 1 even for all-junk histories)
        return w.masked_fill(mask.unsqueeze(-1) == 0, 0.0)

    def weighted_states(self, his: torch.Tensor, his_mask: torch.Tensor,
                        iid: torch.Tensor) -> torch.Tensor:
        """The pre-pool target-weighted per-position states D in [B, L, d]."""
        h = self.dropout(self.item_emb(his))
        e_t = self.item_emb(iid)
        w = self._weights(h, e_t, his_mask)
        return w * h * his_mask.unsqueeze(-1)

    def ctr_logit(self, his, his_mask, iid) -> torch.Tensor:
        """Standalone CTR objective for Phase-0-style pre-training."""
        D = self.weighted_states(his, his_mask, iid)          # [B, L, d]
        pooled = D.sum(dim=1)                                 # DIN sum-pool
        e_t = self.item_emb(iid)
        return self.head(torch.cat([pooled, e_t, pooled * e_t], dim=-1)).squeeze(-1)


class FusedEncoder(nn.Module):
    """SASRec + DIN, exposing fused per-position keys/values for the QFormer.

    Drop-in for the pipeline's `sasrec` slot: it delegates item_embedding /
    ctr_logit, and adds `encode_history_target` returning [B, L, 2d]. The
    training/eval loops dispatch on that method's presence.
    """

    def __init__(self, sasrec: nn.Module, din: DINEncoder):
        super().__init__()
        self.sasrec = sasrec
        self.din = din
        self.kv_dim = sasrec.emb_dim + din.emb_dim

    def encode_history_target(self, his, his_mask, iid) -> torch.Tensor:
        H = self.sasrec.encode_history(his, his_mask)         # [B, L, d]
        D = self.din.weighted_states(his, his_mask, iid)      # [B, L, d]
        return torch.cat([H, D], dim=-1)                      # [B, L, 2d]

    def item_embedding(self, iid):
        return self.sasrec.item_embedding(iid)

    def ctr_logit(self, his, his_mask, iid):
        # average the two pre-trained heads — used only for score-level
        # baselines (headroom / late fusion), never inside the bridge
        s = torch.sigmoid(self.sasrec.ctr_logit(his, his_mask, iid))
        d = torch.sigmoid(self.din.ctr_logit(his, his_mask, iid))
        p = ((s + d) / 2).clamp(1e-7, 1 - 1e-7)
        return torch.log(p / (1 - p))
