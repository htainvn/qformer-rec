"""SASRec collaborative backbone (Kang & McAuley 2018), CTR-style.

Two things distinguish this implementation from a stock SASRec:
  1. `encode_history` returns the FULL sequence of hidden states H in [B, L, d]
     (not just the last position) — the QFormer bridge cross-attends over every
     position, so collapsing to the last state would throw away exactly the
     information the bridge is meant to extract.
  2. The padding embedding (index 0) is frozen at zero: it is zeroed at init and
     re-zeroed after every optimizer step via a gradient hook, so left-padding
     never leaks signal into attention values.
"""

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, n_items: int, emb_dim: int = 64, max_len: int = 10,
                 n_blocks: int = 2, n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.max_len = max_len

        # +1 because ids run 1..n_items with 0 = pad
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.final_norm = nn.LayerNorm(emb_dim)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()  # padding_idx also blocks its gradient

    def encode_history(self, his: torch.Tensor, his_mask: torch.Tensor):
        """his: [B, L] item ids (LEFT-padded with 0); his_mask: [B, L] 1=real.

        Returns H: [B, L, d] — full per-position hidden states. Padding positions
        are zeroed in the output; the caller masks them in cross-attention anyway.
        """
        B, L = his.shape
        pos = torch.arange(L, device=his.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(his) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        # Merge the causal mask (position t sees only <= t) with the padding mask
        # into ONE per-sample [L, L] mask, then force the diagonal open. With
        # LEFT padding a pad-position query would otherwise see zero unmasked
        # keys (its only causal keys are pads), and softmax over all -inf is
        # NaN — which survives the final `H * his_mask` because NaN * 0 = NaN.
        # Letting every position at least attend to itself makes the output
        # finite everywhere; pad positions are zeroed out below regardless.
        causal = torch.triu(torch.ones(L, L, device=his.device, dtype=torch.bool), diagonal=1)
        key_pad = his_mask == 0                              # [B, L] True = pad
        attn_mask = causal.unsqueeze(0) | key_pad.unsqueeze(1)   # [B, L, L]
        attn_mask = attn_mask & ~torch.eye(L, device=his.device, dtype=torch.bool)
        # TransformerEncoder wants per-head masks: [B * n_heads, L, L]
        n_heads = self.blocks.layers[0].self_attn.num_heads
        attn_mask = attn_mask.repeat_interleave(n_heads, dim=0)

        h = self.blocks(x, mask=attn_mask)
        h = self.final_norm(h)
        h = h * his_mask.unsqueeze(-1)                       # zero out pad positions
        return h

    def item_embedding(self, iid: torch.Tensor) -> torch.Tensor:
        """Raw item embedding by id — used by the QFormer for FiLM conditioning
        and by the CTR head below."""
        return self.item_emb(iid)

    def last_state(self, H: torch.Tensor, his_mask: torch.Tensor) -> torch.Tensor:
        """State at the last REAL position (left-padded, so it's just index -1
        unless the history is empty, in which case it's a zero vector)."""
        return H[:, -1, :]

    def ctr_logit(self, his, his_mask, iid) -> torch.Tensor:
        """Phase-0 pretraining head: dot(last state, target embedding)."""
        H = self.encode_history(his, his_mask)
        u = self.last_state(H, his_mask)                     # [B, d]
        v = self.item_embedding(iid)                         # [B, d]
        return (u * v).sum(-1)
