"""QFormer bridge: N learnable queries distill SASRec's per-position history
states into N soft tokens for the LLM.

Why a QFormer instead of CoLLM's MLP mapping? CoLLM compresses the whole history
into ONE vector (SASRec's last state) and MLP-maps it to one soft token. The
QFormer instead cross-attends over ALL history positions, so different queries
can specialize (genres, recency, niche taste) and the LLM receives N tokens of
user representation instead of one bottlenecked vector.

TARGET CONDITIONING (the key extension): the queries are FiLM-modulated by the
target item embedding e_i BEFORE cross-attention:

    Q' = gamma(e_i) * Q + beta(e_i)

so the bridge extracts evidence about THIS candidate from the history — "how does
this user relate to items like e_i" — rather than a one-size-fits-all summary.
This mirrors DIN's target-aware attention, but implemented at the query level so
the SASRec encoder itself stays target-agnostic (and cacheable). gamma is
parameterized as 1 + dgamma with dgamma initialized near zero, so training starts
at the identity (target-agnostic) and learns how much conditioning helps.

The target therefore enters the model TWICE by design: early here (shaping what
gets read out of the history) and late in the prompt (title + projected item
token, letting the LLM do the final matching).
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Small MLP producing per-channel (gamma, beta) from the target embedding."""

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        # start at identity: gamma = 1 + dgamma ~ 1, beta ~ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, queries: torch.Tensor, e_i: torch.Tensor) -> torch.Tensor:
        # queries: [B, N, h], e_i: [B, emb_dim]
        dgamma, beta = self.net(e_i).chunk(2, dim=-1)        # each [B, h]
        return (1.0 + dgamma).unsqueeze(1) * queries + beta.unsqueeze(1)


class QFormerLayer(nn.Module):
    """One block = self-attention over queries + cross-attention (queries -> H) + FFN."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, kv_pad_mask):
        # pre-norm residual blocks
        x = self.norm1(q)
        q = q + self.dropout(self.self_attn(x, x, x, need_weights=False)[0])
        x = self.norm2(q)
        q = q + self.dropout(self.cross_attn(x, kv, kv, key_padding_mask=kv_pad_mask,
                                             need_weights=False)[0])
        q = q + self.dropout(self.ffn(self.norm3(q)))
        return q


class QFormerBridge(nn.Module):
    def __init__(self, emb_dim: int = 64, llm_dim: int = 4096, n_queries: int = 4,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1,
                 target_aware: bool = True):
        super().__init__()
        self.target_aware = target_aware
        self.n_queries = n_queries
        hidden = emb_dim  # run the bridge at SASRec width; project up only at the end

        self.queries = nn.Parameter(torch.empty(1, n_queries, hidden))
        nn.init.normal_(self.queries, std=0.02)

        self.film = FiLM(emb_dim, hidden) if target_aware else None
        self.layers = nn.ModuleList(
            QFormerLayer(hidden, n_heads, dropout) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(hidden)

        # N query outputs -> N <UserID> soft tokens in LLM space
        self.user_proj = nn.Sequential(
            nn.Linear(hidden, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))
        # separate MLP: raw target embedding e_i -> 1 <TargetItemID> soft token
        self.item_proj = nn.Sequential(
            nn.Linear(emb_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))

    def forward(self, H: torch.Tensor, his_mask: torch.Tensor, e_i: torch.Tensor):
        """H: [B, L, d] SASRec states; his_mask: [B, L] 1=real; e_i: [B, d].

        Returns (user_tokens [B, N, llm_dim], item_token [B, 1, llm_dim]).
        """
        B = H.size(0)
        q = self.queries.expand(B, -1, -1)

        if self.film is not None:
            # FiLM BEFORE cross-attention: make the readout target-specific
            q = self.film(q, e_i)

        kv_pad = his_mask == 0                              # True = ignore this key
        all_pad = kv_pad.all(dim=1)
        if all_pad.any():                                   # empty history -> attend to
            kv_pad = kv_pad.clone()                         # zero states instead of NaN
            kv_pad[all_pad] = False

        for layer in self.layers:
            q = layer(q, H, kv_pad)
        q = self.final_norm(q)

        user_tokens = self.user_proj(q)                     # [B, N, llm_dim]
        item_token = self.item_proj(e_i).unsqueeze(1)       # [B, 1, llm_dim]
        return user_tokens, item_token

    def align_scores(self, H, his_mask, e_i_matrix):
        """Optional contrastive pre-training helper (`qformer_align_pretrain`):
        score every history against every candidate embedding in the batch.
        Uses mean-pooled query outputs at bridge width (before LLM projection).
        e_i_matrix: [B, d] -> returns [B, B] logits (row = user, col = item)."""
        B = H.size(0)
        outs = []
        for j in range(B):  # condition each user's queries on each candidate
            e_j = e_i_matrix[j].unsqueeze(0).expand(B, -1)
            q = self.queries.expand(B, -1, -1)
            if self.film is not None:
                q = self.film(q, e_j)
            kv_pad = his_mask == 0
            all_pad = kv_pad.all(dim=1)
            if all_pad.any():
                kv_pad = kv_pad.clone(); kv_pad[all_pad] = False
            for layer in self.layers:
                q = layer(q, H, kv_pad)
            u = self.final_norm(q).mean(dim=1)              # [B, hidden]
            outs.append((u * e_i_matrix[j]).sum(-1))        # dot with candidate
        return torch.stack(outs, dim=1)                     # [B, B]
