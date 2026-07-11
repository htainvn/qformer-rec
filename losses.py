"""Combined loss: pointwise BCE + within-user pairwise ranking.

UAUC is a WITHIN-user ordering metric: it only cares whether a user's positives
score above that same user's negatives. Pointwise BCE optimizes a global
calibration objective and can be minimized while leaving within-user order
wrong (e.g. by exploiting user-level base rates). The pairwise term forms every
(pos, neg) pair from the SAME user inside the batch and penalizes ŷ_pos < ŷ_neg
— a direct surrogate for per-user AUC. It relies on the UserGroupedBatchSampler
to make such pairs exist; with random batches it silently sees ~no pairs.
"""

import torch
import torch.nn.functional as F

EPS = 1e-7


def bce_loss(p_yes: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """p_yes is already a probability (softmax over Yes/No logits), so use log-prob BCE."""
    p = p_yes.clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(p, label)


def pairwise_loss(p_yes: torch.Tensor, label: torch.Tensor, uid: torch.Tensor,
                  margin: float = 0.0) -> torch.Tensor:
    """BPR over same-user (pos, neg) pairs found inside the batch.

    margin == 0 -> softplus(-(s_pos - s_neg))  (classic BPR, smooth)
    margin  > 0 -> relu(margin - (s_pos - s_neg))  (hinge)
    Returns 0 when the batch happens to contain no same-user pair.
    """
    # work in logit space so the loss keeps gradient signal near p ~ 0/1
    s = torch.logit(p_yes.clamp(EPS, 1 - EPS))
    same_user = uid.unsqueeze(0) == uid.unsqueeze(1)                  # [B, B]
    pos_neg = (label.unsqueeze(1) > 0.5) & (label.unsqueeze(0) < 0.5)  # row=pos, col=neg
    pairs = same_user & pos_neg
    if not pairs.any():
        return s.new_zeros(())
    diff = s.unsqueeze(1) - s.unsqueeze(0)                            # s_pos - s_neg
    d = diff[pairs]
    if margin > 0:
        return F.relu(margin - d).mean()
    return F.softplus(-d).mean()


def combined_loss(p_yes, label, uid, lambda_pair: float = 0.5, margin: float = 0.0):
    """Returns (total, bce, pair) — the parts are logged separately in training."""
    bce = bce_loss(p_yes, label)
    pair = pairwise_loss(p_yes, label, uid, margin=margin)
    return bce + lambda_pair * pair, bce.detach(), pair.detach()
