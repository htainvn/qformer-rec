"""Evaluation: AUC, UAUC (primary), bootstrap CIs, paired per-user comparison,
support-stratified diagnostics, NDCG@k, and the final comparison table.

Conventions that make these numbers defensible:
  * UAUC averages per-user AUC over users with BOTH classes present in the split.
    That qualifying-user set is computed ONCE (`qualifying_users`), logged, and
    reused identically across models and seeds — so no model gets a different
    denominator.
  * Every headline UAUC carries a bootstrap CI over USERS (users are the
    exchangeable unit; resampling rows would understate variance).
  * Beating a baseline is claimed via the PAIRED per-user test: mean per-user
    AUC difference with a bootstrap over users, not by comparing two marginal
    numbers.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Published ML-1M baselines (CoLLM paper, arXiv:2310.19488; BinLLM) and our targets.
BASELINES = {
    "CoLLM-SASRec": {"uauc": 0.6990, "auc": 0.7235},
    "BinLLM":       {"uauc": 0.6956, "auc": 0.7425},
    "CoLLM-MF":     {"uauc": 0.6875, "auc": None},
}
TARGETS = {"uauc": 0.71, "auc": 0.75}


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #

def qualifying_users(uids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Users with both a positive and a negative in this split — fixed once,
    used identically for every model/seed."""
    users = []
    for u in np.unique(uids):
        m = uids == u
        if labels[m].min() < 0.5 < labels[m].max() + 0.5 and len(np.unique(labels[m])) == 2:
            users.append(u)
    return np.asarray(users)


def per_user_auc(uids, labels, scores, users) -> dict[int, tuple[float, int]]:
    """{uid: (auc, support)} over the given qualifying users."""
    out = {}
    for u in users:
        m = uids == u
        out[int(u)] = (roc_auc_score(labels[m], scores[m]), int(m.sum()))
    return out


def uauc(uids, labels, scores, users=None) -> float:
    if users is None:
        users = qualifying_users(uids, labels)
    pu = per_user_auc(uids, labels, scores, users)
    return float(np.mean([a for a, _ in pu.values()]))


def auc(labels, scores) -> float:
    return float(roc_auc_score(labels, scores))


def bootstrap_uauc_ci(uids, labels, scores, users=None, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for UAUC, resampling USERS with replacement."""
    if users is None:
        users = qualifying_users(uids, labels)
    pu = per_user_auc(uids, labels, scores, users)
    aucs = np.array([pu[int(u)][0] for u in users])
    rng = np.random.default_rng(seed)
    stats = rng.choice(aucs, size=(n_boot, len(aucs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(aucs.mean()), float(lo), float(hi)


def paired_user_test(uids, labels, scores_a, scores_b, users=None,
                     n_boot=1000, seed=0):
    """Paired per-user comparison of model A vs model B on the SAME samples.

    Returns dict with mean per-user AUC difference (A - B), its bootstrap CI
    over users, and p (two-sided, share of bootstrap means crossing 0).
    This is the correct way to claim A beats B: the pairing removes the
    between-user variance that dominates marginal UAUC comparisons.
    """
    if users is None:
        users = qualifying_users(uids, labels)
    pa = per_user_auc(uids, labels, scores_a, users)
    pb = per_user_auc(uids, labels, scores_b, users)
    diffs = np.array([pa[int(u)][0] - pb[int(u)][0] for u in users])
    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_diff": float(diffs.mean()), "ci": (float(lo), float(hi)),
            "p_boot": float(min(p, 1.0)), "n_users": len(users)}


# --------------------------------------------------------------------------- #
# Diagnostics (reported alongside, never as headline)
# --------------------------------------------------------------------------- #

def support_weighted_uauc(uids, labels, scores, users=None) -> float:
    if users is None:
        users = qualifying_users(uids, labels)
    pu = per_user_auc(uids, labels, scores, users)
    a = np.array([v[0] for v in pu.values()]); w = np.array([v[1] for v in pu.values()])
    return float((a * w).sum() / w.sum())


def stratified_uauc(uids, labels, scores, users=None, min_supports=(5, 10)) -> dict:
    """UAUC restricted to users with >= k test interactions. Reported WITH the
    standard (all qualifying users) number — never instead of it."""
    if users is None:
        users = qualifying_users(uids, labels)
    pu = per_user_auc(uids, labels, scores, users)
    out = {}
    for k in min_supports:
        vals = [a for a, s in pu.values() if s >= k]
        out[f">={k}"] = (float(np.mean(vals)) if vals else float("nan"), len(vals))
    return out


def ndcg_at_k(uids, labels, scores, ks=(5, 10)) -> dict:
    """Per-user NDCG@k with binary gains over each user's test items, averaged
    over users with at least one positive."""
    out = {k: [] for k in ks}
    for u in np.unique(uids):
        m = uids == u
        y, s = labels[m], scores[m]
        if y.sum() == 0:
            continue
        order = np.argsort(-s)
        for k in ks:
            gains = y[order][:k]
            dcg = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
            ideal = np.sort(y)[::-1][:k]
            idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
            out[k].append(dcg / idcg if idcg > 0 else 0.0)
    return {k: float(np.mean(v)) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Scoring a model over a dataset
# --------------------------------------------------------------------------- #

@torch.no_grad()
def score_dataset(llm, sasrec, qformer, dataset, batch_size=16, device="cpu",
                  hybrid=True, progress=False, zero_soft_tokens=False):
    """Run the full pipeline over a dataset; returns (uids, labels, scores).

    zero_soft_tokens: keep the hybrid template but replace the QFormer outputs
    with zeros — the ablation control for "does the LLM read the soft tokens
    at all", scored on the identical prompts."""
    from torch.utils.data import DataLoader
    from data import collate
    llm.eval(); sasrec.eval(); qformer.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    uids, labels, scores = [], [], []
    for i, b in enumerate(dl):
        b = b.to(device)
        if hybrid:
            H = sasrec.encode_history(b.his, b.his_mask)
            e_i = sasrec.item_embedding(b.iid)
            u_tok, i_tok = qformer(H, b.his_mask, e_i)
            if zero_soft_tokens:
                u_tok, i_tok = torch.zeros_like(u_tok), torch.zeros_like(i_tok)
            p = llm(b.his_titles, b.target_titles, u_tok, i_tok)
        else:
            p = llm(b.his_titles, b.target_titles)
        uids.append(b.uid.cpu().numpy()); labels.append(b.label.cpu().numpy())
        scores.append(p.float().cpu().numpy())
        if progress and i % 50 == 0:
            print(f"  scored {i + 1}/{len(dl)} batches", flush=True)
    return np.concatenate(uids), np.concatenate(labels), np.concatenate(scores)


def full_report(uids, labels, scores, users=None, n_boot=1000, ndcg_ks=(5, 10), seed=0):
    """Everything about one run, as a dict."""
    if users is None:
        users = qualifying_users(uids, labels)
    u_mean, u_lo, u_hi = bootstrap_uauc_ci(uids, labels, scores, users, n_boot=n_boot, seed=seed)
    return {
        "auc": auc(labels, scores),
        "uauc": u_mean, "uauc_ci": (u_lo, u_hi),
        "n_qualifying_users": len(users),
        "support_weighted_uauc": support_weighted_uauc(uids, labels, scores, users),
        "stratified_uauc": stratified_uauc(uids, labels, scores, users),
        "ndcg": ndcg_at_k(uids, labels, scores, ks=ndcg_ks),
    }


def comparison_table(our_auc_mean, our_auc_std, our_uauc_mean, our_uauc_std,
                     uauc_ci, n_seeds) -> str:
    rows = [("Ours (SASRec+QFormer+LLM)",
             f"{our_uauc_mean:.4f} ± {our_uauc_std:.4f} (CI {uauc_ci[0]:.4f}–{uauc_ci[1]:.4f})",
             f"{our_auc_mean:.4f} ± {our_auc_std:.4f}",
             f"mean of {n_seeds} seeds")]
    for name, m in BASELINES.items():
        rows.append((name, f"{m['uauc']:.4f}",
                     f"{m['auc']:.4f}" if m["auc"] else "—", "published"))
    rows.append(("TARGET", f"{TARGETS['uauc']:.2f}", f"{TARGETS['auc']:.2f}", ""))
    w = [max(len(r[i]) for r in rows + [("Model", "UAUC", "AUC", "Notes")]) for i in range(4)]
    line = lambda r: " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r))
    sep = "-+-".join("-" * x for x in w)
    return "\n".join([line(("Model", "UAUC", "AUC", "Notes")), sep] + [line(r) for r in rows])
