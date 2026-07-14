"""Robust checkpoint selection, metric-generic (primary + guard).

CoLLM/BinLLM select checkpoints by raw val AUC (their code: agg_metrics = auc;
uauc only logged). We keep AUC-primary comparability but defend both metrics:

  1. SMOOTHING — the selection signal is a trailing moving average of the
     PRIMARY metric over `sel_window` evals, killing single-eval spikes.
  2. NOISE BAND — an exact bootstrap CI (resampling users on the best point's
     stored raw scores) around the best smoothed primary defines a band;
     checkpoints inside it are ties, broken by the GUARD metric. The guard can
     never override a real primary gap.
  3. GREEDY SOUP — weight-average grown one checkpoint at a time; a candidate
     is admitted only if the measured primary improves AND the guard does not
     drop by more than `guard_tol`. Provably no worse than the best single
     checkpoint on the primary, and guard-protected by construction.

Also provides patience-based early stopping on the smoothed primary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from evaluate import per_user_auc, auc as auc_fn, uauc as uauc_fn


@dataclass
class EvalPoint:
    step: int
    uauc: float
    auc: float
    smoothed: float           # trailing mean of the PRIMARY metric
    uids: np.ndarray          # raw val arrays -> exact bootstrap for any metric
    labels: np.ndarray
    scores: np.ndarray
    state: dict               # cpu state_dict of the trainable parts (or None if pruned)

    def metric(self, name: str) -> float:
        return self.auc if name == "auc" else self.uauc


@dataclass
class CheckpointSelector:
    sel_window: int = 3
    top_k: int = 3
    patience: int = 6
    n_boot: int = 1000
    seed: int = 0
    sel_metric: str = "uauc"      # primary; the other of auc/uauc is the guard
    guard_tol: float = 0.003
    keep_states: int = 12   # states are big (QFormer->4096 heads); metrics are kept
    history: list = field(default_factory=list)  # for ALL points, states only for
    _stale: int = 0                              # the top `keep_states` candidates

    @property
    def guard_metric(self) -> str:
        return "uauc" if self.sel_metric == "auc" else "auc"

    def update(self, step: int, uids, labels, scores, users, state: dict) -> bool:
        """Record one evaluation. Returns True if training should STOP (patience
        exhausted on the smoothed primary)."""
        pu = per_user_auc(uids, labels, scores, users)
        raw_uauc = float(np.mean([a for a, _ in pu.values()]))
        raw_auc = auc_fn(labels, scores)
        raw_primary = raw_auc if self.sel_metric == "auc" else raw_uauc
        window = [p.metric(self.sel_metric) for p in self.history[-(self.sel_window - 1):]]
        smoothed = float(np.mean(window + [raw_primary]))
        self.history.append(EvalPoint(
            step=step, uauc=raw_uauc, auc=raw_auc, smoothed=smoothed,
            uids=np.asarray(uids, dtype=np.int32),
            labels=np.asarray(labels, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            state={k: v.detach().cpu().clone() for k, v in state.items()}))
        print(f"[select] step {step}: val UAUC {raw_uauc:.4f} val AUC {raw_auc:.4f} "
              f"(smoothed {self.sel_metric} {smoothed:.4f})")

        best = max(p.smoothed for p in self.history)
        if smoothed >= best - 1e-9:
            self._stale = 0
        else:
            self._stale += 1

        # bound memory: drop the STATE (not the metrics) of checkpoints that can
        # no longer plausibly enter the soup. Curves and the band stay exact.
        with_state = [p for p in self.history if p.state is not None]
        if len(with_state) > self.keep_states:
            with_state.sort(key=lambda p: p.smoothed, reverse=True)
            for p in with_state[self.keep_states:]:
                p.state = None
        return self._stale >= self.patience

    # ------------------------------------------------------------------ #
    def _noise_band(self) -> float:
        """Half-width of the bootstrap CI (resampling USERS) of the PRIMARY
        metric on the best checkpoint's stored raw scores — the resolution
        below which two checkpoints are indistinguishable."""
        best = max(self.history, key=lambda p: p.smoothed)
        rng = np.random.default_rng(self.seed)
        users = np.unique(best.uids)
        rows_of = {u: np.flatnonzero(best.uids == u) for u in users}
        stats = []
        for _ in range(min(self.n_boot, 300)):   # exact resample; 300 is ample for a band
            take = np.concatenate([rows_of[u] for u in rng.choice(users, size=len(users))])
            l, s = best.labels[take], best.scores[take]
            if l.min() == l.max():
                continue
            stats.append(auc_fn(l, s) if self.sel_metric == "auc"
                         else uauc_fn(best.uids[take], l, s))
        lo, hi = np.quantile(stats, [0.025, 0.975])
        return float(hi - lo) / 2

    def rank(self) -> list[EvalPoint]:
        """Checkpoints best-first: smoothed primary desc; within the noise band
        ties break by the GUARD metric (never across a real primary gap)."""
        band = self._noise_band()
        pts = sorted((p for p in self.history if p.state is not None),
                     key=lambda p: p.smoothed, reverse=True)
        best_p = pts[0].smoothed
        in_band = [p for p in pts if best_p - p.smoothed <= band]
        out_band = [p for p in pts if best_p - p.smoothed > band]
        in_band.sort(key=lambda p: p.metric(self.guard_metric), reverse=True)
        print(f"[select] noise band ±{band:.4f} on {self.sel_metric}: "
              f"{len(in_band)} checkpoint(s) tied at top; ties broken by {self.guard_metric}")
        return in_band + out_band

    @staticmethod
    def _avg(states: list[dict]) -> dict:
        avg = {}
        for key in states[0].keys():
            avg[key] = (sum(s[key].float() for s in states) / len(states)
                        ).to(states[0][key].dtype)  # keep bf16 LoRA dtype
        return avg

    def soup(self, k: int | None = None) -> dict:
        """Blind uniform soup of the top-k (kept for reference; greedy preferred)."""
        k = k or self.top_k
        top = self.rank()[:k]
        print(f"[select] souping {len(top)} checkpoints from steps {[p.step for p in top]}")
        return self._avg([p.state for p in top])

    def greedy_soup(self, eval_fn, k: int | None = None) -> dict:
        """Guarded greedy soup. eval_fn(state) -> (primary, guard) measured on
        the selection val set. A candidate joins only if the primary improves
        AND the guard does not drop more than guard_tol below the best guard
        seen in the soup so far. Provably no worse than the best checkpoint on
        the primary; guard-protected by construction."""
        k = k or self.top_k
        cands = self.rank()[:k + 2]              # a couple of spares to try
        members = [cands[0].state]
        best_p, best_g = eval_fn(cands[0].state)
        print(f"[soup] base = step {cands[0].step}: "
              f"{self.sel_metric} {best_p:.4f} {self.guard_metric} {best_g:.4f}")
        for p in cands[1:]:
            if len(members) >= k:
                break
            trial = self._avg(members + [p.state])
            tp, tg = eval_fn(trial)
            keep = tp > best_p and tg >= best_g - self.guard_tol
            print(f"[soup] + step {p.step}: {self.sel_metric} {tp:.4f} "
                  f"{self.guard_metric} {tg:.4f} ({'kept' if keep else 'rejected'})")
            if keep:
                members.append(p.state)
                best_p, best_g = tp, max(best_g, tg)
        return self._avg(members) if len(members) > 1 else members[0]

    def best_state(self) -> dict:
        return copy.deepcopy(self.rank()[0].state)

    def curves(self):
        """(steps, raw_uauc, smoothed_primary, raw_auc) for plotting."""
        h = self.history
        return ([p.step for p in h], [p.uauc for p in h],
                [p.smoothed for p in h], [p.auc for p in h])
