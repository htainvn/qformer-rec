"""Robust checkpoint selection.

With only ~800 validation users, per-step val UAUC is noisy: picking the argmax
checkpoint mostly picks the luckiest evaluation, which does not transfer to test.
Three defenses, applied in order:

  1. SMOOTHING — the selection signal is a trailing moving average of val UAUC
     over `sel_window` evals, killing single-eval spikes.
  2. NOISE BAND — a bootstrap CI (over validation users) around the best smoothed
     UAUC defines a band; checkpoints inside the band are considered ties and the
     tie is broken by val AUC. AUC only ever breaks ties INSIDE the band — it can
     never override a real UAUC gap.
  3. MODEL SOUP — the final model is the WEIGHT AVERAGE of the top-k checkpoints
     by smoothed UAUC (LoRA + QFormer tensors averaged elementwise). Nearby
     fine-tuning optima are linearly connected, so the average sits in a flatter
     region of the loss surface and cancels per-step selection noise.

Also provides patience-based early stopping on the smoothed UAUC.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from evaluate import per_user_auc


@dataclass
class EvalPoint:
    step: int
    uauc: float
    auc: float
    smoothed_uauc: float
    per_user: dict            # {uid: (auc, support)} — kept for the noise band
    state: dict               # cpu state_dict of the trainable parts


@dataclass
class CheckpointSelector:
    sel_window: int = 3
    top_k: int = 3
    patience: int = 6
    n_boot: int = 1000
    seed: int = 0
    keep_states: int = 12   # states are big (QFormer->4096 heads); metrics are kept
    history: list = field(default_factory=list)  # for ALL points, states only for
    _stale: int = 0                              # the top `keep_states` candidates

    def update(self, step: int, uids, labels, scores, users, state: dict) -> bool:
        """Record one evaluation. Returns True if training should STOP (patience
        exhausted on the smoothed signal)."""
        from evaluate import auc as auc_fn
        pu = per_user_auc(uids, labels, scores, users)
        raw = float(np.mean([a for a, _ in pu.values()]))
        window = [p.uauc for p in self.history[-(self.sel_window - 1):]] + [raw]
        smoothed = float(np.mean(window))
        self.history.append(EvalPoint(
            step=step, uauc=raw, auc=auc_fn(labels, scores),
            smoothed_uauc=smoothed, per_user=pu,
            state={k: v.detach().cpu().clone() for k, v in state.items()}))
        print(f"[select] step {step}: val UAUC {raw:.4f} (smoothed {smoothed:.4f}) "
              f"val AUC {self.history[-1].auc:.4f}")

        best = max(p.smoothed_uauc for p in self.history)
        if smoothed >= best - 1e-9:
            self._stale = 0
        else:
            self._stale += 1

        # bound memory: drop the STATE (not the metrics) of checkpoints that can
        # no longer plausibly enter the soup — everything below the top
        # `keep_states` by smoothed UAUC. Curves and the noise band stay exact.
        with_state = [p for p in self.history if p.state is not None]
        if len(with_state) > self.keep_states:
            with_state.sort(key=lambda p: p.smoothed_uauc, reverse=True)
            for p in with_state[self.keep_states:]:
                p.state = None
        return self._stale >= self.patience

    # ------------------------------------------------------------------ #
    def _noise_band(self) -> float:
        """Half-width of the bootstrap CI on the BEST checkpoint's val UAUC —
        the resolution below which two checkpoints are indistinguishable."""
        best = max(self.history, key=lambda p: p.smoothed_uauc)
        aucs = np.array([a for a, _ in best.per_user.values()])
        rng = np.random.default_rng(self.seed)
        boots = rng.choice(aucs, size=(self.n_boot, len(aucs)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        return float(hi - lo) / 2

    def rank(self) -> list[EvalPoint]:
        """Checkpoints ordered best-first: smoothed UAUC desc, but within the
        noise band ties break by val AUC (never across a real UAUC gap)."""
        band = self._noise_band()
        # only checkpoints whose state was retained can be selected/souped
        pts = sorted((p for p in self.history if p.state is not None),
                     key=lambda p: p.smoothed_uauc, reverse=True)
        best_u = pts[0].smoothed_uauc
        in_band = [p for p in pts if best_u - p.smoothed_uauc <= band]
        out_band = [p for p in pts if best_u - p.smoothed_uauc > band]
        # inside the band, AUC decides; outside, smoothed UAUC order stands
        in_band.sort(key=lambda p: p.auc, reverse=True)
        print(f"[select] noise band ±{band:.4f}: {len(in_band)} checkpoint(s) tied at top")
        return in_band + out_band

    @staticmethod
    def _avg(states: list[dict]) -> dict:
        avg = {}
        for key in states[0].keys():
            avg[key] = (sum(s[key].float() for s in states) / len(states)
                        ).to(states[0][key].dtype)  # keep bf16 LoRA dtype
        return avg

    def soup(self, k: int | None = None) -> dict:
        """Weight-average (uniform soup) of the top-k checkpoints' states."""
        k = k or self.top_k
        top = self.rank()[:k]
        print(f"[select] souping {len(top)} checkpoints from steps {[p.step for p in top]}")
        return self._avg([p.state for p in top])

    def greedy_soup(self, eval_fn, k: int | None = None) -> dict:
        """Wortsman-style greedy soup: start from the best checkpoint and add
        the next-ranked candidate ONLY if the averaged weights improve val UAUC
        (eval_fn: state_dict -> val UAUC). Uniform averaging assumes the
        checkpoints share a linearly-connected basin — false for a from-scratch
        bridge whose distant checkpoints implement different attention
        solutions, where blind averaging interferes destructively. Greedy
        verification makes the soup provably no worse than the best single
        checkpoint, at the cost of a few extra val passes."""
        k = k or self.top_k
        cands = self.rank()[:k + 2]              # a couple of spares to try
        members = [cands[0].state]
        best_u = eval_fn(cands[0].state)
        print(f"[soup] base = step {cands[0].step}: val UAUC {best_u:.4f}")
        for p in cands[1:]:
            if len(members) >= k:
                break
            trial = self._avg(members + [p.state])
            u = eval_fn(trial)
            keep = u > best_u
            print(f"[soup] + step {p.step}: val UAUC {u:.4f} ({'kept' if keep else 'rejected'})")
            if keep:
                members.append(p.state)
                best_u = u
        return self._avg(members) if len(members) > 1 else members[0]

    def best_state(self) -> dict:
        return copy.deepcopy(self.rank()[0].state)

    def curves(self):
        """(steps, raw_uauc, smoothed_uauc, auc) for plotting."""
        h = self.history
        return ([p.step for p in h], [p.uauc for p in h],
                [p.smoothed_uauc for p in h], [p.auc for p in h])
