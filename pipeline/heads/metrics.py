"""Gating thresholds, bootstrap confidence intervals and calibration error (spec §4.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PRF = dict[str, tuple[float, float, float]]  # metric -> (point, 95% CI low, 95% CI high)


@dataclass(frozen=True)
class Thresholds:
    """`p <= low` passes, `p >= high` is rejected, anything between escalates."""

    low: float
    high: float

    def zone(self, p: np.ndarray) -> np.ndarray:
        """0 pass, 1 escalate, 2 reject."""
        return np.where(p >= self.high, 2, np.where(p <= self.low, 0, 1))


def pick_thresholds(p: np.ndarray, y: np.ndarray, *, precision: float, recall: float) -> Thresholds:
    """The loosest `high` with precision(reject) >= `precision` and the loosest `low` with
    recall(reject) >= `recall`, i.e. the narrowest escalation band meeting both targets."""
    p = np.asarray(p, np.float64)
    y = np.asarray(y, bool)
    positives = np.sort(p[y])
    if positives.size == 0:
        return Thresholds(math.inf, math.inf)

    order = np.argsort(-p, kind="stable")
    ps, tp = p[order], np.cumsum(y[order])
    ends = np.r_[np.nonzero(np.diff(ps))[0], ps.size - 1]  # last index of each tie group
    ok = tp[ends] / (ends + 1) >= precision
    high = float(ps[ends[np.nonzero(ok)[0][-1]]]) if ok.any() else math.inf

    misses = math.floor((1 - recall) * positives.size + 1e-9)  # positives allowed to pass
    low = float(np.nextafter(positives[misses], -np.inf))
    return Thresholds(min(low, high), high)


def _prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum(y & pred))
    precision = tp / pred.sum() if pred.any() else 0.0
    recall = tp / y.sum() if y.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def bootstrap_prf(y: np.ndarray, pred: np.ndarray, *, n_boot: int = 1000, seed: int = 0) -> PRF:
    """Precision, recall and F1 with percentile-bootstrap 95% CIs."""
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    point = _prf(y, pred)
    rng = np.random.default_rng(seed)
    samples = np.array(
        [_prf(y[idx], pred[idx]) for idx in rng.integers(0, y.size, (n_boot, y.size))]
    )
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0) if n_boot else (point, point)
    return {
        name: (round(point[i], 4), round(float(lo[i]), 4), round(float(hi[i]), 4))
        for i, name in enumerate(("precision", "recall", "f1"))
    }


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    which = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += abs(p[mask].mean() - y[mask].mean()) * mask.sum()
    return total / p.size if p.size else 0.0


@dataclass(frozen=True)
class BudgetCurve:
    """One head's inputs to the escalation budget: per-image escalation keys (an image escalates
    when its key > low; -inf for images the head rejects outright), the calibrated scores of the
    calibration positives, and the auto-reject threshold. `fixed_share` reserves a set share."""

    keys: np.ndarray
    cal_positive_p: np.ndarray
    high: float
    fixed_share: float | None = None

    def low_at(self, share: float) -> float:
        """The pass threshold that escalates `share` of all images."""
        if share <= 0:
            return self.high
        return float(min(np.quantile(self.keys, 1 - share), self.high))

    def recall_at(self, share: float) -> float:
        """Share of calibration positives escalated or rejected (not auto-passed)."""
        p = self.cal_positive_p
        return float(np.mean(p > self.low_at(share))) if p.size else 1.0


@dataclass(frozen=True)
class Allocation:
    share: float
    low: float
    recall: float  # on the calibration positives


def allocate_escalation(
    curves: dict[str, BudgetCurve], *, budget: float, step: float, target_recall: float
) -> dict[str, Allocation]:
    """Split an escalation budget (share of all images) across heads: reserved shares first, then
    greedily `step` by `step` to the head whose recall rises most, until the budget is spent or
    every head reaches `target_recall`. Shares add up to at most the budget; the escalated union
    is smaller when heads overlap."""
    shares = {n: c.fixed_share or 0.0 for n, c in curves.items()}
    left = budget - sum(shares.values())
    open_heads = [n for n, c in curves.items() if c.fixed_share is None]
    while left >= step - 1e-12:
        live = [n for n in open_heads if curves[n].recall_at(shares[n]) < target_recall]
        if not live:
            break
        gains = {
            n: curves[n].recall_at(shares[n] + step) - curves[n].recall_at(shares[n]) for n in live
        }
        best = max(gains, key=lambda n: gains[n])
        if gains[best] <= 0:  # flat everywhere: help the head that is furthest behind
            best = min(live, key=lambda n: curves[n].recall_at(shares[n]))
        shares[best] += step
        left -= step
    return {
        n: Allocation(round(shares[n], 6), c.low_at(shares[n]), c.recall_at(shares[n]))
        for n, c in curves.items()
    }
