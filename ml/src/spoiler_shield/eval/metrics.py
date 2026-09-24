"""Metrics for a spoiler blocker, where the two errors cost different things.

* A **missed spoiler** (false negative) is the failure users remember, so the
  primary metric is recall at the chosen threshold.
* A **false blur** (false positive) hides a harmless sentence. Too many and
  people uninstall, so we track the false-blur rate: FP / all non-spoilers.

All functions take probabilities, never hard labels. The 2023 IMDB notebook
drew ROC curves from 0/1 predictions, which collapses each curve to a single
point; taking scores here makes that mistake impossible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score


def _as_arrays(
    y_true: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true has {y.size} items but scores has {s.size}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return y, s


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_at(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> Confusion:
    """Confusion counts when blurring every score >= threshold."""
    y, s = _as_arrays(y_true, scores)
    pred = s >= threshold
    return Confusion(
        tp=int(np.sum(pred & (y == 1))),
        fp=int(np.sum(pred & (y == 0))),
        tn=int(np.sum(~pred & (y == 0))),
        fn=int(np.sum(~pred & (y == 1))),
    )


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def recall_at(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """Share of spoilers blurred."""
    c = confusion_at(y_true, scores, threshold)
    return _safe_div(c.tp, c.tp + c.fn)


def precision_at(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """Share of blurred sentences that really were spoilers."""
    c = confusion_at(y_true, scores, threshold)
    return _safe_div(c.tp, c.tp + c.fp)


def false_blur_rate(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """Share of harmless sentences that got blurred: FP / (FP + TN)."""
    c = confusion_at(y_true, scores, threshold)
    return _safe_div(c.fp, c.fp + c.tn)


def pr_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision).

    Preferred over ROC-AUC because spoilers are a minority class, where ROC-AUC
    looks flattering even when precision is poor.
    """
    y, s = _as_arrays(y_true, scores)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def expected_calibration_error(y_true: ArrayLike, probs: ArrayLike, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed spoiler rate.

    Equal-width bins over [0, 1]. A value near 0 means "70% sure" really is
    right about 70% of the time, which the sensitivity settings rely on.
    """
    y, p = _as_arrays(y_true, probs)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def threshold_for_recall(y_true: ArrayLike, scores: ArrayLike, target_recall: float) -> float:
    """Highest threshold whose recall is at least ``target_recall``.

    The highest such threshold blurs the fewest harmless sentences while still
    catching the target share of spoilers. Pick it on validation data only.
    """
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    y, s = _as_arrays(y_true, scores)
    positives = np.sort(s[y == 1])[::-1]  # spoiler scores, high to low
    if positives.size == 0:
        raise ValueError("need at least one spoiler to choose a threshold")
    k = int(np.ceil(target_recall * positives.size))  # spoilers that must be caught
    return float(positives[k - 1])


def gate_recall(y_true: ArrayLike, gate_pass: ArrayLike) -> float:
    """Share of spoilers the Stage 1 name gate lets through to the model.

    End-to-end recall can never exceed this: a spoiler the gate drops is never scored.
    """
    y, g = _as_arrays(y_true, np.asarray(gate_pass, dtype=float))
    positives = y == 1
    return _safe_div(float(np.sum(g[positives] > 0)), float(positives.sum()))


@dataclass(frozen=True)
class EvalResult:
    n: int
    n_spoilers: int
    threshold: float
    recall: float
    precision: float
    false_blur_rate: float
    pr_auc: float
    ece: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate(y_true: Sequence[int] | ArrayLike, scores: ArrayLike, threshold: float) -> EvalResult:
    """All headline metrics at one threshold."""
    y, s = _as_arrays(y_true, scores)
    return EvalResult(
        n=int(y.size),
        n_spoilers=int(y.sum()),
        threshold=float(threshold),
        recall=recall_at(y, s, threshold),
        precision=precision_at(y, s, threshold),
        false_blur_rate=false_blur_rate(y, s, threshold),
        pr_auc=pr_auc(y, s),
        ece=expected_calibration_error(y, np.clip(s, 0, 1)),
    )
