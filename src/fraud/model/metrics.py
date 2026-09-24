"""Metrics that fit a 3.5% positive rate.

PR-AUC (average precision) is the headline ranking metric; ROC-AUC is reported but
never alone. Recall is reported at fixed false-positive rates, by count and by value
(the share of fraud dollars), because an alert budget is set in false alarms. Calibration
matters because the policy multiplies probabilities by amounts: Brier score, expected
calibration error over equal-count bins, and the reliability curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FPR_LEVELS = (0.001, 0.005, 0.01, 0.05)


@dataclass(frozen=True)
class RecallAtFpr:
    fpr_target: float
    threshold: float
    fpr: float
    recall: float
    value_recall: float
    alert_rate: float


def recall_at_fpr(y: np.ndarray, score: np.ndarray, amount: np.ndarray, fpr_target: float):
    """Recall at the lowest threshold whose false-positive rate is at most the target.

    Ties are handled by thresholding on the score (``score >= t``), so a coarse score such
    as the rules' points cannot claim a fractional share of a tied group.
    """
    y = np.asarray(y).astype(bool)
    order = np.argsort(-score, kind="stable")
    s, yy, amt = score[order], y[order], amount[order]
    n_neg, n_pos = (~y).sum(), y.sum()
    # Candidate thresholds: the distinct scores, highest first; counts at score >= t.
    distinct_end = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    fp = np.cumsum(~yy)[distinct_end]
    tp = np.cumsum(yy)[distinct_end]
    tp_value = np.cumsum(np.where(yy, amt, 0.0))[distinct_end]
    ok = fp / n_neg <= fpr_target
    if not ok.any():
        return RecallAtFpr(fpr_target, float("inf"), 0.0, 0.0, 0.0, 0.0)
    i = np.flatnonzero(ok)[-1]
    return RecallAtFpr(
        fpr_target=fpr_target,
        threshold=float(s[distinct_end[i]]),
        fpr=float(fp[i] / n_neg),
        recall=float(tp[i] / n_pos),
        value_recall=float(tp_value[i] / amt[y].sum()),
        alert_rate=float((distinct_end[i] + 1) / len(s)),
    )


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """Equal-count bins: mean predicted vs observed rate, and the expected error."""
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    curve = [(float(p[g].mean()), float(y[g].mean()), len(g)) for g in groups if len(g)]
    ece = sum(n * abs(pred - obs) for pred, obs, n in curve) / len(p)
    return curve, float(ece)


def summarise(y, score, amount, *, probability: bool = True) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    amount = np.asarray(amount, dtype=float)
    out = {
        "rows": len(y),
        "fraud": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_fpr": [asdict(recall_at_fpr(y, score, amount, f)) for f in FPR_LEVELS],
    }
    if probability:
        curve, ece = calibration(y, score)
        out |= {"brier": float(brier_score_loss(y, score)), "ece": ece, "calibration": curve}
    return out
