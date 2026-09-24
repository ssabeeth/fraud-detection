"""Probability calibration, chosen and fitted on the validation month.

The policy multiplies probabilities by amounts, so a model trained with class weights
(which inflates its scores) must be mapped back to real probabilities. Three options
are compared with a two-fold time split inside the validation month (fit on the first
half, score the second, and the reverse): none, Platt scaling on the log-odds, and
isotonic regression. The one with the lower Brier score is refitted on the whole month.
The fitted calibrator is plain numbers, stored as JSON with the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    method: str = "none"  # none | platt | isotonic
    a: float = 1.0
    b: float = 0.0
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "platt":
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
            self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
            self.x = iso.X_thresholds_.tolist()
            self.y = iso.y_thresholds_.tolist()
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "platt":
            return 1.0 / (1.0 + np.exp(-(self.a * _logit(p) + self.b)))
        if self.method == "isotonic":
            return np.interp(p, self.x, self.y)
        return p

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, text: str) -> Calibrator:
        return cls(**json.loads(text))


def choose(p: np.ndarray, y: np.ndarray, time_order: np.ndarray) -> tuple[Calibrator, dict]:
    """Pick the method by two-fold time split within validation, refit on all of it."""
    order = np.argsort(time_order, kind="stable")
    half = len(order) // 2
    folds = [(order[:half], order[half:]), (order[half:], order[:half])]
    scores = {}
    for method in ("none", "platt", "isotonic"):
        briers = []
        for fit_idx, eval_idx in folds:
            cal = Calibrator(method).fit(p[fit_idx], y[fit_idx])
            briers.append(brier_score_loss(y[eval_idx], cal(p[eval_idx])))
        scores[method] = float(np.mean(briers))
    best = min(scores, key=scores.get)
    return Calibrator(best).fit(p, y), scores
