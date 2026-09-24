"""The cost model: what each decision costs, expected (from P(fraud)) and realised (from labels).

Three actions per transaction:

- **approve**: a fraud loses the amount plus the chargeback fee; a legitimate one costs
  nothing;
- **decline**: a fraud costs nothing; a legitimate customer costs the lost margin
  (``decline_margin × amount``) plus ``decline_fixed``;
- **review**: always costs ``review_cost``; the analyst stops a fraud with probability
  ``review_catch_rate`` (the rest is approved and lost); a legitimate customer is approved
  late and costs ``review_delay_friction``.

The values and their rationale live in ``configs/costs.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

from fraud.config import config_dir


def costs_file() -> Path:
    return config_dir() / "costs.yaml"


APPROVE, DECLINE, REVIEW = 0, 1, 2
ACTION_NAMES = {APPROVE: "approve", DECLINE: "decline", REVIEW: "review"}


@dataclass(frozen=True)
class CostModel:
    chargeback_fee: float
    review_cost: float
    review_catch_rate: float
    decline_margin: float
    decline_fixed: float
    review_delay_friction: float
    review_capacity_per_day: int

    @classmethod
    def load(cls, path: Path | None = None) -> CostModel:
        raw = yaml.safe_load((path or costs_file()).read_text())
        return cls(**{f.name: raw[f.name]["value"] for f in fields(cls)})

    @staticmethod
    def ranges(path: Path | None = None) -> dict[str, list[float]]:
        raw = yaml.safe_load((path or costs_file()).read_text())
        return {f.name: raw[f.name]["range"] for f in fields(CostModel)}

    def with_(self, **kw) -> CostModel:
        return replace(self, **kw)

    # -- expected cost of each action, given P(fraud) ---------------------------------

    def loss_if_fraud(self, amount: np.ndarray) -> np.ndarray:
        return amount + self.chargeback_fee

    def decline_friction(self, amount: np.ndarray) -> np.ndarray:
        return self.decline_margin * amount + self.decline_fixed

    def expected(self, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """Expected cost of approve, decline and review: shape (n, 3)."""
        p = np.asarray(p, dtype=float)
        amount = np.asarray(amount, dtype=float)
        loss = self.loss_if_fraud(amount)
        approve = p * loss
        decline = (1 - p) * self.decline_friction(amount)
        review = (
            self.review_cost
            + p * (1 - self.review_catch_rate) * loss
            + (1 - p) * self.review_delay_friction
        )
        return np.stack([approve, decline, review], axis=1)

    def review_priority(self, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """Money a review saves over the best action without one.

        With no chargeback fee, a perfect analyst and no delay cost this is exactly
        ``P(fraud) × amount − review cost``, the ranking the brief asks for.
        """
        e = self.expected(p, amount)
        return np.minimum(e[:, APPROVE], e[:, DECLINE]) - e[:, REVIEW]

    # -- realised cost of chosen actions, given labels ----------------------------------

    def realised(self, action: np.ndarray, is_fraud: np.ndarray, amount: np.ndarray) -> dict:
        action = np.asarray(action)
        fraud = np.asarray(is_fraud).astype(bool)
        amount = np.asarray(amount, dtype=float)
        loss = self.loss_if_fraud(amount)
        ap, de, rv = action == APPROVE, action == DECLINE, action == REVIEW
        missed_fraud = (loss * fraud * ap).sum() + (
            (1 - self.review_catch_rate) * loss * fraud * rv
        ).sum()
        declined_legit = (self.decline_friction(amount) * ~fraud * de).sum()
        review = self.review_cost * rv.sum()
        delay = (self.review_delay_friction * ~fraud * rv).sum()
        fraud_value = (amount * fraud).sum()
        caught = (amount * fraud * de).sum() + (self.review_catch_rate * amount * fraud * rv).sum()
        return {
            "total_cost": float(missed_fraud + declined_legit + review + delay),
            "missed_fraud_cost": float(missed_fraud),
            "declined_legit_cost": float(declined_legit),
            "review_cost": float(review),
            "review_delay_cost": float(delay),
            "fraud_value": float(fraud_value),
            "fraud_value_caught": float(caught),
            "fraud_value_caught_share": float(caught / fraud_value) if fraud_value else 0.0,
            "declines": int(de.sum()),
            "declined_legit": int((de & ~fraud).sum()),
            "reviews": int(rv.sum()),
            "reviewed_fraud": int((rv & fraud).sum()),
            "approved_fraud": int((ap & fraud).sum()),
        }
