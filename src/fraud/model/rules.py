"""The rules baseline: hand-written amount and velocity rules with points.

This is the kind of rule set a fraud team runs before it has a model. Each rule adds
points; transactions are ranked by points, and within the same points by amount
(bigger first). The thresholds are tuned on the validation month over a grid of round,
human-readable values, maximising PR-AUC, the same metric the models are tuned on.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

GRID = {
    "amount": [200.0, 300.0, 500.0, 1000.0],
    "txn_1h": [2, 3, 5],
    "txn_24h": [3, 5, 10],
    "spend_24h": [300.0, 500.0, 1000.0, 2000.0],
    "new_device_amount": [100.0, 200.0, 500.0],
}


@dataclass(frozen=True)
class Rules:
    amount: float = 500.0
    txn_1h: int = 3
    txn_24h: int = 5
    spend_24h: float = 1000.0
    new_device_amount: float = 200.0

    def fired(self, df: pd.DataFrame) -> pd.DataFrame:
        amt = df["TransactionAmt"]
        return pd.DataFrame(
            {
                "large_amount": amt >= self.amount,
                "burst_last_hour": df["card_txn_1h"].fillna(0) >= self.txn_1h,
                "many_last_24h": df["card_txn_24h"].fillna(0) >= self.txn_24h,
                "high_spend_24h": (df["card_amt_24h"].fillna(0) + amt) >= self.spend_24h,
                "new_device_and_large": (df["card_new_device"] == 1)
                & (amt >= self.new_device_amount),
            },
            index=df.index,
        )

    POINTS: ClassVar[dict[str, int]] = {
        "large_amount": 2,
        "burst_last_hour": 2,
        "many_last_24h": 1,
        "high_spend_24h": 1,
        "new_device_and_large": 1,
    }

    def points(self, df: pd.DataFrame) -> np.ndarray:
        f = self.fired(df)
        return sum(f[c].to_numpy(dtype=float) * w for c, w in self.POINTS.items())

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Points, with amount as the tie-break inside a points level (never across)."""
        tie = np.clip(df["TransactionAmt"].to_numpy(dtype=float), 0, 1e5) / 1e5 * 0.99
        return self.points(df) + tie


def tune(valid: pd.DataFrame) -> tuple[Rules, list[dict]]:
    y = valid["isFraud"].to_numpy()
    trials = []
    for values in itertools.product(*GRID.values()):
        rules = Rules(**dict(zip(GRID, values, strict=True)))
        trials.append({**asdict(rules), "pr_auc": average_precision_score(y, rules.score(valid))})
    best = max(trials, key=lambda t: t["pr_auc"])
    return Rules(**{k: best[k] for k in GRID}), trials
