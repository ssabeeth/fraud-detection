"""The online decision path for one transaction: features, score, reasons, action.

``OnlineScorer`` holds the per-key feature state and the day's review count. It is used
by the stream processor and by the FastAPI service, so both make exactly the same
decision for the same history. The policy is the one frozen in phase 5
(``reports/policy_frozen.json``): decline when its expected cost is lowest, review when
the expected saving of a review beats the frozen threshold and today's queue has room.
"""

from __future__ import annotations

import time

import numpy as np

from fraud.features.definitions import LABEL_DELAY
from fraud.features.online import OnlineFeatures
from fraud.model.bundle import ModelBundle
from fraud.policy.costs import ACTION_NAMES, APPROVE, DECLINE, REVIEW
from fraud.policy.frozen import FrozenPolicy

SECONDS_PER_DAY = 86_400


class OnlineScorer:
    def __init__(self, bundle: ModelBundle, policy: FrozenPolicy, reasons: int = 3) -> None:
        self.bundle = bundle
        self.policy = policy
        self.k = reasons
        self.features = OnlineFeatures()
        self._day: int | None = None
        self._reviews_today = 0

    def warm(self, history, labels_before: int | None = None) -> int:
        """Feed earlier transactions through the feature state (no scoring). With
        ``labels_before``, also apply the labels released before that time (those of
        transactions more than ``LABEL_DELAY`` earlier); later ones come from the stream."""
        n = 0
        labelled = []
        for event in history:
            self.features.process(event)
            n += 1
            if labels_before is not None and event["TransactionDT"] + LABEL_DELAY < labels_before:
                labelled.append(event)
        for event in labelled:
            self.features.observe_label(event)
        return n

    def observe_label(self, label: dict) -> None:
        self.features.observe_label(label)

    def decide(self, event: dict) -> dict:
        t0 = time.perf_counter()
        feats = self.features.process(event)
        t1 = time.perf_counter()
        row = {**event, **feats}
        p, reasons = self.bundle.score_one(row, self.k)
        t2 = time.perf_counter()

        costs = self.policy.costs
        amount = float(event["TransactionAmt"])
        e = costs.expected(np.array([p]), np.array([amount]))[0]
        fallback = DECLINE if e[DECLINE] < e[APPROVE] else APPROVE
        saving = min(e[APPROVE], e[DECLINE]) - e[REVIEW]
        day = int(event["TransactionDT"]) // SECONDS_PER_DAY
        if day != self._day:
            self._day, self._reviews_today = day, 0
        action = fallback
        if (
            saving > self.policy.review_threshold
            and saving > 0
            and self._reviews_today < costs.review_capacity_per_day
        ):
            action = REVIEW
            self._reviews_today += 1
        t3 = time.perf_counter()
        return {
            "TransactionID": event["TransactionID"],
            "TransactionDT": event["TransactionDT"],
            "TransactionAmt": amount,
            "p_fraud": p,
            "action": ACTION_NAMES[action],
            "expected_cost": {"approve": e[APPROVE], "decline": e[DECLINE], "review": e[REVIEW]},
            "review_saving": saving,
            "reasons": reasons,
            "features": {k: feats[k] for k in feats},
            "model": self.bundle.name,
            "timing_ms": {
                "features": (t1 - t0) * 1e3,
                "score_and_explain": (t2 - t1) * 1e3,
                "policy": (t3 - t2) * 1e3,
                "total": (t3 - t0) * 1e3,
            },
        }
