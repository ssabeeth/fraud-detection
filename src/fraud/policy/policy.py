"""Decision policies: approve, decline or review each transaction, within a daily capacity.

The policies run in arrival order, as a stream would: a transaction that qualifies for
review is sent to the queue only if today's queue is not yet full; otherwise it gets the
best action that needs no analyst. No policy looks at later transactions.

- ``ExpectedLossPolicy`` (the project's policy): decline when that has the lowest
  expected cost, and send to review when the expected saving of a review
  (``CostModel.review_priority``) is above a threshold tuned on validation.
- ``CutoffPolicy``: the usual alternative, with probability cut-offs for decline and
  review, both tuned on validation for money.
- ``RulesPolicy``: the rules baseline's points, with a decline level and a review level
  tuned on validation for money.
- ``ApproveAll``: no screening at all, as a reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud.policy.costs import APPROVE, DECLINE, REVIEW, CostModel


def within_capacity(candidate: np.ndarray, day: np.ndarray, capacity: int) -> np.ndarray:
    """Keep the first ``capacity`` candidates of each day (rows are in arrival order)."""
    cand = pd.Series(candidate.astype(int))
    rank_in_day = cand.groupby(np.asarray(day)).cumsum()
    return candidate & (rank_in_day.to_numpy() <= capacity)


@dataclass(frozen=True)
class ApproveAll:
    name: str = "Approve everything"

    def decide(self, frame: pd.DataFrame, costs: CostModel) -> np.ndarray:
        return np.full(len(frame), APPROVE)


@dataclass(frozen=True)
class ExpectedLossPolicy:
    score: str  # column holding calibrated P(fraud)
    review_threshold: float = 0.0
    name: str = "Expected loss"

    def decide(self, frame: pd.DataFrame, costs: CostModel) -> np.ndarray:
        p, amt = frame[self.score].to_numpy(), frame["TransactionAmt"].to_numpy()
        e = costs.expected(p, amt)
        fallback = np.where(e[:, DECLINE] < e[:, APPROVE], DECLINE, APPROVE)
        priority = np.minimum(e[:, APPROVE], e[:, DECLINE]) - e[:, REVIEW]
        want = (priority > self.review_threshold) & (priority > 0)
        review = within_capacity(
            want, frame["event_date"].to_numpy(), costs.review_capacity_per_day
        )
        return np.where(review, REVIEW, fallback)


@dataclass(frozen=True)
class CutoffPolicy:
    score: str
    decline_at: float = 1.01
    review_at: float = 1.01
    name: str = "Probability cut-offs"

    def decide(self, frame: pd.DataFrame, costs: CostModel) -> np.ndarray:
        p = frame[self.score].to_numpy()
        decline = p >= self.decline_at
        want = (p >= self.review_at) & ~decline
        review = within_capacity(
            want, frame["event_date"].to_numpy(), costs.review_capacity_per_day
        )
        return np.where(decline, DECLINE, np.where(review, REVIEW, APPROVE))


@dataclass(frozen=True)
class RulesPolicy:
    score: str = "score_rules"
    decline_at: float = 99.0
    review_at: float = 99.0
    name: str = "Rules baseline"

    def decide(self, frame: pd.DataFrame, costs: CostModel) -> np.ndarray:
        return CutoffPolicy(self.score, self.decline_at, self.review_at).decide(frame, costs)


def cost(policy, frame: pd.DataFrame, costs: CostModel) -> dict:
    actions = policy.decide(frame, costs)
    return costs.realised(actions, frame["isFraud"].to_numpy(), frame["TransactionAmt"].to_numpy())


# --- tuning on validation --------------------------------------------------------------


def _grid(values: np.ndarray, n: int) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0, 1, n)))


def tune_expected_loss(frame, costs, score) -> tuple[ExpectedLossPolicy, list[dict]]:
    e = costs.expected(frame[score].to_numpy(), frame["TransactionAmt"].to_numpy())
    priority = np.minimum(e[:, APPROVE], e[:, DECLINE]) - e[:, REVIEW]
    positive = priority[priority > 0]
    grid = np.r_[0.0, _grid(positive, 60)] if len(positive) else np.array([0.0])
    trials = []
    for tau in grid:
        pol = ExpectedLossPolicy(score, float(tau))
        trials.append({"review_threshold": float(tau), **cost(pol, frame, costs)})
    best = min(trials, key=lambda r: r["total_cost"])
    return ExpectedLossPolicy(score, best["review_threshold"]), trials


def tune_cutoffs(frame, costs, score, cls=CutoffPolicy, n: int = 40):
    s = frame[score].to_numpy()
    grid = np.r_[_grid(s[s >= np.quantile(s, 0.9)], n), np.inf]
    trials = []
    for d in grid:
        for r in grid[grid <= d]:
            pol = cls(score, float(d), float(r))
            trials.append(
                {"decline_at": float(d), "review_at": float(r), **cost(pol, frame, costs)}
            )
    best = min(trials, key=lambda t: t["total_cost"])
    return cls(score, best["decline_at"], best["review_at"]), trials


def tune_rules(frame, costs, score="score_rules"):
    return tune_cutoffs(frame, costs, score, cls=RulesPolicy, n=60)


# --- ranking comparison (rule 8) -------------------------------------------------------


def daily_top_k(frame: pd.DataFrame, priority: np.ndarray, k: int) -> np.ndarray:
    """With hindsight over each day: the k highest-priority rows per day.

    Not deployable (a stream cannot see the rest of the day); used only to compare two
    rankings on equal terms, with the same number of reviews.
    """
    rank = (
        pd.Series(priority)
        .groupby(frame["event_date"].to_numpy())
        .rank(ascending=False, method="first")
    )
    return rank.to_numpy() <= k


def compare_rankings(frame: pd.DataFrame, costs: CostModel, score: str) -> dict:
    """Review the daily top-k by expected saving vs by probability; approve the rest.

    Only review or approve is allowed here, so the saving is measured against approving:
    ``P × (amount + fee) × catch rate − review cost − (1 − P) × delay``, which is the
    brief's ``P(fraud) × amount − review cost`` with the fee, the analyst's misses and
    the delay cost added.
    """
    p, amt = frame[score].to_numpy(), frame["TransactionAmt"].to_numpy()
    k = costs.review_capacity_per_day
    e = costs.expected(p, amt)
    rankings = {
        "expected saving": e[:, APPROVE] - e[:, REVIEW],
        "probability": p,
    }
    out = {}
    for label, priority in rankings.items():
        review = daily_top_k(frame, priority, k)
        actions = np.where(review, REVIEW, APPROVE)
        out[label] = costs.realised(actions, frame["isFraud"].to_numpy(), amt)
    return out
