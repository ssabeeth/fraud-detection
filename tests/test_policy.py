"""The cost model and the decision policies, on numbers small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from fraud.policy import policy as P
from fraud.policy.costs import APPROVE, DECLINE, REVIEW, CostModel

COSTS = CostModel(
    chargeback_fee=20.0,
    review_cost=7.0,
    review_catch_rate=0.9,
    decline_margin=0.25,
    decline_fixed=10.0,
    review_delay_friction=2.0,
    review_capacity_per_day=1,
)


def test_default_cost_file_loads_with_ranges():
    c = CostModel.load()
    ranges = CostModel.ranges()
    assert set(ranges) == set(c.__dataclass_fields__)
    for name, (lo, hi) in ranges.items():
        assert lo <= getattr(c, name) <= hi, name


def test_expected_costs_by_hand():
    e = COSTS.expected(np.array([0.5]), np.array([100.0]))[0]
    assert e[APPROVE] == pytest.approx(0.5 * 120)  # p × (amount + fee)
    assert e[DECLINE] == pytest.approx(0.5 * (25 + 10))  # (1 − p) × (margin + fixed)
    assert e[REVIEW] == pytest.approx(7 + 0.5 * 0.1 * 120 + 0.5 * 2)


def test_review_priority_reduces_to_the_brief_formula():
    simple = COSTS.with_(
        chargeback_fee=0.0, review_catch_rate=1.0, review_delay_friction=0.0, decline_margin=100.0
    )
    p, amt = np.array([0.6, 0.9, 0.01]), np.array([2000.0, 20.0, 50.0])
    np.testing.assert_allclose(simple.review_priority(p, amt), p * amt - 7.0)


def test_realised_costs_by_hand():
    actions = np.array([APPROVE, APPROVE, DECLINE, DECLINE, REVIEW, REVIEW])
    fraud = np.array([1, 0, 1, 0, 1, 0])
    amount = np.array([100.0, 50.0, 200.0, 40.0, 300.0, 60.0])
    r = COSTS.realised(actions, fraud, amount)
    missed = 120 + 0.1 * 320
    assert r["missed_fraud_cost"] == pytest.approx(missed)
    assert r["declined_legit_cost"] == pytest.approx(0.25 * 40 + 10)
    assert r["review_cost"] == pytest.approx(14) and r["review_delay_cost"] == pytest.approx(2)
    assert r["total_cost"] == pytest.approx(missed + 20 + 14 + 2)
    assert r["fraud_value_caught"] == pytest.approx(200 + 0.9 * 300)
    assert r["fraud_value_caught_share"] == pytest.approx(470 / 600)


def _frame(p, amount, days, fraud=None):
    return pd.DataFrame(
        {
            "score": p,
            "TransactionAmt": amount,
            "event_date": days,
            "isFraud": fraud if fraud is not None else [0] * len(p),
        }
    )


def test_capacity_is_per_day_and_in_arrival_order():
    cand = np.array([True, True, False, True, True, True])
    day = np.array([1, 1, 1, 2, 2, 2])
    assert P.within_capacity(cand, day, 2).tolist() == [True, True, False, True, True, False]


def test_expected_loss_policy_never_exceeds_capacity_and_declines_by_cost():
    f = _frame([0.9, 0.3, 0.3, 0.02], [1000.0, 500.0, 400.0, 10.0], ["d1", "d1", "d1", "d1"])
    actions = P.ExpectedLossPolicy("score").decide(f, COSTS)
    # 0.9 on $1,000: decline is far cheaper than approve (0.1 × 260 vs 0.9 × 1020)
    assert actions[0] == DECLINE
    # the first $500 case takes the day's only review slot; the $400 case falls back
    assert actions[1] == REVIEW and actions[2] != REVIEW
    assert actions[3] == APPROVE
    assert (actions == REVIEW).sum() <= COSTS.review_capacity_per_day


def test_expected_saving_ranks_a_likely_small_fraud_below_a_less_likely_large_one():
    f = _frame([0.9, 0.6], [20.0, 2000.0], ["d1", "d1"], fraud=[1, 1])
    costs = COSTS.with_(decline_margin=10.0)  # make declining unattractive
    by_saving = P.daily_top_k(f, costs.review_priority(f["score"], f["TransactionAmt"]), 1)
    by_prob = P.daily_top_k(f, f["score"].to_numpy(), 1)
    assert by_saving.tolist() == [False, True]
    assert by_prob.tolist() == [True, False]
    ranked = P.compare_rankings(f, costs, "score")
    saving, prob = ranked.values()
    assert saving["total_cost"] < prob["total_cost"]


def test_tuning_picks_the_cheapest_trial():
    rng = np.random.default_rng(0)
    n = 3000
    p = rng.beta(0.3, 8, n)
    f = _frame(p, rng.lognormal(4, 1, n), rng.integers(0, 30, n), (rng.random(n) < p).astype(int))
    f = f.sort_values("event_date", kind="stable").reset_index(drop=True)
    costs = COSTS.with_(review_capacity_per_day=5)
    best, trials = P.tune_expected_loss(f, costs, "score")
    assert P.cost(best, f, costs)["total_cost"] == pytest.approx(
        min(t["total_cost"] for t in trials)
    )
    cut, trials = P.tune_cutoffs(f, costs, "score", n=10)
    assert P.cost(cut, f, costs)["total_cost"] == pytest.approx(
        min(t["total_cost"] for t in trials)
    )
    # screening beats approving everything on this data
    assert P.cost(best, f, costs)["total_cost"] < P.cost(P.ApproveAll(), f, costs)["total_cost"]
