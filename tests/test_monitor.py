"""Monitoring: PSI, label arrival and the drift detector on a known shift."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fraud.config import load_settings
from fraud.monitor.monitor import (
    Thresholds,
    _prepare,
    cohort_performance,
    input_drift,
    psi,
    with_cohorts,
)
from fraud.policy.costs import CostModel


def test_psi_is_zero_for_the_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.beta(1, 30, 50_000)
    assert psi(ref, rng.beta(1, 30, 20_000)) < 0.01
    assert psi(ref, rng.beta(1, 10, 20_000)) > 0.2


def test_thresholds_load():
    th = Thresholds.load()
    assert th.psi_warn < th.psi_alert and th.window_days == 7


def _decisions(n=400, seed=1):
    s = load_settings()
    rng = np.random.default_rng(seed)
    start = s.to_seconds(date(2018, 5, 1))
    return s, pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": start + np.sort(rng.integers(0, 14 * 86_400, n)),
            "TransactionAmt": rng.lognormal(4, 1, n),
            "p_fraud": rng.beta(1, 20, n),
            "action": rng.choice(["approve", "decline", "review"], n, p=[0.95, 0.03, 0.02]),
        }
    )


def test_performance_uses_only_labels_that_have_arrived():
    s, d = _decisions()
    d = with_cohorts(d, s, date(2018, 5, 1))
    labels = pd.DataFrame(
        {
            "TransactionID": d["TransactionID"],
            "isFraud": (np.arange(len(d)) % 10 == 0).astype(int),
            "released_at": d["TransactionDT"] + s.label_delay_seconds,
        }
    )
    costs = CostModel.load()
    # on 20 May no label of a May transaction has arrived
    early = cohort_performance(d, labels, s.to_seconds(date(2018, 5, 20)), costs)
    assert early["labelled"].sum() == 0 and not early["complete"].any()
    # on 7 June the first week (1-7 May) is complete, the second is not
    mid = cohort_performance(d, labels, s.to_seconds(date(2018, 6, 7)), costs)
    first, second = mid.iloc[0], mid.iloc[1]
    assert first["complete"] and not second["complete"]
    assert "pr_auc" in mid.columns and not np.isnan(first["pr_auc"])


@pytest.mark.slow
def test_input_drift_flags_a_silent_identity_feed():
    _, ref = _decisions(3000, seed=2)
    rng = np.random.default_rng(3)
    base = {
        "card_txn_24h": rng.poisson(1, 3000),
        "card_amt_24h": rng.lognormal(4, 1, 3000),
        "card_txn_prior": rng.poisson(5, 3000),
        "card_secs_since_prev": rng.exponential(1e5, 3000),
        "card_amt_zscore": rng.normal(0, 1, 3000),
        "device_txn_24h": rng.poisson(3, 3000),
        "email_txn_7d": rng.poisson(900, 3000),
        "ProductCD": rng.choice(["W", "C", "H"], 3000),
        "card4": rng.choice(["visa", "mastercard"], 3000),
        "card6": rng.choice(["debit", "credit"], 3000),
        "DeviceType": rng.choice(["mobile", "desktop", None], 3000),
        "has_identity": rng.random(3000) < 0.25,
        "P_emaildomain": rng.choice(["gmail.com", None], 3000),
    }
    ref = _prepare(ref.assign(**base), ["gmail.com"])
    cur = ref.copy()
    same = input_drift(ref, cur)
    assert same["drifted_share"] == 0
    outage = cur.assign(DeviceType="(missing)", has_identity="False", device_txn_24h=np.nan)
    shifted = input_drift(ref, outage)
    assert shifted["columns"]["DeviceType"]["drifted"]
    assert shifted["columns"]["has_identity"]["drifted"]


def test_feed_health_flags_a_field_that_goes_silent():
    from fraud.monitor.monitor import feed_health

    _, ref = _decisions(3000, seed=4)
    rng = np.random.default_rng(5)
    ref = ref.assign(
        **{
            c: rng.poisson(2, 3000).astype(float)
            for c in (
                "card_txn_24h",
                "card_amt_24h",
                "card_txn_prior",
                "card_secs_since_prev",
                "card_amt_zscore",
                "email_txn_7d",
            )
        },
        device_txn_24h=np.where(rng.random(3000) < 0.2, 3.0, np.nan),
        ProductCD="W",
        card4="visa",
        card6="debit",
        DeviceType=np.where(rng.random(3000) < 0.2, "mobile", None),
        has_identity=rng.random(3000) < 0.2,
        P_emaildomain="gmail.com",
    )
    ref = _prepare(ref, ["gmail.com"])
    normal = ref.sample(400, random_state=1)
    assert feed_health(ref, normal) == []
    silent = normal.assign(DeviceType="(missing)", has_identity="False", device_txn_24h=np.nan)
    assert set(feed_health(ref, silent)) == {"DeviceType", "has_identity", "device_txn_24h"}
