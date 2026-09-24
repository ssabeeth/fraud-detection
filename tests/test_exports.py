"""The dashboard export: aggregates only, and its totals equal the policy report's."""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "exports" / "daily_policy_results.csv"
RESULTS = ROOT / "reports" / "policy_results.json"


@pytest.fixture(scope="module")
def daily():
    return pd.read_csv(CSV)


def test_one_row_per_day_and_policy_and_no_row_level_columns(daily):
    assert not daily.duplicated(["date", "policy"]).any()
    assert daily["date"].nunique() == 31
    assert not {"TransactionID", "card_key", "card1", "isFraud"} & set(daily.columns)


def test_monthly_totals_match_the_policy_report(daily):
    test = json.loads(RESULTS.read_text())["test"]
    totals = daily.groupby("policy").sum(numeric_only=True)
    for policy, r in test.items():
        t = totals.loc[policy]
        assert t["total_cost_usd"] == pytest.approx(r["total_cost"], abs=1.0), policy
        assert t["fraud_value_caught_usd"] == pytest.approx(r["fraud_value_caught"], abs=1.0)
        assert t["reviews"] == r["reviews"] and t["declines"] == r["declines"]
        assert t["false_declines"] == r["declined_legit"]
