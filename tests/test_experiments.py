"""The experiments: candidate features only look back, and the adoption rule is the rule."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud.model import experiments as ex
from fraud.model.data import list_test_touches
from fraud.model.patterns import known_fraud_counts

DELAY = 5


def _frame():
    return pd.DataFrame(
        {
            "card_key": ["a", "a", "a", "b", "a", None, "b", "a", "b"],
            "TransactionDT": [0, 3, 3, 4, 9, 9, 10, 20, 20],
            "isFraud": [1, 0, 1, 0, 0, 1, 1, 0, 0],
            "C1": [1.0, 2.0, np.nan, 5.0, 4.0, 7.0, 6.0, 8.0, 9.0],
        }
    )


def _naive(df: pd.DataFrame):
    means, frauds, labelled = [], [], []
    for _, r in df.iterrows():
        if r["card_key"] is None:
            means.append(np.nan)
            frauds.append(0)
            labelled.append(0)
            continue
        same = df["card_key"] == r["card_key"]
        before = df[same & (df["TransactionDT"] < r["TransactionDT"])]
        means.append(before["C1"].mean() if before["C1"].notna().any() else np.nan)
        known = df[same & (df["TransactionDT"] < r["TransactionDT"] - DELAY)]
        frauds.append(int(known["isFraud"].sum()))
        labelled.append(len(known))
    return np.array(means), np.array(frauds), np.array(labelled)


def test_candidate_features_only_see_earlier_seconds():
    df = _frame()
    means, frauds, labelled = _naive(df)
    got = ex.prior_means(df, "card_key", ["C1"])["C1"].to_numpy(dtype=float)
    np.testing.assert_allclose(got, means, equal_nan=True)
    np.testing.assert_array_equal(known_fraud_counts(df, DELAY), frauds)
    np.testing.assert_array_equal(ex.known_counts(df, DELAY), labelled)
    # the two rows at second 3 see only the row at second 0, never each other
    assert got[1] == got[2] == 1.0


def test_adoption_needs_every_month_and_an_interval_above_zero():
    base = [np.full(30, 100.0), np.full(30, 100.0), np.full(30, 100.0)]
    better = [d - 5 for d in base]
    mixed = [base[0] - 5, base[1] + 5, base[2] - 5]
    results = {
        "baseline": [{"pr_auc": 0.5, "total_cost": 3000.0}] * 3,
        "x": [{"pr_auc": 0.52, "total_cost": 2850.0}] * 3,
    }
    assert ex.compare_to_baseline(better, base, results, "x")["adopted"]
    assert not ex.compare_to_baseline(mixed, base, results, "x")["adopted"]
    noisy = [d - 5 + np.random.default_rng(i).normal(0, 400, 30) for i, d in enumerate(base)]
    out = ex.compare_to_baseline(noisy, base, results, "x")
    assert out["saving_interval"][0] < 0 < out["saving_interval"][1]
    assert not out["adopted"]


@pytest.mark.spark
@pytest.mark.slow
def test_run_on_fixtures_without_reading_may(spark, feature_lake, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "MAX_ROUNDS", 30)
    monkeypatch.setattr(ex, "PATIENCE", 10)
    monkeypatch.setattr(ex, "BOOTSTRAP", 50)
    patterns = {"adversarial": {"top": [{"feature": "C1"}, {"feature": "D15"}]}}
    (tmp_path / "data_patterns.json").write_text(json.dumps(patterns))
    touches = len(list_test_touches())
    result = ex.run(spark, feature_lake, out_dir=tmp_path, names=["no_v", "label_history"])
    assert len(list_test_touches()) == touches
    names = [e["name"] for e in result["experiments"]]
    assert names == ["baseline", "no_v", "label_history"]
    by = {e["name"]: e for e in result["experiments"]}
    assert by["baseline"]["pooled_saving"] == 0.0 and not by["baseline"]["adopted"]
    assert by["no_v"]["folds"][0]["features"] == by["baseline"]["folds"][0]["features"] - 339
    assert "May is not read" in (tmp_path / "experiments.md").read_text()
