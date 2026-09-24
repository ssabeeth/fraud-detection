"""Reason codes, feature families and segment checks."""

import numpy as np
import pandas as pd
import pytest

from fraud.explain.experiment import family, segment_table
from fraud.explain.reasons import plain_name, reasons_for_row, top_reasons
from fraud.policy.costs import APPROVE, DECLINE, REVIEW, CostModel


def test_plain_names():
    assert plain_name("card_txn_1h") == "Card transactions, last hour"
    assert plain_name("TransactionAmt") == "Amount (USD)"
    assert plain_name("V258") == "V258 — Vesta risk signal (masked)"
    assert plain_name("C13") == "C13 — count of linked entities (masked)"
    assert plain_name("D15") == "D15 — days since an earlier event (masked)"
    assert plain_name("id_31") == "Browser"
    assert plain_name("id_02").startswith("id_02 — identity or network detail")


def test_reasons_are_the_largest_positive_contributions_in_order():
    features = ["a", "b", "c", "d"]
    reasons = reasons_for_row(np.array([0.2, -1.0, 0.9, 0.0]), features, {"c": 3.0}, k=3)
    assert [r["feature"] for r in reasons] == ["c", "a"]  # b is negative, d is zero
    assert reasons[0]["value"] == 3.0 and reasons[1]["value"] is None
    many = top_reasons(
        pd.DataFrame([[0.1, 0.5], [0.7, -0.2]], columns=["x", "y"]),
        pd.DataFrame([{"x": 1, "y": "gmail.com"}, {"x": 2, "y": None}]),
        k=1,
    )
    assert many == [
        [{"feature": "y", "reason": "y", "value": "gmail.com", "contribution": 0.5}],
        [{"feature": "x", "reason": "x", "value": 2, "contribution": 0.7}],
    ]


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("V12", "V: Vesta signals (masked)"),
        ("M4", "M: match checks (masked)"),
        ("id_30", "identity and device fields (masked)"),
        ("DeviceInfo", "identity and device fields (masked)"),
        ("card_txn_24h", "history aggregates (this project)"),
        ("device_cards_7d", "history aggregates (this project)"),
        ("card1", "readable transaction fields"),
        ("DeviceType", "readable transaction fields"),
        ("dist1", "readable transaction fields"),
    ],
)
def test_feature_families(feature, expected):
    assert family(feature) == expected


def test_segment_table_by_hand():
    frame = pd.DataFrame(
        {
            "ProductCD": ["W", "W", "W", "C"],
            "card4": ["visa"] * 4,
            "card6": ["debit"] * 4,
            "DeviceType": [None, None, "mobile", "mobile"],
            "P_emaildomain": ["gmail.com", None, "gmail.com", "yahoo.com"],
            "isFraud": [1, 0, 1, 0],
            "TransactionAmt": [100.0, 50.0, 10.0, 20.0],
        }
    )
    actions = np.array([DECLINE, REVIEW, APPROVE, APPROVE])
    rows = segment_table(frame, actions, CostModel.load())
    w = next(r for r in rows if r["segment"] == "ProductCD" and r["value"] == "W")
    assert w["transactions"] == 3
    assert w["alert_rate"] == pytest.approx(2 / 3)
    assert w["precision"] == pytest.approx(1 / 2)  # the decline was fraud, the review was not
    assert w["recall"] == pytest.approx(1 / 2)  # one of two frauds alerted
    assert w["value_caught"] == pytest.approx(100 / 110)
    missing = next(r for r in rows if r["segment"] == "DeviceType" and r["value"] == "(missing)")
    assert missing["transactions"] == 2
    c = next(r for r in rows if r["segment"] == "ProductCD" and r["value"] == "C")
    assert c["precision"] is None and c["recall"] is None
