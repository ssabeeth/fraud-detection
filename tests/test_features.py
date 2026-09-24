"""Point-in-time correctness and online/offline agreement of the features."""

import math

import pytest

from fraud.features.catalogue import render
from fraud.features.definitions import AGGREGATE_NAMES, AGGREGATES, DAY, HOUR, LOCAL_NAMES
from fraud.features.offline import add_features, build_features
from fraud.features.online import OnlineFeatures
from fraud.features.pit import check
from fraud.features.reference import _close
from fraud.lakehouse.tables import GOLD_FEATURES, GOLD_TRANSACTIONS, table_path
from tests.conftest import FIXTURES


def _event(t, amt=10.0, card1=1000, addr1=100.0, d1=0.0, device=None, email="gmail.com"):
    return {
        "TransactionDT": t,
        "TransactionAmt": amt,
        "card1": card1,
        "addr1": addr1,
        # keep the first-seen day fixed while time moves on
        "D1": d1 + (t // DAY),
        "P_emaildomain": email,
        "DeviceType": "mobile" if device else None,
        "DeviceInfo": device,
        "has_identity": device is not None,
    }


def test_same_second_events_do_not_see_each_other():
    f = OnlineFeatures()
    a = f.process(_event(10 * DAY))
    b = f.process(_event(10 * DAY, amt=20.0))
    assert a["card_txn_prior"] == 0
    assert b["card_txn_prior"] == 0  # same second: not visible
    c = f.process(_event(10 * DAY + 1))
    assert c["card_txn_prior"] == 2
    assert c["card_amt_24h"] == 30.0


def test_window_lower_bound_is_inclusive():
    f = OnlineFeatures()
    t0 = 10 * DAY
    f.process(_event(t0))
    at_edge = f.process(_event(t0 + HOUR))  # t - W == t0: included
    assert at_edge["card_txn_1h"] == 1
    past_edge = f.process(_event(t0 + HOUR + HOUR + 1))
    # t0 is outside; t0 + HOUR is 3601 s earlier, also outside
    assert past_edge["card_txn_1h"] == 0
    assert past_edge["card_txn_prior"] == 2


def test_new_device_and_email_flags():
    f = OnlineFeatures()
    t = 10 * DAY
    first = f.process(_event(t, device="iOS Device"))
    again = f.process(_event(t + 60, device="iOS Device"))
    other = f.process(_event(t + 120, device="Windows", email="yahoo.com"))
    none = f.process(_event(t + 180, device=None))
    assert first["card_new_device"] == 1
    assert again["card_new_device"] == 0
    assert other["card_new_device"] == 1 and other["card_new_email"] == 1
    assert other["card_devices_30d"] == 1  # only iOS Device was seen before
    assert none["card_new_device"] is None and none["device_txn_24h"] is None


def test_amount_zscore_needs_two_prior_transactions():
    f = OnlineFeatures()
    t = 10 * DAY
    assert f.process(_event(t, amt=10.0))["card_amt_zscore"] is None
    assert f.process(_event(t + 1, amt=20.0))["card_amt_zscore"] is None
    z = f.process(_event(t + 2, amt=30.0))["card_amt_zscore"]
    assert math.isclose(z, (30.0 - 15.0) / math.sqrt(50.0))


def test_out_of_order_events_are_rejected():
    f = OnlineFeatures()
    f.process(_event(10 * DAY))
    with pytest.raises(ValueError, match="time order"):
        f.process(_event(10 * DAY - 5))


def test_catalogue_is_up_to_date():
    path = FIXTURES.parents[1] / "docs" / "features.md"
    assert path.read_text() == render(), "run `uv run fraud catalogue`"


def test_every_aggregate_has_a_reviewer_name_and_unique_name():
    assert len(set(AGGREGATE_NAMES)) == len(AGGREGATE_NAMES)
    assert all(a.reviewer_name and a.definition() for a in AGGREGATES)
    assert not set(AGGREGATE_NAMES) & set(LOCAL_NAMES)


# --- Spark: point-in-time test, canary, parity -------------------------------------------


@pytest.fixture(scope="module")
def features(spark, lake):
    build_features(spark, lake)
    return spark.read.format("delta").load(table_path(lake, GOLD_FEATURES))


@pytest.mark.spark
def test_point_in_time_every_fixture_row(spark, lake, features):
    diffs = check(spark, lake, features=features)
    assert diffs == [], diffs[:10]


@pytest.mark.spark
def test_point_in_time_canary_catches_a_leaky_window(spark, lake):
    """Including the current row in the window must be caught by the check."""
    gold = spark.read.format("delta").load(table_path(lake, GOLD_TRANSACTIONS))
    leaky = add_features(gold, leaky=True)
    diffs = check(spark, lake, features=leaky)
    leaked = {name for _, name, _, _ in diffs}
    assert "card_txn_1h" in leaked and "card_amt_24h" in leaked


@pytest.mark.spark
def test_online_features_equal_offline_on_every_fixture_row(spark, lake, features):
    offline = features.orderBy("TransactionDT", "TransactionID").toPandas()
    online = OnlineFeatures()
    mismatches = []
    for row in offline.to_dict("records"):
        got = online.process(row)
        for name in [*AGGREGATE_NAMES, "amt_cents", "hour", "weekday", "email_match"]:
            if not _close(row[name], got[name], rel_tol=1e-9):
                mismatches.append((row["TransactionID"], name, row[name], got[name]))
    assert mismatches == [], mismatches[:10]


def test_parity_compare_reports_a_changed_value():
    import pandas as pd

    from fraud.features.parity import COMPARED, compare

    row = {name: 1.0 for name in COMPARED} | {
        "card_key": "1_2_3",
        "split": "test",
        "TransactionID": 1,
    }
    offline = pd.DataFrame([row, row | {"TransactionID": 2}])
    online = [dict(row), dict(row) | {"card_txn_1h": 2.0}]
    result = compare(offline, online, {"test"})
    assert result["mismatches"] == 1 and result["mismatches_by_feature"] == {"card_txn_1h": 1}
