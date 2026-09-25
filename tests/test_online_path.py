"""The online path (stream scorer and API) against the offline model, policy and features."""

import numpy as np
import pytest

from fraud.features.definitions import AGGREGATE_NAMES
from fraud.features.parity import EVENT_FIELDS
from fraud.lakehouse.tables import GOLD_FEATURES, table_path
from fraud.model.bundle import ModelBundle
from fraud.model.pipeline import models_dir
from fraud.policy.costs import ACTION_NAMES
from fraud.policy.frozen import FrozenPolicy
from fraud.policy.policy import ExpectedLossPolicy
from fraud.stream.messages import EVENT_COLUMNS, decode, encode, records
from fraud.stream.producer import LABEL, TX, timeline
from fraud.stream.scorer import OnlineScorer

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def gold(spark, fixture_models):
    lake, _ = fixture_models
    pdf = spark.read.format("delta").load(table_path(lake, GOLD_FEATURES)).toPandas()
    return pdf.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def bundle(fixture_models):
    lake, _ = fixture_models
    return ModelBundle.load(models_dir(lake) / "lightgbm")


def test_single_event_scores_equal_batch_scores(gold, bundle):
    batch = bundle.predict(gold)
    single = [bundle.score_one(r)[0] for r in gold.iloc[:300].to_dict("records")]
    np.testing.assert_allclose(single, batch[:300], rtol=1e-10, atol=1e-12)


def test_scorer_matches_offline_features_scores_and_policy(gold, bundle, fixture_models):
    lake, policy_path = fixture_models
    frozen = FrozenPolicy.load(policy_path)
    scorer = OnlineScorer(bundle, frozen)
    test_start = lake.split_bounds_seconds()["test"][0]
    history = gold[gold["TransactionDT"] < test_start]
    window = gold[gold["TransactionDT"] >= test_start].reset_index(drop=True)
    # older labels at warm-up, the rest released during the window, as in the stream
    scorer.warm(records(history, [*EVENT_FIELDS, "isFraud"]), labels_before=test_start)
    delay = lake.label_delay_seconds
    earlier = history[history["TransactionDT"] >= test_start - delay]
    end = int(window["TransactionDT"].max()) + 1
    decisions = []
    for _, kind, payload in timeline(
        window[[*EVENT_COLUMNS, "isFraud"]], earlier, test_start, end, delay
    ):
        if kind == TX:
            decisions.append(scorer.decide({k: payload[k] for k in payload if k != "isFraud"}))
        else:
            scorer.observe_label(payload)

    for d, (_, row) in zip(decisions, window.iterrows(), strict=True):
        for name in AGGREGATE_NAMES:
            a, b = d["features"][name], row[name]
            assert (a is None and b != b) or a == pytest.approx(b, rel=1e-9), name
        assert d["reasons"] and all(r["contribution"] > 0 for r in d["reasons"])
    np.testing.assert_allclose(
        [d["p_fraud"] for d in decisions], bundle.predict(window), rtol=1e-10, atol=1e-12
    )
    offline = ExpectedLossPolicy("p", frozen.review_threshold).decide(
        window.assign(p=bundle.predict(window)), frozen.costs
    )
    assert [d["action"] for d in decisions] == [ACTION_NAMES[a] for a in offline]
    per_day = {}
    for d in decisions:
        if d["action"] == "review":
            day = d["TransactionDT"] // 86_400
            per_day[day] = per_day.get(day, 0) + 1
    assert max(per_day.values()) <= frozen.costs.review_capacity_per_day


def test_messages_round_trip_without_nans():
    event = {
        "TransactionID": 1,
        "TransactionAmt": 12.5,
        "V1": float("nan"),
        "x": None,
        "nested": {"a": np.float64(1.5), "b": float("nan")},
    }
    assert decode(encode(event)) == {
        "TransactionID": 1,
        "TransactionAmt": 12.5,
        "nested": {"a": 1.5},
    }


def test_label_timeline_releases_each_label_after_the_delay(gold, fixture_models):
    import pandas as pd

    lake, _ = fixture_models
    lo, hi = lake.split_bounds_seconds()["test"]
    delay = lake.label_delay_seconds
    window = gold[(gold["TransactionDT"] >= lo) & (gold["TransactionDT"] < hi)]
    earlier = gold[(gold["TransactionDT"] >= lo - delay) & (gold["TransactionDT"] < lo)]
    items = list(
        timeline(
            window[[*EVENT_COLUMNS, "isFraud"]], earlier, lo, hi, delay, labels_until=hi + delay
        )
    )
    times = [t for t, _, _ in items]
    assert times == sorted(times)
    txs = [p for _, k, p in items if k == TX]
    labels = [(t, p) for t, k, p in items if k == LABEL]
    assert len(txs) == len(window)
    assert all(t == p["TransactionDT"] + delay for t, p in labels)
    assert {p["TransactionID"] for _, p in labels} == set(
        pd.concat([earlier, window])["TransactionID"]
    )
