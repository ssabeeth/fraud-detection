"""The scoring API: features from supplied history, reasons, rejection of future history."""

import pytest
from fastapi.testclient import TestClient

from fraud.lakehouse.tables import GOLD_FEATURES, table_path
from fraud.model.pipeline import models_dir
from fraud.stream.messages import EVENT_COLUMNS, decode, encode

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def gold(spark, fixture_models):
    lake, _ = fixture_models
    pdf = spark.read.format("delta").load(table_path(lake, GOLD_FEATURES)).toPandas()
    return pdf.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def test_api_scores_explains_and_rejects_future_history(gold, fixture_models):
    from fraud.serve import app as api

    lake, policy_path = fixture_models
    api.load(models_dir(lake) / "lightgbm", policy_path)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"

    # a card with several transactions: score its last one from the earlier ones
    key = gold["card_key"].value_counts().index[0]
    card = gold[gold["card_key"] == key]
    tx, history = card.iloc[-1], card.iloc[:-1]
    body = {
        "transaction": encode_row(tx),
        "history": [encode_row(r) for _, r in history.iterrows()],
    }
    r = client.post("/score", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert 0 <= out["p_fraud"] <= 1 and out["action"] in ("approve", "decline", "review")
    assert out["reasons"] and out["reasons"][0]["reason"]
    # card-level features from the supplied card history equal the offline ones
    for name in ("card_txn_prior", "card_txn_30d", "card_amt_7d", "card_secs_since_prev"):
        off = tx[name]
        assert (out["features"][name] is None and off != off) or out["features"][
            name
        ] == pytest.approx(off), name

    body["history"].append(encode_row(tx) | {"TransactionDT": int(tx["TransactionDT"]) + 1})
    assert client.post("/score", json=body).status_code == 422
    metrics = client.get("/metrics").text
    assert "fraud_score_requests_total" in metrics and "fraud_score_latency_seconds" in metrics


def encode_row(row) -> dict:
    return decode(encode({k: row[k] for k in EVENT_COLUMNS if k in row.index}))
