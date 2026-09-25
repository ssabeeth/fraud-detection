"""Metrics, rules, calibration, preprocessing and model bundles."""

import builtins
import io

import mlflow
import numpy as np
import pandas as pd
import pytest

from fraud.model import train as t
from fraud.model.bundle import ModelBundle
from fraud.model.calibration import Calibrator, choose
from fraud.model.data import list_test_touches, load_frame, record_test_touch
from fraud.model.inputs import FEATURE_SETS, OTHER, Preprocessor
from fraud.model.metrics import calibration, recall_at_fpr, summarise
from fraud.model.rules import Rules

# --- metrics ---------------------------------------------------------------------------


def test_recall_at_fpr_on_a_known_example():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    amount = np.array([100.0, 10.0, 1.0, 1000.0, 1, 1, 1, 1, 1, 1])
    r = recall_at_fpr(y, score, amount, 0.0)
    assert r.recall == pytest.approx(2 / 3) and r.fpr == 0.0
    assert r.value_recall == pytest.approx(110 / 1110)
    r = recall_at_fpr(y, score, amount, 1 / 7)
    assert r.recall == 1.0 and r.fpr == pytest.approx(1 / 7)


def test_recall_at_fpr_does_not_split_ties():
    y = np.array([1, 0, 1, 0])
    score = np.array([1.0, 1.0, 0.0, 0.0])
    r = recall_at_fpr(y, score, np.ones(4), 0.4)
    # the top tie contains one negative (FPR 0.5 > 0.4), so nothing can be flagged
    assert r.recall == 0.0


def test_calibration_of_a_calibrated_score_is_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 200_000)
    y = rng.random(200_000) < p
    _, ece = calibration(y.astype(int), p)
    assert ece < 0.003
    m = summarise(y, p, np.ones_like(p))
    assert 0.0 < m["brier"] < 0.1 and 0.5 < m["roc_auc"] < 1.0


# --- rules -----------------------------------------------------------------------------


def _rules_frame():
    return pd.DataFrame(
        {
            "TransactionAmt": [50.0, 600.0, 50.0, 250.0, 99_999.0],
            "card_txn_1h": [0, 0, 4, 0, np.nan],
            "card_txn_24h": [0, 0, 6, 0, np.nan],
            "card_amt_24h": [0.0, 0.0, 1200.0, 0.0, np.nan],
            "card_new_device": [np.nan, 0, 0, 1, np.nan],
        }
    )


def test_rules_points_and_tie_break():
    rules = Rules(amount=500, txn_1h=3, txn_24h=5, spend_24h=1000, new_device_amount=200)
    df = _rules_frame()
    assert rules.points(df).tolist() == [0, 2, 4, 1, 3]
    score = rules.score(df)
    # the amount tie-break never lifts a row into the next points level
    assert np.all(np.floor(score) == rules.points(df))


# --- calibration and preprocessing -----------------------------------------------------


def test_calibrators_round_trip_and_choose():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 20_000)
    y = (rng.random(20_000) < raw**3).astype(int)  # raw scores are over-confident
    for method in ("none", "platt", "isotonic"):
        cal = Calibrator(method).fit(raw, y)
        again = Calibrator.from_json(cal.to_json())
        np.testing.assert_allclose(cal(raw[:100]), again(raw[:100]))
    best, scores = choose(raw, y, np.arange(len(raw)))
    assert best.method in ("platt", "isotonic") and scores[best.method] < scores["none"]


def test_preprocessor_levels_unknowns_and_single_rows():
    train = pd.DataFrame(
        {
            "card4": ["visa", "visa", "mastercard", None],
            "M1": ["T", "F", None, "T"],
            "TransactionAmt": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pre = Preprocessor(["card4", "M1", "TransactionAmt"]).fit(train)
    again = Preprocessor.from_json(pre.to_json())
    new = pd.DataFrame(
        {
            "card4": ["discover", "visa", None],
            "M1": ["F", None, "T"],
            "TransactionAmt": [5.0, 6.0, 7.0],
        }
    )
    x = again.transform(new)
    assert list(x["card4"].astype(object)) == [OTHER, "visa", np.nan] or pd.isna(x["card4"][2])
    assert x["M1"].tolist()[0] == 0.0 and np.isnan(x["M1"].tolist()[1])
    one = again.transform(new.iloc[[1]])
    pd.testing.assert_frame_equal(one, x.iloc[[1]])


def test_feature_sets():
    assert set(FEATURE_SETS["explainable"]) < set(FEATURE_SETS["all"])
    assert "V339" in FEATURE_SETS["all"] and "V1" not in FEATURE_SETS["explainable"]


# --- the test-month guard --------------------------------------------------------------


def test_registered_names_follow_the_registry(monkeypatch):
    assert t.uc_schema("/Volumes/workspace/fraud/lake") == "workspace.fraud"
    assert t.uc_schema("data") == "workspace.fraud"
    monkeypatch.setattr(t.mlflow, "get_registry_uri", lambda: "sqlite:///m.db")
    assert t.registered_name("lightgbm") == "fraud-lightgbm"
    monkeypatch.setattr(t.mlflow, "get_registry_uri", lambda: "databricks-uc")
    monkeypatch.setenv("FRAUD_UC_SCHEMA", "main.risk")
    assert t.registered_name("lightgbm_explainable") == "main.risk.fraud_lightgbm_explainable"


def _forbid_appends(mp: pytest.MonkeyPatch) -> None:
    """Databricks volumes reject appends ("Illegal seek"); fail any file opened for one."""
    real_open = io.open

    def guarded(file, mode="r", *args, **kwargs):
        if "a" in mode:
            raise OSError(29, f"append to {file} (not supported on Databricks volumes)")
        return real_open(file, mode, *args, **kwargs)

    mp.setattr(builtins, "open", guarded)
    mp.setattr(io, "open", guarded)


def test_test_touches_are_logged(tmp_path, monkeypatch):
    _forbid_appends(monkeypatch)
    log = tmp_path / "touches.jsonl"
    record_test_touch("unit test", log)
    record_test_touch("again", log)
    assert [e["purpose"] for e in list_test_touches(log)] == ["unit test", "again"]


@pytest.mark.spark
def test_reading_the_test_month_needs_a_purpose(spark, feature_lake):
    with pytest.raises(PermissionError):
        load_frame(spark, feature_lake, ["test"])


# --- training on the fixtures ----------------------------------------------------------


@pytest.fixture(scope="module")
def frames(spark, feature_lake):
    return load_frame(spark, feature_lake, ["train"]), load_frame(spark, feature_lake, ["valid"])


@pytest.fixture(scope="module")
def mlflow_tmp(feature_lake):
    t.setup_mlflow(feature_lake)
    return feature_lake


@pytest.mark.spark
@pytest.mark.slow
def test_train_save_load_predict(frames, mlflow_tmp, tmp_path, monkeypatch):
    monkeypatch.setattr(t, "LGBM_GRID", [t.LGBM_GRID[0]])
    monkeypatch.setattr(t, "LOGREG_GRID", [t.LOGREG_GRID[1]])
    train, valid = frames
    fitted = [
        t.fit_rules(train, valid),
        t.fit_logreg(train, valid),
        t.fit_lightgbm(train, valid),
    ]
    # Unity Catalog registers only models with a signature, so both carry one here too
    client = mlflow.MlflowClient()
    for short in ("logreg", "lightgbm"):
        name = t.registered_name(short)
        version = max(int(v.version) for v in client.search_model_versions(f"name='{name}'"))
        info = mlflow.models.get_model_info(f"models:/{name}/{version}")
        assert info.signature.inputs and info.signature.outputs, short
    for bundle, metrics in fitted:
        assert 0.0 < metrics["pr_auc"] <= 1.0
        with pytest.MonkeyPatch.context() as mp:
            _forbid_appends(mp)
            path = bundle.save(tmp_path)
        loaded = ModelBundle.load(path)
        np.testing.assert_allclose(loaded.predict(valid), bundle.predict(valid), rtol=1e-9)
        # one event on its own scores the same as in a batch
        np.testing.assert_allclose(
            loaded.predict(valid.iloc[[7]]), bundle.predict(valid)[[7]], rtol=1e-9
        )
    lgbm = fitted[2][0]
    contrib = lgbm.contributions(valid.iloc[:5])
    assert contrib.shape == (5, len(FEATURE_SETS["all"]))
    # SHAP values plus the bias add up to the raw log-odds
    raw = lgbm.model.predict(lgbm.matrix(valid.iloc[:5]), raw_score=True)
    full = lgbm.model.predict(lgbm.matrix(valid.iloc[:5]), pred_contrib=True).sum(axis=1)
    np.testing.assert_allclose(full, raw, rtol=1e-6)
