"""``fraud train`` and ``fraud evaluate``.

``train`` fits the three models in the order the brief sets (rules, then logistic
regression, then LightGBM) on months 1-4, tunes each on month 5, and saves the bundles.
``evaluate`` scores the frozen bundles on validation and, once, on the test month, and
writes ``reports/model.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraud.config import Settings, reports_dir
from fraud.model import train as t
from fraud.model.bundle import ModelBundle
from fraud.model.data import load_frame
from fraud.model.metrics import summarise

log = logging.getLogger(__name__)

MODELS = ("rules", "logreg", "lightgbm")


def models_dir(s: Settings) -> Path:
    return s.path("models")


def train_models(spark, s: Settings, which: tuple[str, ...] = MODELS) -> dict[str, dict]:
    train = load_frame(spark, s, ["train"])
    valid = load_frame(spark, s, ["valid"])
    t.setup_mlflow(s)
    results = {}
    for name in MODELS:
        if name not in which:
            continue
        fit = {"rules": t.fit_rules, "logreg": t.fit_logreg, "lightgbm": t.fit_lightgbm}[name]
        bundle, metrics = fit(train, valid)
        path = t.save(bundle, models_dir(s))
        log.info("saved %s to %s (valid PR-AUC %.4f)", name, path, metrics["pr_auc"])
        results[name] = metrics
    return results


def load_bundles(s: Settings, names: tuple[str, ...] = MODELS) -> dict[str, ModelBundle]:
    return {n: ModelBundle.load(models_dir(s) / n) for n in names}


def score_frame(bundles: dict[str, ModelBundle], df: pd.DataFrame) -> pd.DataFrame:
    out = df[["TransactionID", "TransactionDT", "event_date", "isFraud", "TransactionAmt"]].copy()
    for name, b in bundles.items():
        out[f"score_{name}"] = b.predict(df)
    return out


def evaluate(spark, s: Settings, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or reports_dir()
    from fraud.model.report import write_model_report

    bundles = load_bundles(s)
    valid = load_frame(spark, s, ["valid"])
    test = load_frame(
        spark,
        s,
        ["test"],
        test_purpose="phase 4: frozen model metrics and the top-alert diagnostic, reports/model.md",
    )
    scored = {"valid": score_frame(bundles, valid), "test": score_frame(bundles, test)}
    results: dict[str, dict] = {}
    for split, frame in scored.items():
        for name, b in bundles.items():
            results.setdefault(name, {})[split] = summarise(
                frame["isFraud"],
                frame[f"score_{name}"],
                frame["TransactionAmt"],
                probability=b.is_probability,
            )
    for name, b in bundles.items():
        results[name]["calibration"] = b.calibrator.method if b.calibrator else None
    t.setup_mlflow(s)
    with mlflow.start_run(run_name="evaluation"):
        for name, by_split in results.items():
            for split in ("valid", "test"):
                m = by_split[split]
                mlflow.log_metrics(
                    {f"{name}_{split}_pr_auc": m["pr_auc"], f"{name}_{split}_roc_auc": m["roc_auc"]}
                )
    diagnostic = busiest_key_diagnostic(bundles, scored, valid, test)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_metrics.json").write_text(
        json.dumps(results | {"diagnostic": diagnostic}, indent=2) + "\n"
    )
    write_model_report(bundles, results, scored, valid, out_dir, diagnostic)
    return results


def busiest_key_diagnostic(bundles, scored, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    """How much of each model's top 1% of test alerts comes from the busiest pseudo-card.

    Aggregates only: the key itself is not reported.
    """
    counts = test["card_key"].value_counts()
    busiest = counts.index[0]
    out = {
        "busiest_key_rows": int(counts.iloc[0]),
        "busiest_key_fraud_rate": float(test.loc[test["card_key"] == busiest, "isFraud"].mean()),
        "share_card_txn_30d_over_100": {
            "valid": float((valid["card_txn_30d"] > 100).mean()),
            "test": float((test["card_txn_30d"] > 100).mean()),
        },
        "top_1pct": {},
    }
    frame = scored["test"]
    k = int(len(frame) * 0.01)
    for name in bundles:
        top = np.argsort(-frame[f"score_{name}"].to_numpy(), kind="stable")[:k]
        out["top_1pct"][name] = {
            "rows": k,
            "from_busiest_key": int((test["card_key"].to_numpy()[top] == busiest).sum()),
            "fraud_share": float(frame["isFraud"].to_numpy()[top].mean()),
        }
    return out


def rerender_report(s: Settings, out_dir: Path | None = None) -> None:
    """Rewrite reports/model.md from reports/model_metrics.json, without reading data."""
    from fraud.model.report import write_model_report

    out_dir = out_dir or reports_dir()
    saved = json.loads((out_dir / "model_metrics.json").read_text())
    diagnostic = saved.pop("diagnostic", None)
    write_model_report(load_bundles(s), saved, None, None, out_dir, diagnostic)
