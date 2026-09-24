"""``fraud explain``: the explainable-features experiment, segment checks and global SHAP.

1. Train LightGBM on the explainable features only (history aggregates and readable
   fields; none of Vesta's masked columns), with the same grid, early stopping and
   calibration as the main model.
2. Tune the expected-loss policy for it on validation, exactly as phase 5 did for the
   main model, and cost both on the test month: the difference is what it costs in money
   to give every decision a reason a reviewer can check.
3. Segment checks for the frozen policy on the test month: alert rate, precision and
   recall by product, card network, card type, device type and email domain.
4. Global importance: mean |SHAP| on the validation month for both models, and how often
   each feature family is the top reason for an alert.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import Settings, reports_dir
from fraud.explain.reasons import plain_name
from fraud.model import train as t
from fraud.model.bundle import ModelBundle
from fraud.model.data import load_frame
from fraud.model.metrics import summarise
from fraud.model.pipeline import models_dir
from fraud.policy import policy as P
from fraud.policy.costs import APPROVE, CostModel
from fraud.stream.scorer import FrozenPolicy

log = logging.getLogger(__name__)

SEGMENTS = ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain"]
TOP_DOMAINS = 8


def family(feature: str) -> str:
    if feature.startswith("id_") or feature == "DeviceInfo":
        return "identity and device fields (masked)"
    if feature[0] in "CDMV" and feature[1:].isdigit():
        return {
            "C": "C: linked-entity counts (masked)",
            "D": "D: time deltas (masked)",
            "M": "M: match checks (masked)",
            "V": "V: Vesta signals (masked)",
        }[feature[0]]
    if feature.startswith(("card_", "device_", "email_")):
        return "history aggregates (this project)"
    return "readable transaction fields"


def mean_abs_shap(bundle: ModelBundle, frame: pd.DataFrame, n: int = 10_000) -> pd.Series:
    sample = frame.sample(n=min(n, len(frame)), random_state=0)
    contrib = bundle.contributions(sample)
    return contrib.abs().mean().sort_values(ascending=False)


def segment_table(frame: pd.DataFrame, actions: np.ndarray, costs: CostModel) -> list[dict]:
    f = frame.assign(_alert=(actions != APPROVE), _action=actions)
    domains = f["P_emaildomain"].value_counts().head(TOP_DOMAINS).index
    f["P_emaildomain"] = np.where(
        f["P_emaildomain"].isna(),
        "(missing)",
        np.where(f["P_emaildomain"].isin(domains), f["P_emaildomain"], "(other)"),
    )
    rows = []
    for seg in SEGMENTS:
        col = f[seg].astype(object).where(f[seg].notna(), "(missing)")
        for value, g in f.groupby(col, sort=False):
            fraud = g["isFraud"].to_numpy().astype(bool)
            alert = g["_alert"].to_numpy()
            r = costs.realised(g["_action"].to_numpy(), fraud, g["TransactionAmt"].to_numpy())
            rows.append(
                {
                    "segment": seg,
                    "value": str(value),
                    "transactions": len(g),
                    "fraud_rate": float(fraud.mean()),
                    "alert_rate": float(alert.mean()),
                    "precision": float(fraud[alert].mean()) if alert.any() else None,
                    "recall": float(alert[fraud].mean()) if fraud.any() else None,
                    "value_caught": r["fraud_value_caught_share"] if fraud.any() else None,
                }
            )
    return rows


def run(spark, s: Settings, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or reports_dir()
    costs = CostModel.load()
    frozen = FrozenPolicy.load()
    train = load_frame(spark, s, ["train"])
    valid = load_frame(spark, s, ["valid"])
    t.setup_mlflow(s)
    expl, expl_valid = t.fit_lightgbm(
        train, valid, feature_set="explainable", name="lightgbm_explainable"
    )
    t.save(expl, models_dir(s))
    del train
    full = ModelBundle.load(models_dir(s) / "lightgbm")

    vs = valid.copy()
    vs["score_lightgbm"] = full.predict(valid)
    vs["score_lightgbm_explainable"] = expl.predict(valid)
    pol_full = P.ExpectedLossPolicy("score_lightgbm", frozen.review_threshold)
    pol_expl, _ = P.tune_expected_loss(vs, costs, "score_lightgbm_explainable")

    test = load_frame(
        spark,
        s,
        ["test"],
        test_purpose="phase 6: explainable-only model vs all features, and segment checks",
    )
    ts = test.copy()
    ts["score_lightgbm"] = full.predict(test)
    ts["score_lightgbm_explainable"] = expl.predict(test)
    comparison = {}
    for name, pol, col in (
        ("all", pol_full, "score_lightgbm"),
        ("explainable", pol_expl, "score_lightgbm_explainable"),
    ):
        comparison[name] = {
            "features": len((full if name == "all" else expl).preprocessor.features),
            "valid": P.cost(pol, vs, costs)
            | {"pr_auc": summarise(vs["isFraud"], vs[col], vs["TransactionAmt"])["pr_auc"]},
            "test": P.cost(pol, ts, costs)
            | {"pr_auc": summarise(ts["isFraud"], ts[col], ts["TransactionAmt"])["pr_auc"]},
            "review_threshold": pol.review_threshold,
        }
    actions = pol_full.decide(ts, costs)
    segments = segment_table(ts, actions, costs)

    shap_full = mean_abs_shap(full, valid)
    shap_expl = mean_abs_shap(expl, valid)
    alerts = ts[actions != APPROVE]
    contrib = full.contributions(alerts)
    top = contrib.idxmax(axis=1)
    top_family = top.map(family).value_counts(normalize=True).to_dict()
    top_feature = top.value_counts(normalize=True).head(10)

    result = {
        "comparison": comparison,
        "segments": segments,
        "shap_all_top20": [
            {"feature": f, "name": plain_name(f), "family": family(f), "mean_abs_shap": float(v)}
            for f, v in shap_full.head(20).items()
        ],
        "shap_explainable_top20": [
            {"feature": f, "name": plain_name(f), "mean_abs_shap": float(v)}
            for f, v in shap_expl.head(20).items()
        ],
        "shap_share_by_family": {
            k: float(v)
            for k, v in (shap_full.groupby(shap_full.index.map(family)).sum() / shap_full.sum())
            .sort_values(ascending=False)
            .items()
        },
        "alerts": len(alerts),
        "top_reason_family_share": top_family,
        "top_reason_feature_share": [
            {"feature": f, "name": plain_name(f), "share": float(v)} for f, v in top_feature.items()
        ],
        "explainable_params": expl.meta["params"],
        "explainable_trees": expl.meta["trees"],
        "explainable_calibration": expl.calibrator.method,
        "explainable_valid_pr_auc": expl_valid["pr_auc"],
    }
    (out_dir / "explainability.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    from fraud.explain.report import write_explain_report

    write_explain_report(result, out_dir)
    return result
