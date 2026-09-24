"""Train the rules baseline, logistic regression and LightGBM on months 1-4.

All tuning uses the validation month only. Order matters and is kept: the rules and
logistic regression are fitted and measured before LightGBM, and every LightGBM result
is reported against them. Imbalance is handled by weighting (``class_weight`` and
``scale_pos_weight``, both tuned), never by oversampling; see DECISIONS.md.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud.config import Settings
from fraud.features.definitions import AGGREGATE_NAMES
from fraud.model import calibration, rules
from fraud.model.bundle import ModelBundle, as_strings
from fraud.model.inputs import FEATURE_SETS, Preprocessor
from fraud.model.metrics import summarise

log = logging.getLogger(__name__)

EXPERIMENT = "fraud-models"

LGBM_BASE = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    # Every decision carries exact TreeSHAP reasons, whose cost grows with trees × leaves ×
    # depth². Unconstrained leaf-wise trees reached depth 57 and 162 ms per decision; a
    # depth limit keeps the reasons within the streaming budget (DECISIONS.md).
    "max_depth": 8,
    "cat_smooth": 10.0,
    "min_data_per_group": 50,
    "max_cat_threshold": 32,
    "feature_pre_filter": False,
    "verbosity": -1,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
}
LGBM_GRID = [
    {"num_leaves": nl, "min_child_samples": mcs, "scale_pos_weight": spw}
    for nl in (63, 255)
    for mcs in (50, 200)
    for spw in (1.0, 5.0)
]
LOGREG_GRID = [
    {"C": c, "class_weight": cw} for c in (0.01, 0.1, 1.0, 10.0) for cw in (None, "balanced")
]

# Logistic regression sees the explainable set minus the masked card and address codes,
# which are identifiers, not quantities, and mean nothing on a linear scale.
LOGREG_NUMERIC = [
    *AGGREGATE_NAMES,
    "TransactionAmt",
    "amt_cents",
    "dist1",
    "email_match",
    "has_identity",
]
LOGREG_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "hour",
    "weekday",
]


def setup_mlflow(s: Settings) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from fraud.spark import on_databricks

    if on_databricks():
        # The workspace's own tracking server and the Unity Catalog model registry.
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(os.environ.get("FRAUD_MLFLOW_EXPERIMENT", "/Shared/fraud-models"))
        return
    root = s.path("mlflow")
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _log_metrics(prefix: str, m: dict) -> None:
    flat = {f"{prefix}_pr_auc": m["pr_auc"], f"{prefix}_roc_auc": m["roc_auc"]}
    for r in m["recall_at_fpr"]:
        tag = f"{r['fpr_target'] * 100:g}pct".replace(".", "_")
        flat[f"{prefix}_recall_at_fpr_{tag}"] = r["recall"]
        flat[f"{prefix}_value_recall_at_fpr_{tag}"] = r["value_recall"]
    for k in ("brier", "ece"):
        if k in m:
            flat[f"{prefix}_{k}"] = m[k]
    mlflow.log_metrics(flat)


# --- rules ---------------------------------------------------------------------------


def fit_rules(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[ModelBundle, dict]:
    with mlflow.start_run(run_name="rules"):
        best, trials = rules.tune(valid)
        bundle = ModelBundle("rules", "rules", "explainable", best)
        m = summarise(
            valid["isFraud"], bundle.predict(valid), valid["TransactionAmt"], probability=False
        )
        mlflow.log_params(asdict(best) | {"grid_size": len(trials)})
        _log_metrics("valid", m)
    log.info("rules: valid PR-AUC %.4f with %s", m["pr_auc"], best)
    return bundle, m


# --- logistic regression -------------------------------------------------------------


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _logreg_pipeline(C: float, class_weight) -> Pipeline:
    numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        FunctionTransformer(_signed_log, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100),
    )
    pre = ColumnTransformer(
        [("num", numeric, LOGREG_NUMERIC), ("cat", categorical, LOGREG_CATEGORICAL)]
    )
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000)
    return Pipeline([("pre", pre), ("clf", clf)])


def _logreg_frame(pre: Preprocessor, df: pd.DataFrame) -> pd.DataFrame:
    return as_strings(pre.transform(df), LOGREG_CATEGORICAL)


def fit_logreg(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[ModelBundle, dict]:
    pre = Preprocessor([*LOGREG_NUMERIC, *LOGREG_CATEGORICAL]).fit(train)
    xtr, xva = _logreg_frame(pre, train), _logreg_frame(pre, valid)
    ytr, yva = train["isFraud"].to_numpy(), valid["isFraud"].to_numpy()
    best = None
    with mlflow.start_run(run_name="logreg"):
        for params in LOGREG_GRID:
            with mlflow.start_run(
                run_name=f"logreg C={params['C']} cw={params['class_weight']}", nested=True
            ):
                t0 = time.perf_counter()
                pipe = _logreg_pipeline(**params).fit(xtr, ytr)
                ap = average_precision_score(yva, pipe.predict_proba(xva)[:, 1])
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({"valid_pr_auc": ap, "fit_seconds": time.perf_counter() - t0})
            log.info("logreg %s: valid PR-AUC %.4f", params, ap)
            if best is None or ap > best[0]:
                best = (ap, params, pipe)
        ap, params, pipe = best
        raw = pipe.predict_proba(xva)[:, 1]
        cal, cal_scores = calibration.choose(raw, yva, valid["TransactionDT"].to_numpy())
        bundle = ModelBundle(
            "logreg",
            "logreg",
            "explainable",
            pipe,
            pre,
            cal,
            meta={
                "params": params,
                "calibration_brier": cal_scores,
                "string_columns": LOGREG_CATEGORICAL,
            },
        )
        m = summarise(yva, bundle.predict(valid), valid["TransactionAmt"])
        mlflow.log_params(
            {f"best_{k}": str(v) for k, v in params.items()} | {"calibration": cal.method}
        )
        _log_metrics("valid", m)
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            registered_model_name="fraud-logreg",
            # skops refuses unknown callables unless they are named as trusted.
            skops_trusted_types=[f"{__name__}._signed_log", "numpy.dtype"],
        )
    return bundle, m


# --- LightGBM ------------------------------------------------------------------------


def fit_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_set: str = "all",
    name: str = "lightgbm",
    register: bool = True,
) -> tuple[ModelBundle, dict]:
    import lightgbm as lgb

    pre = Preprocessor(FEATURE_SETS[feature_set]).fit(train)
    xtr, xva = pre.transform(train), pre.transform(valid)
    ytr, yva = train["isFraud"].to_numpy(), valid["isFraud"].to_numpy()
    dtr = lgb.Dataset(xtr, ytr, free_raw_data=False)
    dva = lgb.Dataset(xva, yva, reference=dtr, free_raw_data=False)
    best = None
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"feature_set": feature_set, "features": len(pre.features)})
        for params in LGBM_GRID:
            label = " ".join(f"{k}={v}" for k, v in params.items())
            with mlflow.start_run(run_name=f"{name} {label}", nested=True):
                t0 = time.perf_counter()
                booster = lgb.train(
                    {**LGBM_BASE, **params},
                    dtr,
                    num_boost_round=4000,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(150, verbose=False)],
                )
                ap = average_precision_score(yva, booster.predict(xva))
                mlflow.log_params(params | {"best_iteration": booster.best_iteration})
                mlflow.log_metrics({"valid_pr_auc": ap, "fit_seconds": time.perf_counter() - t0})
            log.info(
                "%s %s: valid PR-AUC %.4f at %d trees", name, label, ap, booster.best_iteration
            )
            if best is None or ap > best[0]:
                best = (ap, params, booster)
        ap, params, booster = best
        booster = lgb.Booster(
            model_str=booster.model_to_string(num_iteration=booster.best_iteration)
        )
        raw = booster.predict(xva)
        cal, cal_scores = calibration.choose(raw, yva, valid["TransactionDT"].to_numpy())
        bundle = ModelBundle(
            name,
            "lightgbm",
            feature_set,
            booster,
            pre,
            cal,
            meta={
                "params": {**LGBM_BASE, **params},
                "trees": booster.num_trees(),
                "calibration_brier": cal_scores,
            },
        )
        m = summarise(yva, bundle.predict(valid), valid["TransactionAmt"])
        mlflow.log_params({f"best_{k}": v for k, v in params.items()} | {"calibration": cal.method})
        _log_metrics("valid", m)
        if register:
            mlflow.lightgbm.log_model(booster, name="model", registered_model_name=f"fraud-{name}")
    return bundle, m


def save(bundle: ModelBundle, models_dir: Path) -> Path:
    path = bundle.save(models_dir)
    if mlflow.active_run() is None:
        with mlflow.start_run(run_name=f"{bundle.name} bundle"):
            mlflow.log_artifacts(str(path), artifact_path="bundle")
    return path
