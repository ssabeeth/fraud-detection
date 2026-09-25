"""Rolling-origin comparison of LightGBM, XGBoost and CatBoost on the training months.

The project's LightGBM settings were chosen on one month (April). This checks, on three
months and without reading the test month (May):

- whether LightGBM is the right library, against XGBoost and CatBoost given the same
  features, the same weighting and grids of the same size;
- how far results move from month to month, so a gap between libraries can be judged
  against that spread.

Each fold scores one month once: train on December-January and score February,
December-February and March, December-March and April. Inside a fold, early stopping
and calibration use only the last 14 days of its training window. Every library learns
at the same rate (0.1, to keep the run to about an hour) and is judged by PR-AUC and by
the money of the same untuned expected-loss policy (review whenever the expected saving
is positive, within the daily capacity), so nothing is tuned on the month being scored.
The frozen model and policy are not changed by this module.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import Settings, reports_dir
from fraud.model import calibration
from fraud.model.data import load_frame
from fraud.model.inputs import FEATURE_SETS, Preprocessor
from fraud.model.metrics import summarise
from fraud.model.train import LGBM_BASE
from fraud.policy.costs import CostModel
from fraud.policy.policy import ApproveAll, ExpectedLossPolicy, cost

log = logging.getLogger(__name__)

LIBRARIES = ("lightgbm", "xgboost", "catboost")
SCORED_MONTHS = ("2018-02", "2018-03", "2018-04")
LEARNING_RATE = 0.1
MAX_ROUNDS = 3000
PATIENCE = 100
INNER_DAYS = 14
SEED = 7
GRIDS: dict[str, list[dict]] = {
    "lightgbm": [{"num_leaves": n, "min_child_samples": m} for n in (63, 255) for m in (50, 200)],
    "xgboost": [{"max_depth": d, "min_child_weight": w} for d in (6, 8) for w in (1, 10)],
    "catboost": [{"depth": d, "l2_leaf_reg": r} for d in (6, 8) for r in (3.0, 10.0)],
}


@dataclass
class Fold:
    scored: str
    fit: pd.DataFrame
    inner: pd.DataFrame
    score: pd.DataFrame


def folds(df: pd.DataFrame, months: tuple[str, ...] = SCORED_MONTHS) -> list[Fold]:
    """Expanding-window folds; the last ``INNER_DAYS`` of each training window are held
    out for early stopping and calibration. Rows must be in time order."""
    month = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m")
    day = pd.to_datetime(df["event_date"])
    out = []
    for m in months:
        train = df[month < m]
        cut = day[month < m].max() - pd.Timedelta(days=INNER_DAYS - 1)
        inner_mask = day[month < m] >= cut
        out.append(
            Fold(
                scored=m,
                fit=train[~inner_mask.to_numpy()],
                inner=train[inner_mask.to_numpy()],
                score=df[month == m],
            )
        )
    return out


# --- the three libraries, behind one interface -------------------------------------------


def _fit(library: str, params: dict, xf, yf, xi, yi, categorical: list[str]):
    """Return (predict, best_iteration, explain_one) for a model fitted with early
    stopping on the inner window. ``explain_one`` scores one row with its SHAP values."""
    if library == "lightgbm":
        import lightgbm as lgb

        p = {**LGBM_BASE, "learning_rate": LEARNING_RATE, "scale_pos_weight": 1.0, **params}
        dtr = lgb.Dataset(xf, yf, free_raw_data=False)
        dva = lgb.Dataset(xi, yi, reference=dtr, free_raw_data=False)
        b = lgb.train(
            p,
            dtr,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
        )
        n = b.best_iteration
        return (
            lambda x: b.predict(x, num_iteration=n),
            n,
            lambda row: b.predict(row, num_iteration=n, pred_contrib=True),
        )
    if library == "xgboost":
        import xgboost as xgb

        p = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "eta": LEARNING_RATE,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "lambda": 1.0,
            "max_cat_to_onehot": 1,
            "seed": SEED,
            **params,
        }
        dtr = xgb.DMatrix(xf, yf, enable_categorical=True)
        dva = xgb.DMatrix(xi, yi, enable_categorical=True)
        b = xgb.train(
            p,
            dtr,
            MAX_ROUNDS,
            evals=[(dva, "inner")],
            early_stopping_rounds=PATIENCE,
            verbose_eval=False,
        )
        rng = (0, b.best_iteration + 1)

        def matrix(x):
            return xgb.DMatrix(x, enable_categorical=True)

        return (
            lambda x: b.predict(matrix(x), iteration_range=rng),
            b.best_iteration + 1,
            lambda row: b.predict(matrix(row), iteration_range=rng, pred_contribs=True),
        )
    if library == "catboost":
        from catboost import CatBoostClassifier, Pool

        def pool(x, y=None):
            x = x.copy()
            for c in categorical:
                x[c] = x[c].astype(str)
            return Pool(x, y, cat_features=categorical)

        m = CatBoostClassifier(
            iterations=MAX_ROUNDS,
            learning_rate=LEARNING_RATE,
            eval_metric="PRAUC",
            od_type="Iter",
            od_wait=PATIENCE,
            random_seed=SEED,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        m.fit(pool(xf, yf), eval_set=pool(xi, yi), use_best_model=True)
        return (
            lambda x: m.predict_proba(pool(x))[:, 1],
            m.get_best_iteration() + 1,
            lambda row: m.get_feature_importance(pool(row), type="ShapValues"),
        )
    raise ValueError(library)


def _label(params: dict) -> str:
    return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def run(
    spark, s: Settings, out_dir: Path | None = None, libraries: tuple[str, ...] = LIBRARIES
) -> dict:
    costs = CostModel.load()
    df = load_frame(spark, s, ["train", "valid"])  # never the test month
    pre_features = FEATURE_SETS["all"]
    results: dict[str, dict[str, list[dict]]] = {lib: {} for lib in libraries}
    explainers: dict[tuple[str, str], object] = {}
    fold_info = []
    last_x = None
    for f in folds(df):
        pre = Preprocessor(pre_features).fit(f.fit)
        xf, xi, xs = pre.transform(f.fit), pre.transform(f.inner), pre.transform(f.score)
        yf, yi = f.fit["isFraud"].to_numpy(), f.inner["isFraud"].to_numpy()
        ys, amount = f.score["isFraud"].to_numpy(), f.score["TransactionAmt"].to_numpy()
        approve_all = cost(ApproveAll(), f.score, costs)["total_cost"]
        fold_info.append(
            {
                "scored": f.scored,
                "fit_rows": len(f.fit),
                "inner_rows": len(f.inner),
                "fit_last_day": str(pd.to_datetime(f.fit["event_date"]).max().date()),
                "inner_days": [
                    str(pd.to_datetime(f.inner["event_date"]).min().date()),
                    str(pd.to_datetime(f.inner["event_date"]).max().date()),
                ],
                "scored_rows": len(f.score),
                "scored_fraud": int(ys.sum()),
                "approve_all_cost": approve_all,
            }
        )
        for lib in libraries:
            for params in GRIDS[lib]:
                label = _label(params)
                t0 = time.perf_counter()
                predict, rounds, explain = _fit(lib, params, xf, yf, xi, yi, pre.categorical)
                fit_s = time.perf_counter() - t0
                cal, _ = calibration.choose(predict(xi), yi, f.inner["TransactionDT"].to_numpy())
                p = cal(predict(xs))
                m = summarise(ys, p, amount)
                money = cost(ExpectedLossPolicy("p"), f.score.assign(p=p), costs)
                results[lib].setdefault(label, []).append(
                    {
                        "scored": f.scored,
                        "pr_auc": m["pr_auc"],
                        "roc_auc": m["roc_auc"],
                        "recall_at_1pct_fpr": next(
                            r["recall"] for r in m["recall_at_fpr"] if r["fpr_target"] == 0.01
                        ),
                        "total_cost": money["total_cost"],
                        "fraud_value_caught_share": money["fraud_value_caught_share"],
                        "calibration": cal.method,
                        "rounds": int(rounds),
                        "fit_seconds": round(fit_s, 1),
                    }
                )
                log.info(
                    "%s %s %s: PR-AUC %.4f, cost $%s, %d rounds, %.0fs",
                    f.scored,
                    lib,
                    label,
                    m["pr_auc"],
                    f"{money['total_cost']:,.0f}",
                    rounds,
                    fit_s,
                )
                if f.scored == SCORED_MONTHS[-1]:
                    explainers[(lib, label)] = explain
        last_x = xs
    result = summarise_runs(results, fold_info)
    result["n_features"] = len(pre_features)
    result["latency_ms"] = {
        lib: _latency(explainers[(lib, result["best"][lib])], last_x)
        for lib in libraries
        if (lib, result["best"][lib]) in explainers
    }
    result["settings"] = {
        "learning_rate": LEARNING_RATE,
        "max_rounds": MAX_ROUNDS,
        "patience": PATIENCE,
        "inner_days": INNER_DAYS,
        "grids": {lib: [_label(g) for g in GRIDS[lib]] for lib in libraries},
        "policy": "expected loss, review whenever the expected saving is positive (untuned)",
    }
    out_dir = out_dir or reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    from fraud.model.compare_report import write_report

    write_report(result, out_dir)
    return result


def summarise_runs(results: dict, fold_info: list[dict]) -> dict:
    """Pick each library's setting by mean PR-AUC over the folds and compare in money,
    fold by fold, against LightGBM."""
    best, summary = {}, {}
    for lib, by_setting in results.items():
        label = max(by_setting, key=lambda k: np.mean([r["pr_auc"] for r in by_setting[k]]))
        best[lib] = label
        runs = by_setting[label]
        summary[lib] = {
            "setting": label,
            "pr_auc": [r["pr_auc"] for r in runs],
            "total_cost": [r["total_cost"] for r in runs],
            "caught_share": [r["fraud_value_caught_share"] for r in runs],
            "mean_pr_auc": float(np.mean([r["pr_auc"] for r in runs])),
            "mean_cost": float(np.mean([r["total_cost"] for r in runs])),
            "fit_seconds": float(np.sum([r["fit_seconds"] for r in runs])),
        }
    if "lightgbm" in summary:
        base = summary["lightgbm"]
        for s_ in summary.values():
            for key, out in (("total_cost", "cost_vs_lightgbm"), ("pr_auc", "pr_auc_vs_lightgbm")):
                s_[out] = [a - b for a, b in zip(s_[key], base[key], strict=True)]
    all_settings = {
        lib: {
            label: {
                "mean_pr_auc": float(np.mean([r["pr_auc"] for r in runs])),
                "mean_cost": float(np.mean([r["total_cost"] for r in runs])),
            }
            for label, runs in by_setting.items()
        }
        for lib, by_setting in results.items()
    }
    return {
        "folds": fold_info,
        "best": best,
        "summary": summary,
        "all_settings": all_settings,
        "runs": results,
    }


def _latency(explain, x: pd.DataFrame, n: int = 200) -> float:
    """Median milliseconds to score one row with its SHAP values."""
    rows = x.sample(n=min(n, len(x)), random_state=SEED)
    times = []
    for i in range(len(rows)):
        row = rows.iloc[[i]]
        t0 = time.perf_counter()
        explain(row)
        times.append((time.perf_counter() - t0) * 1e3)
    return round(float(np.median(times)), 2)
