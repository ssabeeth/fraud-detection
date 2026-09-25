"""The pre-registered experiments (DECISIONS.md, 2026-09-25), on the three folds.

Each experiment changes one thing against the current feature set and is scored on the
same folds as the model comparison (February, March and April, each once; early stopping
and calibration on the last 14 days of each training window). May is not read. Every
experiment uses LightGBM with the comparison's chosen setting, so only the thing under
test changes.

The candidate features are built here for experimenting, in pandas, from the December to
April frame and only from earlier transactions (tested in ``tests/test_experiments.py``).
A feature that passes the adoption rule is then built properly in the Spark and stream
feature code, with the point-in-time and parity tests.

Adoption rule, fixed before the run: cheaper than the current features in all three
months, and the 95% interval of the pooled saving (resampling whole days within each
month) excludes zero.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import Settings, reports_dir
from fraud.features.definitions import AGGREGATE_NAMES
from fraud.lakehouse.schema import C_COLS, D_COLS, V_COLS
from fraud.model import calibration
from fraud.model.compare import INNER_DAYS, LEARNING_RATE, MAX_ROUNDS, PATIENCE, folds
from fraud.model.data import load_frame
from fraud.model.inputs import FEATURE_SETS, Preprocessor
from fraud.model.metrics import summarise
from fraud.model.patterns import known_fraud_counts
from fraud.model.train import LGBM_BASE
from fraud.policy.costs import CostModel
from fraud.policy.policy import ExpectedLossPolicy

log = logging.getLogger(__name__)

DAY = 86_400
SETTING = {"num_leaves": 63, "min_child_samples": 200}  # the comparison's choice
D_DAYS = [c for c in D_COLS if c != "D9"]  # D9 is a fraction of a day, not a day count
D_NORM = [f"{c}n" for c in D_DAYS]
CARD_MEANS = [f"card_mean_{c}" for c in (*C_COLS, *D_NORM)]
LABEL_HISTORY = ["card_known_chargebacks", "card_known_labelled", "card_known_fraud_rate"]
RECENT_DAYS = 60
DRIFTING_DROPPED = 10
BOOTSTRAP = 1000


@dataclass(frozen=True)
class Experiment:
    name: str
    question: str
    pattern: str  # section of data_patterns.md that raised it
    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    recent_days: int | None = None
    params: dict = field(default_factory=dict)


def experiments(drifting: list[str]) -> list[Experiment]:
    base = FEATURE_SETS["all"]
    return [
        Experiment("baseline", "The current 452 features", "—"),
        Experiment(
            "no_aggregates",
            "Without the 17 point-in-time aggregates",
            "9",
            remove=tuple(AGGREGATE_NAMES),
        ),
        Experiment("no_v", "Without the 339 V columns", "5", remove=tuple(V_COLS)),
        Experiment(
            "d_normalised",
            "D columns normalised to dates (day − D)",
            "10",
            add=tuple(D_NORM),
            remove=tuple(D_DAYS),
        ),
        Experiment(
            "card_means",
            "Plus per-card means of earlier C and normalised D values",
            "9",
            add=tuple(CARD_MEANS),
        ),
        Experiment(
            "label_history",
            "Plus the card's chargebacks known after the 30-day delay",
            "9",
            add=tuple(LABEL_HISTORY),
        ),
        Experiment(
            "all_additions",
            "Normalised D, card means and label history together",
            "9, 10",
            add=(*D_NORM, *CARD_MEANS, *LABEL_HISTORY),
            remove=tuple(D_DAYS),
        ),
        Experiment(
            "drop_drifting",
            f"Without the {DRIFTING_DROPPED} most time-dependent features",
            "11",
            remove=tuple(f for f in drifting[:DRIFTING_DROPPED] if f in base),
        ),
        Experiment(
            "recent_window",
            f"Trained on the latest {RECENT_DAYS} days only",
            "2",
            recent_days=RECENT_DAYS,
        ),
        Experiment("weight_5x", "Fraud weighted five times", "1", params={"scale_pos_weight": 5.0}),
    ]


# --- candidate features, from earlier transactions only -----------------------------------


def add_candidates(df: pd.DataFrame, delay_seconds: int) -> pd.DataFrame:
    """Normalised D columns, per-card means of earlier values and delayed-label history.
    Rows must be in time order; a transaction never sees itself or anything at the same
    second, as in the project's feature definitions."""
    df = df.copy()
    day = df["TransactionDT"] // DAY
    for c, n in zip(D_DAYS, D_NORM, strict=True):
        df[n] = (day - df[c]).astype("float32")
    means = prior_means(df, "card_key", [*C_COLS, *D_NORM])
    for c in [*C_COLS, *D_NORM]:
        df[f"card_mean_{c}"] = means[c]
    df["card_known_chargebacks"] = known_fraud_counts(df, delay_seconds).astype("float32")
    df["card_known_labelled"] = known_counts(df, delay_seconds).astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = df["card_known_chargebacks"] / df["card_known_labelled"]
    df["card_known_fraud_rate"] = rate.where(df["card_known_labelled"] > 0).astype("float32")
    keyless = df["card_key"].isna()
    df.loc[keyless, LABEL_HISTORY] = np.nan
    return df


def prior_means(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Mean of each column over the key's transactions strictly earlier in time (same
    second excluded); missing values are skipped; NaN when there is none."""
    keyed = df[key].notna()
    work = df.loc[keyed, [key, "TransactionDT", *cols]]
    vals = work[cols].astype("float64")
    sums = vals.fillna(0).add_suffix("__s")
    counts = vals.notna().astype("int64").add_suffix("__n")
    parts = pd.concat([work[[key, "TransactionDT"]], sums, counts], axis=1)
    per_second = parts.groupby([key, "TransactionDT"], sort=True).sum()
    csum = per_second.groupby(level=0).cumsum() - per_second  # exclusive of this second
    out = pd.DataFrame(index=df.index, columns=cols, dtype="float32")
    idx = pd.MultiIndex.from_frame(work[[key, "TransactionDT"]])
    prior = csum.reindex(idx)
    for c in cols:
        s, n = prior[f"{c}__s"].to_numpy(), prior[f"{c}__n"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
        out.loc[keyed, c] = m.astype("float32")
    return out


def known_counts(df: pd.DataFrame, delay_seconds: int) -> np.ndarray:
    """Earlier transactions on the same card whose labels had arrived (at least
    ``delay_seconds`` before this one, strictly), fraud or not."""
    ones = df.assign(isFraud=1)
    return known_fraud_counts(ones, delay_seconds)


# --- running ------------------------------------------------------------------------------


def _fit_predict(xf, yf, xi, yi, xs, params: dict):
    import lightgbm as lgb

    p = {**LGBM_BASE, "learning_rate": LEARNING_RATE, "scale_pos_weight": 1.0}
    p = {**p, **SETTING, **params}
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
    return b.predict(xi, num_iteration=n), b.predict(xs, num_iteration=n), n


def daily_costs(frame: pd.DataFrame, p: np.ndarray, costs: CostModel) -> np.ndarray:
    """Cost of the untuned expected-loss policy, day by day (the queue is per day)."""
    f = frame.assign(p=p)
    actions = ExpectedLossPolicy("p").decide(f, costs)
    y, amt = f["isFraud"].to_numpy(), f["TransactionAmt"].to_numpy()
    day = pd.to_datetime(f["event_date"]).to_numpy()
    out = []
    for d in np.unique(day):
        m = day == d
        out.append(costs.realised(actions[m], y[m], amt[m])["total_cost"])
    return np.asarray(out)


def run(spark, s: Settings, out_dir: Path | None = None, names: list[str] | None = None) -> dict:
    out_dir = out_dir or reports_dir()
    patterns = json.loads((out_dir / "data_patterns.json").read_text())
    drifting = [t["feature"] for t in patterns["adversarial"]["top"]]
    costs = CostModel.load()
    df = load_frame(spark, s, ["train", "valid"])  # never the test month
    df = add_candidates(df, s.label_delay_days * DAY)
    chosen = [e for e in experiments(drifting) if names is None or e.name in names]
    if chosen[0].name != "baseline":
        chosen = [experiments(drifting)[0], *chosen]
    base_features = FEATURE_SETS["all"]
    results: dict[str, list[dict]] = {e.name: [] for e in chosen}
    daily: dict[str, list[list[float]]] = {e.name: [] for e in chosen}
    fold_list = folds(df)
    for f in fold_list:
        for e in chosen:
            features = [c for c in base_features if c not in set(e.remove)] + list(e.add)
            fit = f.fit
            if e.recent_days:
                start = pd.to_datetime(f.inner["event_date"]).max() - pd.Timedelta(
                    days=e.recent_days - 1
                )
                fit = fit[pd.to_datetime(fit["event_date"]) >= start]
            pre = Preprocessor(features).fit(fit)
            xf, xi, xs = pre.transform(fit), pre.transform(f.inner), pre.transform(f.score)
            yf, yi = fit["isFraud"].to_numpy(), f.inner["isFraud"].to_numpy()
            ys, amt = f.score["isFraud"].to_numpy(), f.score["TransactionAmt"].to_numpy()
            raw_i, raw_s, rounds = _fit_predict(xf, yf, xi, yi, xs, e.params)
            cal, _ = calibration.choose(raw_i, yi, f.inner["TransactionDT"].to_numpy())
            p = cal(raw_s)
            m = summarise(ys, p, amt)
            per_day = daily_costs(f.score, p, costs)
            daily[e.name].append(per_day.tolist())
            results[e.name].append(
                {
                    "scored": f.scored,
                    "features": len(features),
                    "fit_rows": len(fit),
                    "pr_auc": m["pr_auc"],
                    "total_cost": float(per_day.sum()),
                    "rounds": int(rounds),
                }
            )
            log.info(
                "%s %s: %d features, PR-AUC %.4f, cost $%s",
                f.scored,
                e.name,
                len(features),
                m["pr_auc"],
                f"{per_day.sum():,.0f}",
            )
    result = {
        "experiments": [
            {
                "name": e.name,
                "question": e.question,
                "pattern": e.pattern,
                "added": list(e.add),
                "removed": list(e.remove),
                "folds": results[e.name],
                **compare_to_baseline(daily[e.name], daily["baseline"], results, e.name),
            }
            for e in chosen
        ],
        "folds": [f.scored for f in fold_list],
        "setting": SETTING,
        "learning_rate": LEARNING_RATE,
        "inner_days": INNER_DAYS,
        "bootstrap_draws": BOOTSTRAP,
        "rule": "adopt only if cheaper in every month and the 95% interval of the pooled "
        "saving excludes zero",
    }
    (out_dir / "experiments.json").write_text(json.dumps(result, indent=2) + "\n")
    from fraud.model.experiments_report import write_report

    write_report(result, out_dir)
    return result


def compare_to_baseline(exp_days, base_days, results, name, seed: int = 7) -> dict:
    """Saving against the baseline (positive = cheaper), per month and pooled, with a 95%
    interval from resampling whole days within each month."""
    saving = [float(np.sum(b) - np.sum(e)) for e, b in zip(exp_days, base_days, strict=True)]
    rng = np.random.default_rng(seed)
    draws = np.zeros(BOOTSTRAP)
    for e, b in zip(exp_days, base_days, strict=True):
        diff = np.asarray(b) - np.asarray(e)
        idx = rng.integers(0, len(diff), size=(BOOTSTRAP, len(diff)))
        draws += diff[idx].sum(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    base_pr = [r["pr_auc"] for r in results["baseline"]]
    pr = [r["pr_auc"] for r in results[name]]
    adopted = name != "baseline" and all(x > 0 for x in saving) and lo > 0
    return {
        "saving_by_month": saving,
        "pooled_saving": float(sum(saving)),
        "saving_interval": [float(lo), float(hi)],
        "pr_auc_change": [a - b for a, b in zip(pr, base_pr, strict=True)],
        "mean_pr_auc": float(np.mean(pr)),
        "mean_cost": float(np.mean([r["total_cost"] for r in results[name]])),
        "adopted": bool(adopted),
    }
