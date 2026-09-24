"""``fraud monitor``: drift and delayed-label performance on the replayed stream.

Inputs are what production would have: the decision log and the raw transactions that
the stream landed in Delta bronze, and the labels topic, where each label appears 30 days
after its transaction. The reference is the validation month, scored offline with the
same model and policy.

Each simulated day ``T``:

- **input drift**: Evidently's per-column drift tests on the trailing 7 days of inputs;
- **score drift**: PSI of P(fraud) against validation deciles;
- **alert rate**: declines plus reviews as a share of transactions.

Each weekly cohort of transactions gets **performance** (PR-AUC, fraud value caught by
the policy) only from labels that have arrived by ``T``; a cohort is complete 30 days
after its last day. The same monitors also run on a stress scenario, replayed in-process
through the same ``OnlineScorer``: the identity feed (device and browser fields) goes
silent for the last 10 days of May, as when an upstream fingerprinting service fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraud.config import Settings, config_dir
from fraud.policy.costs import ACTION_NAMES, CostModel

log = logging.getLogger(__name__)

NUMERIC = [
    "p_fraud",
    "TransactionAmt",
    "card_txn_24h",
    "card_amt_24h",
    "card_txn_prior",
    "card_secs_since_prev",
    "card_amt_zscore",
    "device_txn_24h",
    "email_txn_7d",
]
CATEGORICAL = ["ProductCD", "card4", "card6", "DeviceType", "has_identity", "email_domain"]
IDENTITY_FIELDS = ["DeviceType", "DeviceInfo", *(f"id_{i:02d}" for i in range(1, 39))]


@dataclass(frozen=True)
class Thresholds:
    window_days: int
    drifted_share_warn: float
    psi_warn: float
    psi_alert: float
    alert_rate_change_warn: float
    pr_auc_ratio_alert: float
    value_caught_drop_alert: float
    consecutive_score_alert_days: int
    feed_min_present: float = 0.05
    feed_drop: float = 0.5

    @classmethod
    def load(cls, path: Path | None = None) -> Thresholds:
        c = yaml.safe_load((path or config_dir() / "monitoring.yaml").read_text())
        return cls(
            window_days=c["window_days"],
            drifted_share_warn=c["input_drift"]["drifted_share_warn"],
            psi_warn=c["score_drift"]["psi_warn"],
            psi_alert=c["score_drift"]["psi_alert"],
            alert_rate_change_warn=c["alert_rate"]["relative_change_warn"],
            pr_auc_ratio_alert=c["performance"]["pr_auc_ratio_alert"],
            value_caught_drop_alert=c["performance"]["value_caught_drop_alert"],
            consecutive_score_alert_days=c["retrain"]["consecutive_score_alert_days"],
            feed_min_present=c["feed_health"]["min_present"],
            feed_drop=c["feed_health"]["drop"],
        )


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index on the reference's quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / max(len(current), 1)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _prepare(frame: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    f = frame.copy()
    f["has_identity"] = f["has_identity"].fillna(False).astype(bool).astype(str)
    f["email_domain"] = np.where(
        f["P_emaildomain"].isna(),
        "(missing)",
        np.where(f["P_emaildomain"].isin(domains), f["P_emaildomain"], "(other)"),
    )
    for c in CATEGORICAL:
        f[c] = f[c].astype(object).where(f[c].notna(), "(missing)").astype(str)
    for c in NUMERIC:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f


MISSING_SHIFT = 0.2  # absolute change in a column's missing share that counts as drift


def input_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Evidently's per-column drift tests, plus a missing-values check.

    A feed that goes silent shows up as missing values, which a distribution test on the
    values that remain cannot see (and Evidently refuses an all-empty column). So a column
    also counts as drifted when its missing share moves by ``MISSING_SHIFT`` or more.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    per_column: dict[str, dict] = {}
    for c in NUMERIC:
        change = float(current[c].isna().mean() - reference[c].isna().mean())
        per_column[c] = {"missing_change": change, "drifted": abs(change) >= MISSING_SHIFT}
    numeric = [c for c in NUMERIC if current[c].notna().any() and reference[c].notna().any()]
    cols = [*numeric, *CATEGORICAL]
    dd = DataDefinition(numerical_columns=numeric, categorical_columns=CATEGORICAL)
    snap = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current[cols], data_definition=dd),
        Dataset.from_pandas(reference[cols], data_definition=dd),
    )
    for m in snap.dict()["metrics"]:
        name = m["metric_name"]
        if name.startswith("ValueDrift(column="):
            col = name.split("column=")[1].split(",")[0]
            threshold = float(name.split("threshold=")[1].rstrip(")"))
            entry = per_column.setdefault(col, {"drifted": False})
            entry["value"] = float(m["value"])
            entry["drifted"] = entry["drifted"] or m["value"] >= threshold
    share = sum(v["drifted"] for v in per_column.values()) / len(per_column)
    return {"drifted_share": float(share), "columns": per_column, "snapshot": snap}


def present_share(frame: pd.DataFrame, column: str) -> float:
    """Share of rows where the field is present (``has_identity``: share with a record)."""
    if column == "has_identity":
        return float((frame[column] == "True").mean())
    if column in CATEGORICAL:
        return float((frame[column] != "(missing)").mean())
    return float(frame[column].notna().mean())


def feed_health(
    reference: pd.DataFrame, day: pd.DataFrame, min_present: float = 0.05, drop: float = 0.5
) -> list[str]:
    """Fields that went (mostly) silent on the latest day.

    A failed upstream feed shows up as a field that is suddenly absent. The 7-day drift
    window sees it only once most of the window is affected, and an absolute change in the
    missing share misses fields that are usually missing (identity fields are absent for
    three quarters of transactions). So the check compares each field's presence on the
    last day with its presence in the reference. A feed alert calls for fixing the feed,
    not for retraining on its broken output.
    """
    if day.empty:
        return []
    alerts = []
    for c in [*NUMERIC, *CATEGORICAL]:
        ref = present_share(reference, c)
        if ref >= min_present and present_share(day, c) < drop * ref:
            alerts.append(c)
    return alerts


def cohort_performance(
    decisions: pd.DataFrame, labels: pd.DataFrame, as_of: int, costs: CostModel
) -> pd.DataFrame:
    """Weekly cohorts of transactions: labels arrived by ``as_of`` and their metrics."""
    from sklearn.metrics import average_precision_score

    arrived = labels[labels["released_at"] < as_of][["TransactionID", "isFraud"]]
    d = decisions.merge(arrived, on="TransactionID", how="left")
    rows = []
    for week, g in d.groupby("cohort", sort=True):
        known = g[g["isFraud"].notna()]
        row = {
            "cohort": week,
            "transactions": len(g),
            "labelled": len(known),
            "complete": len(known) == len(g),
        }
        if len(known) and known["isFraud"].sum() > 0:
            y = known["isFraud"].to_numpy().astype(int)
            actions = known["action_code"].to_numpy()
            r = costs.realised(actions, y, known["TransactionAmt"].to_numpy())
            row |= {
                "pr_auc": float(average_precision_score(y, known["p_fraud"])),
                "value_caught": r["fraud_value_caught_share"],
                "fraud_rate": float(y.mean()),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def timeline(
    decisions: pd.DataFrame,
    labels: pd.DataFrame,
    reference: pd.DataFrame,
    ref_perf: dict,
    start: date,
    tx_end: date,
    label_end: date,
    th: Thresholds,
    costs: CostModel,
    s: Settings,
) -> dict:
    daily = []
    snapshots = {}
    ref_alert_rate = float((reference["action"] != "approve").mean())
    score_alert_run = 0
    retrain_on = None
    retrain_reason = None
    first_feed_alert = None
    d = start + timedelta(days=th.window_days)
    while d <= label_end:
        t_sec = s.to_seconds(d)
        row = {"date": d.isoformat()}
        if d <= tx_end + timedelta(days=1):
            win = decisions[
                (decisions["TransactionDT"] >= t_sec - th.window_days * 86_400)
                & (decisions["TransactionDT"] < t_sec)
            ]
            drift = input_drift(reference, win)
            snapshots[d.isoformat()] = drift.pop("snapshot")
            row |= {
                "transactions": len(win),
                "drifted_share": drift["drifted_share"],
                "drifted_columns": sorted(c for c, v in drift["columns"].items() if v["drifted"]),
                "score_psi": psi(reference["p_fraud"].to_numpy(), win["p_fraud"].to_numpy()),
                "alert_rate": float((win["action"] != "approve").mean()),
            }
            row["input_drift_warn"] = row["drifted_share"] >= th.drifted_share_warn
            row["score_status"] = (
                "alert"
                if row["score_psi"] >= th.psi_alert
                else "warn"
                if row["score_psi"] >= th.psi_warn
                else "ok"
            )
            row["alert_rate_warn"] = (
                abs(row["alert_rate"] / ref_alert_rate - 1) >= th.alert_rate_change_warn
            )
            day = decisions[decisions["TransactionDT"] >= t_sec - 86_400]
            day = day[day["TransactionDT"] < t_sec]
            row["feed_alerts"] = feed_health(reference, day, th.feed_min_present, th.feed_drop)
            if row["feed_alerts"] and first_feed_alert is None:
                first_feed_alert = {"date": d.isoformat(), "columns": row["feed_alerts"]}
            score_alert_run = score_alert_run + 1 if row["score_status"] == "alert" else 0
            if (
                retrain_on is None
                and score_alert_run >= th.consecutive_score_alert_days
                and row["input_drift_warn"]
            ):
                retrain_on, retrain_reason = d.isoformat(), "score drift with input drift"
        perf = cohort_performance(decisions, labels, t_sec, costs)
        complete = perf[perf["complete"] & perf.get("pr_auc", pd.Series(dtype=float)).notna()]
        row["complete_cohorts"] = len(complete)
        for _, c in complete.iterrows():
            bad = (
                c["pr_auc"] < th.pr_auc_ratio_alert * ref_perf["pr_auc"]
                or c["value_caught"] <= ref_perf["value_caught"] - th.value_caught_drop_alert
            )
            if bad and retrain_on is None:
                retrain_on, retrain_reason = d.isoformat(), f"performance alert on {c['cohort']}"
        daily.append(row)
        d += timedelta(days=1)
    final = cohort_performance(
        decisions, labels, s.to_seconds(label_end + timedelta(days=1)), costs
    )
    final["pr_auc_alert"] = final["pr_auc"] < th.pr_auc_ratio_alert * ref_perf["pr_auc"]
    final["value_caught_alert"] = (
        final["value_caught"] <= ref_perf["value_caught"] - th.value_caught_drop_alert
    )
    complete_on = {
        c: (date.fromisoformat(c) + timedelta(days=6 + 30 + 1)).isoformat() for c in final["cohort"]
    }
    final["complete_on"] = final["cohort"].map(complete_on)
    return {
        "daily": daily,
        "cohorts": final.to_dict("records"),
        "retrain_on": retrain_on,
        "retrain_reason": retrain_reason,
        "reference_alert_rate": ref_alert_rate,
        "first_feed_alert": first_feed_alert,
        "snapshots": snapshots,
    }


def with_cohorts(frame: pd.DataFrame, s: Settings, start: date) -> pd.DataFrame:
    days = (frame["TransactionDT"] - s.to_seconds(start)) // 86_400
    week_start = [start + timedelta(days=int(w) * 7) for w in (days // 7)]
    codes = {v: k for k, v in ACTION_NAMES.items()}
    return frame.assign(
        cohort=[w.isoformat() for w in week_start], action_code=frame["action"].map(codes)
    )
