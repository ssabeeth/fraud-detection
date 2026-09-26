"""Orchestrate ``fraud monitor``: reference, the replayed stream, the stress scenario."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from fraud.config import Settings, reports_dir
from fraud.model.bundle import ModelBundle
from fraud.model.data import load_frame
from fraud.monitor.monitor import (
    IDENTITY_FIELDS,
    NUMERIC,
    Thresholds,
    _prepare,
    timeline,
    with_cohorts,
)
from fraud.policy.costs import ACTION_NAMES, CostModel
from fraud.policy.frozen import FrozenPolicy
from fraud.policy.policy import ExpectedLossPolicy
from fraud.stream.scorer import OnlineScorer

log = logging.getLogger(__name__)

RAW = ["ProductCD", "card4", "card6", "DeviceType", "has_identity", "P_emaildomain"]


def _bronze_json(spark, s: Settings, role: str) -> list[dict]:
    from fraud.stream.sink import stream_table

    df = spark.read.format("delta").load(stream_table(s, role)).select("value")
    return [json.loads(r["value"]) for r in df.toLocalIterator()]


def stream_log(spark, s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.DataFrame(
        [
            {
                "TransactionID": d["TransactionID"],
                "TransactionDT": d["TransactionDT"],
                "TransactionAmt": d["TransactionAmt"],
                "p_fraud": d["p_fraud"],
                "action": d["action"],
                **{k: d["features"].get(k) for k in NUMERIC if k in d["features"]},
            }
            for d in _bronze_json(spark, s, "decisions")
        ]
    )
    raw = pd.DataFrame(
        [
            {"TransactionID": t["TransactionID"], **{k: t.get(k) for k in RAW}}
            for t in _bronze_json(spark, s, "transactions")
        ]
    )
    labels = pd.DataFrame(_bronze_json(spark, s, "labels"))
    return decisions.merge(raw, on="TransactionID", how="left"), labels


def reference_frame(spark, s: Settings, bundle: ModelBundle, frozen: FrozenPolicy):
    valid = load_frame(spark, s, ["valid"], extra=RAW)
    valid["p_fraud"] = bundle.predict(valid)
    actions = ExpectedLossPolicy("p_fraud", frozen.review_threshold).decide(valid, frozen.costs)
    valid["action"] = [ACTION_NAMES[a] for a in actions]
    r = frozen.costs.realised(
        actions, valid["isFraud"].to_numpy(), valid["TransactionAmt"].to_numpy()
    )
    from sklearn.metrics import average_precision_score

    perf = {
        "pr_auc": float(average_precision_score(valid["isFraud"], valid["p_fraud"])),
        "value_caught": r["fraud_value_caught_share"],
    }
    return valid, perf


def outage_scenario(
    spark,
    s: Settings,
    bundle: ModelBundle,
    frozen: FrozenPolicy,
    start: date,
    end: date,
    outage_from: date,
):
    """Replay the window in-process with identity fields blanked from ``outage_from``.

    Labels arrive as in the stream: each ``LABEL_DELAY`` after its transaction, applied
    before every later transaction (only frauds change the feature state)."""
    from fraud.features.definitions import LABEL_DELAY
    from fraud.features.parity import EVENT_FIELDS
    from fraud.stream.messages import records, silver_between

    lo, hi, cut = s.to_seconds(start), s.to_seconds(end), s.to_seconds(outage_from)
    history = silver_between(spark, s, 0, lo)[[*EVENT_FIELDS, "isFraud"]]
    window = silver_between(spark, s, lo, hi)
    scorer = OnlineScorer(bundle, frozen)
    scorer.warm(records(history, [*EVENT_FIELDS, "isFraud"]), labels_before=lo)
    due = pd.concat([history, window])
    due = due[(due["isFraud"] == 1) & (due["TransactionDT"] + LABEL_DELAY >= lo)]
    due = due.assign(released=due["TransactionDT"] + LABEL_DELAY).sort_values(
        ["released", "TransactionID"], kind="stable"
    )
    pending = list(
        zip(
            due["released"],
            records(due, ["TransactionDT", "isFraud", "card1", "addr1", "D1"]),
            strict=True,
        )
    )
    applied = 0
    rows = []
    for event, fraud in zip(records(window), window["isFraud"], strict=True):
        while applied < len(pending) and pending[applied][0] < event["TransactionDT"]:
            scorer.observe_label(pending[applied][1])
            applied += 1
        if event["TransactionDT"] >= cut:
            event = {**event, **dict.fromkeys(IDENTITY_FIELDS), "has_identity": False}
        d = scorer.decide(event)
        rows.append(
            {
                "TransactionID": d["TransactionID"],
                "TransactionDT": d["TransactionDT"],
                "TransactionAmt": d["TransactionAmt"],
                "p_fraud": d["p_fraud"],
                "action": d["action"],
                **{k: d["features"].get(k) for k in NUMERIC[2:]},
                **{k: event.get(k) for k in RAW},
                "_fraud": int(fraud),
            }
        )
    decisions = pd.DataFrame(rows)
    labels = pd.DataFrame(
        {
            "TransactionID": decisions["TransactionID"],
            "isFraud": decisions.pop("_fraud"),
            "released_at": decisions["TransactionDT"] + s.label_delay_seconds,
        }
    )
    return decisions, labels


def _before_outage(scenarios: dict, cut: int) -> dict:
    """Up to the outage the in-process replay must decide exactly as the stream did."""
    stream = scenarios["replay"][0].set_index("TransactionID")
    stress = scenarios["identity_outage"][0].set_index("TransactionID")
    ids = stress.index[stress["TransactionDT"] < cut]
    a, b = stream.loc[ids], stress.loc[ids]
    out = {
        "transactions": len(ids),
        "actions_equal": int((a["action"] == b["action"]).sum()),
        "max_p_fraud_difference": float((a["p_fraud"] - b["p_fraud"]).abs().max()),
    }
    if out["actions_equal"] != out["transactions"] or out["max_p_fraud_difference"] > 1e-9:
        raise RuntimeError(f"stress replay differs from the stream before the outage: {out}")
    return out


def run(
    spark, s: Settings, out_dir: Path | None = None, outage_from: date = date(2018, 5, 22)
) -> dict:
    out_dir = out_dir or reports_dir()
    th = Thresholds.load()
    frozen = FrozenPolicy.load()
    costs: CostModel = frozen.costs
    bundle = ModelBundle.load(s.path("models") / "lightgbm")
    reference, ref_perf = reference_frame(spark, s, bundle, frozen)
    domains = reference["P_emaildomain"].value_counts().head(8).index.tolist()
    ref = _prepare(reference, domains)

    start, tx_end = s.split.test.start, s.split.test.end - timedelta(days=1)
    label_end = tx_end + timedelta(days=s.label_delay_days)
    results = {
        "reference": ref_perf | {"alert_rate": float((reference["action"] != "approve").mean())},
        "thresholds": th.__dict__,
        "outage_from": outage_from.isoformat(),
    }
    scenarios = {
        "replay": stream_log(spark, s),
        "identity_outage": outage_scenario(
            spark, s, bundle, frozen, start, s.split.test.end, outage_from
        ),
    }
    results["before_outage"] = _before_outage(scenarios, s.to_seconds(outage_from))
    evidently_dir = s.path("monitoring")
    evidently_dir.mkdir(parents=True, exist_ok=True)
    for name, (decisions, labels) in scenarios.items():
        dec = with_cohorts(_prepare(decisions, domains), s, start)
        tl = timeline(dec, labels, ref, ref_perf, start, tx_end, label_end, th, costs, s)
        snaps = tl.pop("snapshots")
        last = max(snaps)
        snaps[last].save_html(str(evidently_dir / f"evidently_{name}_{last}.html"))
        last_day = dec.groupby("cohort")["TransactionDT"].max()
        for c in tl["cohorts"]:
            c["complete_on"] = (
                s.to_datetime(int(last_day[c["cohort"]]) + s.label_delay_seconds).date()
                + timedelta(days=1)
            ).isoformat()
        results[name] = tl
        log.info("%s: retrain on %s (%s)", name, tl["retrain_on"], tl["retrain_reason"])
    (out_dir / "monitoring.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    from fraud.monitor.report import write_monitoring_report

    write_monitoring_report(results, out_dir)
    return results
