"""``fraud stream``: the whole online path end to end on a replayed period.

1. Read the replay window (default: the test month) and the labels that fall due in it.
2. Recreate the three topics (one partition each).
3. Start the processor in its own process; it warms its feature state with every earlier
   transaction from the lake, then consumes.
4. Replay transactions and delayed labels into Redpanda at the chosen speed-up.
5. Land all three topics in Delta bronze with Spark Structured Streaming.
6. The parity test through the stream: every feature the processor published, read back
   from the Delta bronze table, against the offline Spark feature table; and every action
   against the frozen policy applied offline to the same rows.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from fraud.config import Settings, reports_dir
from fraud.features.definitions import AGGREGATE_NAMES
from fraud.features.parity import EVENT_FIELDS, _same
from fraud.lakehouse.tables import GOLD_FEATURES, SILVER_TRANSACTIONS, table_path
from fraud.stream.messages import silver_between
from fraud.stream.producer import LABEL_FIELDS

log = logging.getLogger(__name__)

KEYS = ["card_key", "device_key", "email_key"]


def _kafka_conf(s: Settings) -> dict:
    return {"bootstrap.servers": s.kafka.bootstrap_servers}


def recreate_topics(s: Settings) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient(_kafka_conf(s))
    names = list(s.kafka.topics.model_dump().values())
    existing = set(admin.list_topics(timeout=10).topics)
    to_delete = [n for n in names if n in existing]
    if to_delete:
        for f in admin.delete_topics(to_delete).values():
            f.result()
        time.sleep(2)
    for f in admin.create_topics([NewTopic(n, 1, 1) for n in names]).values():
        f.result()


def _processor_main(
    s: Settings,
    models_dir: str,
    warm_path: str,
    stats_path: str,
    ready,
    idle_seconds: float,
    labels_before: int,
) -> None:
    from confluent_kafka import Consumer, Producer

    from fraud.model.bundle import ModelBundle
    from fraud.policy.frozen import FrozenPolicy
    from fraud.stream.processor import run
    from fraud.stream.scorer import OnlineScorer

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    scorer = OnlineScorer(ModelBundle.load(Path(models_dir) / "lightgbm"), FrozenPolicy.load())
    warm = pd.read_parquet(warm_path)
    t0 = time.perf_counter()
    n = scorer.warm(
        (
            {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
            for r in warm.to_dict("records")
        ),
        labels_before=labels_before,
    )
    log.info(
        "warmed feature state with %s transactions in %.1f s", f"{n:,}", time.perf_counter() - t0
    )
    ready.set()
    consumer = Consumer(
        _kafka_conf(s)
        | {
            "group.id": f"processor-{time.time_ns()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    producer = Producer(_kafka_conf(s) | {"linger.ms": 5})
    stats = run(consumer, producer, scorer, s.kafka.topics.model_dump(), idle_seconds=idle_seconds)
    stats["warm_transactions"] = n
    Path(stats_path).write_text(json.dumps(stats, indent=2))


def replay_and_process(
    spark,
    s: Settings,
    start: date,
    end: date,
    speedup: float,
    idle_seconds: float = 15.0,
    label_tail_days: int = 30,
) -> dict:
    from confluent_kafka import Producer

    from fraud.stream.producer import replay, timeline

    lo, hi = s.to_seconds(start), s.to_seconds(end)
    delay = s.label_delay_seconds
    window = silver_between(spark, s, lo, hi)
    # their labels are released during the window; older ones are applied at warm-up
    earlier = silver_between(spark, s, lo - delay, lo)[LABEL_FIELDS]
    work = s.path("stream")
    work.mkdir(parents=True, exist_ok=True)
    warm_path, stats_path = work / "warm.parquet", work / "processor_stats.json"
    (
        spark.read.format("delta")
        .load(table_path(s, SILVER_TRANSACTIONS))
        .filter(F.col("TransactionDT") < lo)
        .select(*EVENT_FIELDS, "isFraud")
        .orderBy("TransactionDT", "TransactionID")
        .toPandas()
        .to_parquet(warm_path)
    )
    recreate_topics(s)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(
        target=_processor_main,
        args=(
            s,
            str(s.path("models")),
            str(warm_path),
            str(stats_path),
            ready,
            idle_seconds,
            lo,
        ),
    )
    proc.start()
    if not ready.wait(timeout=900):
        proc.terminate()
        raise TimeoutError("processor did not finish warming")
    producer = Producer(_kafka_conf(s) | {"linger.ms": 5, "queue.buffering.max.messages": 500_000})
    produced = replay(
        producer,
        s.kafka.topics.model_dump(),
        timeline(window, earlier, lo, hi, delay, labels_until=hi + label_tail_days * 86_400),
        speedup=speedup,
    )
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"processor exited with {proc.exitcode}")
    stats = json.loads(stats_path.read_text())
    warm_path.unlink()
    return {
        "window": [start.isoformat(), end.isoformat()],
        "label_tail_days": label_tail_days,
        "speedup": speedup,
        "produced": produced,
        "processor": stats,
        "window_frame": window,
    }


def decisions_from_bronze(spark, s: Settings) -> pd.DataFrame:
    from fraud.stream.sink import stream_table

    values = spark.read.format("delta").load(stream_table(s, "decisions")).select("value")
    rows = [json.loads(r["value"]) for r in values.toLocalIterator()]
    flat = []
    for d in rows:
        rec = {k: d.get(k) for k in ("TransactionID", "p_fraud", "action")}
        rec.update({k: d.get("features", {}).get(k) for k in [*AGGREGATE_NAMES, *KEYS]})
        rec["n_reasons"] = len(d.get("reasons", []))
        flat.append(rec)
    return pd.DataFrame(flat)


def stream_parity(spark, s: Settings, decisions: pd.DataFrame, window: pd.DataFrame) -> dict:
    """Features published by the stream vs the offline table, and actions vs offline policy."""
    from fraud.model.bundle import ModelBundle
    from fraud.policy.costs import ACTION_NAMES
    from fraud.policy.frozen import FrozenPolicy
    from fraud.policy.policy import ExpectedLossPolicy

    ids = spark.createDataFrame(decisions[["TransactionID"]])
    offline = (
        spark.read.format("delta")
        .load(table_path(s, GOLD_FEATURES))
        .join(ids, "TransactionID")
        .toPandas()
        .sort_values(["TransactionDT", "TransactionID"], kind="stable")
        .reset_index(drop=True)
    )
    merged = offline.merge(decisions, on="TransactionID", suffixes=("", "_stream"))
    mismatches: dict[str, int] = {}
    for name in [*AGGREGATE_NAMES, *KEYS]:
        bad = sum(
            not _same(a, b) for a, b in zip(merged[name], merged[f"{name}_stream"], strict=True)
        )
        if bad:
            mismatches[name] = bad

    bundle = ModelBundle.load(s.path("models") / "lightgbm")
    frozen = FrozenPolicy.load()
    offline["score_lightgbm"] = bundle.predict(offline)
    offline_actions = ExpectedLossPolicy("score_lightgbm", frozen.review_threshold).decide(
        offline, frozen.costs
    )
    offline["offline_action"] = [ACTION_NAMES[a] for a in offline_actions]
    both = offline[["TransactionID", "offline_action", "score_lightgbm"]].merge(
        decisions[["TransactionID", "action", "p_fraud"]], on="TransactionID"
    )
    return {
        "rows_compared": len(merged),
        "window_rows": len(window),
        "features_compared": len(AGGREGATE_NAMES) + len(KEYS),
        "feature_mismatches": sum(mismatches.values()),
        "feature_mismatches_by_name": mismatches,
        "actions_agree": int((both["action"] == both["offline_action"]).sum()),
        "actions_compared": len(both),
        "max_abs_score_difference": float(np.max(np.abs(both["p_fraud"] - both["score_lightgbm"]))),
        "decisions_without_reasons": int((decisions["n_reasons"] == 0).sum()),
        "actions": decisions["action"].value_counts().to_dict(),
    }


def run_all(
    s: Settings, start: date, end: date, speedup: float, report_dir: Path | None = None
) -> dict:
    report_dir = report_dir or reports_dir()
    from fraud.spark import get_spark
    from fraud.stream.report import write_stream_report
    from fraud.stream.sink import reset, run_sink

    spark = get_spark("stream", kafka=True)
    reset(s)
    result = replay_and_process(spark, s, start, end, speedup)
    window = result.pop("window_frame")
    t0 = time.perf_counter()
    result["bronze_rows"] = run_sink(spark, s)
    result["sink_seconds"] = round(time.perf_counter() - t0, 1)
    decisions = decisions_from_bronze(spark, s)
    result["parity"] = stream_parity(spark, s, decisions, window)
    report_dir.mkdir(parents=True, exist_ok=True)
    name = f"stream_results_speedup_{int(speedup)}.json"
    (report_dir / name).write_text(json.dumps(result, indent=2, default=str) + "\n")
    runs = [
        json.loads(p.read_text()) for p in sorted(report_dir.glob("stream_results_speedup_*.json"))
    ]
    write_stream_report(runs, report_dir)
    return result
