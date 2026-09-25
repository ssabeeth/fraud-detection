"""Online/offline parity: replay transactions through ``OnlineFeatures`` and compare.

Every transaction is replayed in event-time order from the first day, so the online state
is warm, and every feature of every row in the chosen splits is compared with the Spark
value. Each label is released ``LABEL_DELAY`` after its transaction, in the same timeline
(a transaction goes before a label released in the same second, as in the stream).
``fraud check-parity`` runs it on the real data in-process; phase 7 runs the same
comparison on what the stream processor actually published.
"""

from __future__ import annotations

import heapq
import json
import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from fraud.config import Settings
from fraud.features.definitions import AGGREGATE_NAMES, LABEL_DELAY
from fraud.features.online import OnlineFeatures
from fraud.features.reference import _close
from fraud.lakehouse.tables import GOLD_FEATURES, table_path

log = logging.getLogger(__name__)

# The raw fields the online path needs to compute keys and features.
EVENT_FIELDS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "D1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
    "has_identity",
]
COMPARED = [
    *AGGREGATE_NAMES,
    "card_key",
    "device_key",
    "email_key",
    "amt_cents",
    "hour",
    "weekday",
    "email_match",
]


LABEL_FIELDS = ["TransactionID", "TransactionDT", "isFraud", "card1", "addr1", "D1"]
TX, LABEL = 0, 1


def offline_frame(spark: SparkSession, s: Settings) -> pd.DataFrame:
    cols = list(dict.fromkeys([*EVENT_FIELDS, "isFraud", "split", *COMPARED]))
    pdf = spark.read.format("delta").load(table_path(s, GOLD_FEATURES)).select(*cols).toPandas()
    return pdf.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def _clean(rec: dict) -> dict:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()}


def events(frame: pd.DataFrame) -> Iterator[dict]:
    for rec in frame[EVENT_FIELDS].to_dict("records"):
        yield _clean(rec)


def with_labels(frame: pd.DataFrame, delay: int = LABEL_DELAY) -> Iterator[tuple[int, dict]]:
    """Transactions and their labels as one timeline: (kind, payload). A label is released
    ``delay`` after its transaction; on a tie the transaction goes first."""
    txs = ((int(e["TransactionDT"]), TX, i, e) for i, e in enumerate(events(frame)))
    labels = frame[LABEL_FIELDS].assign(released=frame["TransactionDT"] + delay)
    labels = labels.sort_values(["released", "TransactionID"], kind="stable")
    released = labels["released"].to_numpy()
    lab = (
        (int(released[i]), LABEL, i, _clean(r))
        for i, r in enumerate(labels[LABEL_FIELDS].to_dict("records"))
    )
    for _, kind, _, payload in heapq.merge(txs, lab, key=lambda x: (x[0], x[1], x[2])):
        yield kind, payload


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return _close(a, b, rel_tol=1e-9)


def compare(offline: pd.DataFrame, online: Iterable[dict], splits: set[str]) -> dict:
    """Compare online outputs (in the same row order as ``offline``) for rows in splits."""
    mismatches: dict[str, int] = {}
    examples = []
    checked = 0
    for off, on in zip(offline.to_dict("records"), online, strict=True):
        if off["split"] not in splits:
            continue
        checked += 1
        for name in COMPARED:
            if not _same(off[name], on[name]):
                mismatches[name] = mismatches.get(name, 0) + 1
                if len(examples) < 10:
                    examples.append((off["TransactionID"], name, off[name], on[name]))
    return {
        "rows_compared": checked,
        "features_compared": len(COMPARED),
        "values_compared": checked * len(COMPARED),
        "mismatches": sum(mismatches.values()),
        "mismatches_by_feature": mismatches,
        "examples": examples,
    }


def check(spark: SparkSession, s: Settings, splits: set[str], report: Path | None = None) -> dict:
    offline = offline_frame(spark, s)
    online = OnlineFeatures()
    start = time.perf_counter()
    outputs = []
    for kind, payload in with_labels(offline):
        if kind == TX:
            outputs.append(online.process(payload))
        else:
            online.observe_label(payload)
    elapsed = time.perf_counter() - start
    result = compare(offline, outputs, splits)
    result |= {
        "splits": sorted(splits),
        "events_replayed": len(outputs),
        "online_seconds": round(elapsed, 1),
        "online_events_per_second": round(len(outputs) / elapsed),
        "online_keys": online.state_size(),
    }
    log.info("parity: %s", {k: v for k, v in result.items() if k != "examples"})
    if report is not None:
        # Examples name transactions, so they go to the log only, never to the report.
        public = {k: v for k, v in result.items() if k != "examples"}
        report.write_text(json.dumps(public, indent=2, default=str) + "\n")
    return result
