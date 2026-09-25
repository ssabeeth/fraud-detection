"""The point-in-time check: recompute features for sampled rows from raw history.

It reads *silver* (raw, typed history), derives the entity keys with the plain-Python
key functions, recomputes every aggregate with the naive reference implementation, and
compares with the Spark feature table. Any difference fails. Run it on the real data
with ``fraud check-pit``; the test suite runs it on every fixture row.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from fraud.config import Settings
from fraud.features.definitions import AGGREGATE_NAMES
from fraud.features.keys import card_key, device_key, email_key
from fraud.features.reference import differences, recompute
from fraud.lakehouse.tables import GOLD_FEATURES, SILVER_TRANSACTIONS, table_path

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
    "card1",
    "addr1",
    "D1",
    "P_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]


def raw_history(spark: SparkSession, s: Settings) -> pd.DataFrame:
    raw = spark.read.format("delta").load(table_path(s, SILVER_TRANSACTIONS))
    pdf = raw.select(*RAW_COLUMNS).toPandas()
    records = pdf.to_dict("records")
    pdf["card_key"] = [
        card_key(r["card1"], r["addr1"], r["TransactionDT"], r["D1"]) for r in records
    ]
    pdf["device_key"] = [device_key(r) for r in records]
    pdf["email_key"] = [email_key(r) for r in records]
    return pdf


def sample_rows(raw: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """A random sample, topped up with fraud rows and rows with long card histories."""
    if n >= len(raw):
        return raw
    rng_rows = raw.sample(n=n, random_state=seed)
    fraud = raw[raw["isFraud"] == 1]
    fraud = fraud.sample(n=min(n // 4, len(fraud)), random_state=seed)
    busy = raw[raw["card_key"].isin(raw["card_key"].value_counts().head(20).index)]
    busy = busy.sample(n=min(n // 4, len(busy)), random_state=seed)
    return pd.concat([rng_rows, fraud, busy]).drop_duplicates("TransactionID")


def check(
    spark: SparkSession,
    s: Settings,
    *,
    sample: int | None = None,
    features: DataFrame | None = None,
    report: Path | None = None,
) -> list[tuple]:
    """Return the differences (empty when the features are point-in-time correct)."""
    t0 = time.perf_counter()
    raw = raw_history(spark, s)
    rows = raw if sample is None else sample_rows(raw, sample)
    log.info("history: %d rows, %d sampled (%.1fs)", len(raw), len(rows), time.perf_counter() - t0)
    t0 = time.perf_counter()
    expected = recompute(raw, rows)
    log.info("recomputed from history (%.1fs)", time.perf_counter() - t0)
    t0 = time.perf_counter()
    if features is None:
        features = spark.read.format("delta").load(table_path(s, GOLD_FEATURES))
    ids = spark.createDataFrame(rows[["TransactionID"]])
    actual = features.join(ids, "TransactionID").select("TransactionID", *AGGREGATE_NAMES)
    actual = actual.toPandas()
    log.info("feature table rows fetched (%.1fs)", time.perf_counter() - t0)
    diffs = differences(expected, actual)
    log.info(
        "point-in-time check: %d rows x %d features, %d differences",
        len(rows),
        len(AGGREGATE_NAMES),
        len(diffs),
    )
    if report is not None:
        by_feature: dict[str, int] = {}
        for _, name, _, _ in diffs:
            by_feature[name] = by_feature.get(name, 0) + 1
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "rows_checked": len(rows),
                    "fraud_rows_checked": int(rows["isFraud"].sum()),
                    "features_checked": len(AGGREGATE_NAMES),
                    "values_compared": len(rows) * len(AGGREGATE_NAMES),
                    "differences": len(diffs),
                    "differences_by_feature": by_feature,
                },
                indent=2,
            )
            + "\n"
        )
    return diffs
