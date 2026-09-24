"""Silver: typed, deduplicated, joined, on a calendar, and checked.

Steps, each of which can fail the job:

1. cast every column to its type with ``try_cast`` and count values that did not parse;
2. drop exact duplicate rows, then require ``TransactionID`` to be unique in each table;
3. require every identity row to match a transaction, then left-join identity on;
4. apply the time anchor (``event_ts``, ``event_date``, ``event_month``);
5. row checks on keys, nulls, allowed values and ranges.
"""

from __future__ import annotations

import logging
from datetime import UTC

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from fraud.config import Settings
from fraud.lakehouse.quality import CheckResult, RowCheck, enforce, run_row_checks, unique_check
from fraud.lakehouse.schema import (
    ALLOWED,
    AMOUNT,
    IDENTITY_COLUMNS,
    KEY,
    LABEL,
    TIME,
    TRANSACTION_COLUMNS,
    spark_type,
)
from fraud.lakehouse.tables import (
    BRONZE_IDENTITY,
    BRONZE_TRANSACTION,
    QUALITY_DIR,
    SILVER_TRANSACTIONS,
    lake_dir,
    table_path,
)

log = logging.getLogger(__name__)


def _typed(df: DataFrame, columns: list[str], table: str) -> tuple[DataFrame, CheckResult]:
    """Cast each column; count raw values that were present but did not parse."""
    casts = {c: F.col(c).try_cast(spark_type(c)) for c in columns}
    failed = [
        F.sum(
            F.when(F.col(c).isNotNull() & (F.trim(F.col(c)) != "") & casts[c].isNull(), 1)
            .otherwise(0)
            .cast("long")
        ).alias(c)
        for c in columns
        if spark_type(c) != "string"
    ]
    row = df.agg(*failed).collect()[0].asDict() if failed else {}
    bad = {c: int(v or 0) for c, v in row.items() if v}
    result = CheckResult(
        f"{table}_casts",
        sum(bad.values()),
        "every value parses as its type"
        + (f"; failures by column: {dict(list(bad.items())[:10])}" if bad else ""),
    )
    return df.select(*(casts[c].alias(c) for c in columns)), result


def _dedupe(df: DataFrame, table: str) -> tuple[DataFrame, CheckResult]:
    before = df.count()
    df = df.dropDuplicates()
    dropped = before - df.count()
    if dropped:
        log.warning("%s: dropped %s exact duplicate rows", table, dropped)
    # Exact duplicates are harmless and removed; this check records how many there were.
    return df, CheckResult(f"{table}_exact_duplicates_dropped", 0, f"{dropped} rows dropped")


def anchor_epoch_seconds(s: Settings) -> int:
    return int(s.anchor.replace(tzinfo=UTC).timestamp())


def with_calendar(df: DataFrame, s: Settings) -> DataFrame:
    ts = F.timestamp_seconds(F.lit(anchor_epoch_seconds(s)) + F.col(TIME))
    return (
        df.withColumn("event_ts", ts)
        .withColumn("event_date", F.to_date("event_ts"))
        .withColumn("event_month", F.date_format("event_ts", "yyyy-MM"))
    )


def row_checks(s: Settings) -> list[RowCheck]:
    start = s.split.train.start.isoformat()
    end = s.split.test.end.isoformat()

    def allowed(c: str) -> RowCheck:
        return RowCheck(
            f"{c}_allowed",
            F.col(c).isNotNull() & ~F.col(c).isin(*sorted(ALLOWED[c])),
            f"{c} is null or one of {sorted(ALLOWED[c])}",
        )

    return [
        RowCheck(f"{KEY}_not_null", F.col(KEY).isNull(), "every row has a TransactionID"),
        RowCheck(
            f"{LABEL}_binary",
            F.col(LABEL).isNull() | ~F.col(LABEL).isin(0, 1),
            "isFraud is 0 or 1",
        ),
        RowCheck(f"{TIME}_valid", F.col(TIME).isNull() | (F.col(TIME) < 0), "TransactionDT >= 0"),
        RowCheck(
            f"{AMOUNT}_positive",
            F.col(AMOUNT).isNull() | (F.col(AMOUNT) <= 0),
            "TransactionAmt is present and > 0",
        ),
        RowCheck("card1_not_null", F.col("card1").isNull(), "card1 is present (card key)"),
        RowCheck("ProductCD_not_null", F.col("ProductCD").isNull(), "every row has a product code"),
        *(allowed(c) for c in ("ProductCD", "card4", "card6", "DeviceType")),
        RowCheck(
            "event_ts_in_range",
            (F.col("event_ts") < F.lit(start).cast("timestamp"))
            | (F.col("event_ts") >= F.lit(end).cast("timestamp")),
            f"every event falls in [{start}, {end})",
        ),
    ]


def build_silver(spark: SparkSession, s: Settings) -> int:
    tx_raw = spark.read.format("delta").load(table_path(s, BRONZE_TRANSACTION))
    id_raw = spark.read.format("delta").load(table_path(s, BRONZE_IDENTITY))

    results: list[CheckResult] = []
    tx, r = _typed(tx_raw, TRANSACTION_COLUMNS, "transaction")
    results.append(r)
    ident, r = _typed(id_raw, IDENTITY_COLUMNS, "identity")
    results.append(r)

    tx, r = _dedupe(tx, "transaction")
    results.append(r)
    ident, r = _dedupe(ident, "identity")
    results.append(r)
    results.append(unique_check(tx, KEY))
    id_unique = unique_check(ident, KEY)
    results.append(
        CheckResult(f"identity_{id_unique.name}", id_unique.failed_rows, "one row per transaction")
    )

    orphans = ident.join(tx.select(KEY), KEY, "left_anti").count()
    results.append(
        CheckResult("identity_orphans", orphans, "every identity row matches a transaction")
    )

    joined = tx.join(ident.withColumn("has_identity", F.lit(True)), KEY, "left").withColumn(
        "has_identity", F.coalesce(F.col("has_identity"), F.lit(False))
    )
    silver = with_calendar(joined, s)
    results += run_row_checks(silver, row_checks(s))
    enforce(results, "silver", lake_dir(s).joinpath(*QUALITY_DIR))

    path = table_path(s, SILVER_TRANSACTIONS)
    silver.orderBy(TIME, KEY).write.format("delta").mode("overwrite").option(
        "overwriteSchema", True
    ).save(path)
    n = spark.read.format("delta").load(path).count()
    log.info("silver transactions: %s rows", f"{n:,}")
    return n
