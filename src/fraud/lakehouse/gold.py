"""Gold: the modelling table.

Silver plus the split each row belongs to (boundaries from ``configs/base.yaml``) and
the entity keys the point-in-time features aggregate over. Phase 3 builds the feature
table on top of this one.
"""

from __future__ import annotations

import logging

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from fraud.config import Settings
from fraud.features.keys import card_key_col, device_key_col, email_key_col
from fraud.lakehouse.quality import RowCheck, enforce, run_row_checks
from fraud.lakehouse.schema import KEY, TIME
from fraud.lakehouse.tables import (
    GOLD_TRANSACTIONS,
    QUALITY_DIR,
    SILVER_TRANSACTIONS,
    lake_dir,
    table_path,
)

log = logging.getLogger(__name__)


def split_col(s: Settings) -> Column:
    expr = None
    for name, (lo, hi) in s.split_bounds_seconds().items():
        cond = (F.col(TIME) >= lo) & (F.col(TIME) < hi)
        expr = F.when(cond, F.lit(name)) if expr is None else expr.when(cond, F.lit(name))
    return expr


def with_keys(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("card_key", card_key_col())
        .withColumn("device_key", device_key_col())
        .withColumn("email_key", email_key_col())
    )


def build_gold(spark: SparkSession, s: Settings) -> dict[str, int]:
    silver = spark.read.format("delta").load(table_path(s, SILVER_TRANSACTIONS))
    gold = with_keys(silver.withColumn("split", split_col(s)))
    results = run_row_checks(
        gold,
        [
            RowCheck(
                "split_assigned", F.col("split").isNull(), "every row is in exactly one split"
            ),
            RowCheck("card_key_not_null", F.col("card_key").isNull(), "every row has a card key"),
        ],
    )
    enforce(results, "gold", lake_dir(s).joinpath(*QUALITY_DIR))
    path = table_path(s, GOLD_TRANSACTIONS)
    gold.orderBy(TIME, KEY).write.format("delta").mode("overwrite").option(
        "overwriteSchema", True
    ).save(path)
    counts = {
        r["split"]: r["count"]
        for r in spark.read.format("delta").load(path).groupBy("split").count().collect()
    }
    log.info("gold transactions by split: %s", counts)
    return counts
