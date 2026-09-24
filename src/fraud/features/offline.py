"""Offline features: Spark window functions ordered by event time.

Every window has an exclusive upper bound: ``rangeBetween(-W, -1)`` on the integer
``TransactionDT`` keeps rows with ``t - W <= t_e <= t - 1``, which is exactly
``t - W <= t_e < t``. All-history aggregates use ``rangeBetween(unboundedPreceding, -1)``.
A range frame (not a row frame) is what makes same-second transactions invisible to
each other, whatever their order in the partition.
"""

from __future__ import annotations

import logging

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from fraud.config import Settings
from fraud.features.definitions import AGGREGATES, STD_EPSILON, Aggregate
from fraud.lakehouse.schema import AMOUNT, KEY, TIME
from fraud.lakehouse.tables import GOLD_FEATURES, GOLD_TRANSACTIONS, table_path

log = logging.getLogger(__name__)


def _partition(entity: str) -> Column:
    # Rows without a key get a partition of their own. Their features are null anyway,
    # and pooling them would build one huge window (three quarters of all rows have no
    # device) for results that are thrown away.
    return F.coalesce(F.col(entity), F.concat(F.lit("__none__"), F.col(KEY).cast("string")))


def _window(a: Aggregate, *, leaky: bool = False):
    upper = 0 if leaky else -1  # `leaky` exists only for the canary test
    lower = Window.unboundedPreceding if a.window is None else -a.window
    return Window.partitionBy(_partition(a.entity)).orderBy(TIME).rangeBetween(lower, upper)


def aggregate_col(a: Aggregate, *, leaky: bool = False) -> Column:
    w = _window(a, leaky=leaky)
    amt = F.col(AMOUNT)
    if a.kind == "count" and a.window is not None:
        # rows in [t - W, t) = rows before t minus rows before t - W. Both are running
        # counts, which Spark updates incrementally; a sliding count frame is recounted
        # for every row, which is quadratic on a hot key such as gmail.com.
        before_t = _window(Aggregate(a.name, a.entity, a.kind, None, ""), leaky=leaky)
        before_lo = (
            Window.partitionBy(_partition(a.entity))
            .orderBy(TIME)
            .rangeBetween(Window.unboundedPreceding, -a.window - 1)
        )
        expr = F.count(F.lit(1)).over(before_t) - F.count(F.lit(1)).over(before_lo)
    elif a.kind == "count":
        expr = F.count(F.lit(1)).over(w)
    elif a.kind == "sum_amount":
        expr = F.coalesce(F.sum(amt).over(w), F.lit(0.0))
    elif a.kind == "distinct":
        expr = F.size(F.collect_set(a.column).over(w))
    elif a.kind == "seconds_since_prev":
        expr = F.col(TIME) - F.max(TIME).over(w)
    elif a.kind == "amount_zscore":
        n, mean, std = F.count(F.lit(1)).over(w), F.avg(amt).over(w), F.stddev_samp(amt).over(w)
        expr = F.when((n >= 2) & (std > STD_EPSILON), (amt - mean) / std)
    elif a.kind == "amount_ratio":
        mean = F.avg(amt).over(w)
        expr = F.when(mean > 0, amt / mean)
    elif a.kind == "is_new":
        seen = F.collect_set(a.column).over(w)
        new = F.when(F.array_contains(seen, F.col(a.column)), 0).otherwise(1)
        expr = F.when(F.col(a.column).isNotNull(), new)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(a.kind)
    # A missing entity key means no history can be attributed: the feature is unknown.
    return F.when(F.col(a.entity).isNotNull(), expr)


def local_cols() -> dict[str, Column]:
    amt = F.col(AMOUNT)
    p, r = F.col("P_emaildomain"), F.col("R_emaildomain")
    return {
        "amt_cents": amt - F.floor(amt),
        "hour": (F.col(TIME) % 86_400 / 3_600).cast("int"),
        "weekday": (F.floor(F.col(TIME) / 86_400) % 7).cast("int"),
        "email_match": F.when(p.isNotNull() & r.isNotNull(), (p == r).cast("int")),
        "has_identity": F.col("has_identity").cast("int"),
    }


def add_features(df: DataFrame, *, leaky: bool = False) -> DataFrame:
    """Add every aggregate and local feature to a gold-shaped DataFrame."""
    return df.withColumns(
        {
            **{a.name: aggregate_col(a, leaky=leaky) for a in AGGREGATES},
            **local_cols(),
        }
    )


def build_features(spark: SparkSession, s: Settings) -> int:
    gold = spark.read.format("delta").load(table_path(s, GOLD_TRANSACTIONS))
    path = table_path(s, GOLD_FEATURES)
    add_features(gold).orderBy(TIME, KEY).write.format("delta").mode("overwrite").option(
        "overwriteSchema", True
    ).save(path)
    n = spark.read.format("delta").load(path).count()
    log.info("gold features: %s rows, %d aggregates", f"{n:,}", len(AGGREGATES))
    return n
