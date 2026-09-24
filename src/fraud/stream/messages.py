"""Message formats on the three topics, and reading the replay period from the lake.

- ``transactions``: one raw transaction as the payment gateway would send it (every
  field of silver except the label), JSON, keyed by ``TransactionID``. Missing values are
  left out to keep messages small.
- ``labels``: ``{TransactionID, TransactionDT, isFraud, released_at}``, published when the
  simulated clock reaches ``TransactionDT + label delay``.
- ``decisions``: the processor's output: score, action, expected costs, reasons, the
  online feature values and timings.
"""

from __future__ import annotations

import json
import math

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from fraud.config import Settings
from fraud.lakehouse.schema import IDENTITY_COLUMNS, LABEL, TRANSACTION_COLUMNS
from fraud.lakehouse.tables import SILVER_TRANSACTIONS, table_path

EVENT_COLUMNS = [
    *(c for c in TRANSACTION_COLUMNS if c != LABEL),
    *(c for c in IDENTITY_COLUMNS if c != "TransactionID"),
    "has_identity",
]


def _clean(v):
    if hasattr(v, "item") and not isinstance(v, str):  # numpy scalar
        v = v.item()
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, dict):
        return {k: c for k, x in v.items() if (c := _clean(x)) is not None}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    return v


def encode(record: dict) -> bytes:
    return json.dumps(_clean(record), separators=(",", ":")).encode()


def decode(value: bytes) -> dict:
    return json.loads(value)


def silver_between(
    spark: SparkSession, s: Settings, start: int, end: int, with_label: bool = True
) -> pd.DataFrame:
    """Silver rows with ``start <= TransactionDT < end``, in arrival order."""
    cols = [*EVENT_COLUMNS, LABEL] if with_label else EVENT_COLUMNS
    df = (
        spark.read.format("delta")
        .load(table_path(s, SILVER_TRANSACTIONS))
        .filter((F.col("TransactionDT") >= start) & (F.col("TransactionDT") < end))
        .select(*cols)
    )
    pdf = df.toPandas()
    return pdf.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def records(frame: pd.DataFrame, columns: list[str] | None = None, chunk: int = 5_000):
    """Rows as event dicts (missing values dropped), generated a chunk at a time.

    A month of transactions with every column is several GB as Python dicts, so rows are
    never all materialised at once.
    """
    cols = columns or [c for c in EVENT_COLUMNS if c in frame.columns]
    for start in range(0, len(frame), chunk):
        for r in frame[cols].iloc[start : start + chunk].to_dict("records"):
            yield {k: c for k, v in r.items() if (c := _clean(v)) is not None}
