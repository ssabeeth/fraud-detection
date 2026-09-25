"""Load modelling frames from the gold feature table, and guard the test month.

Every read of the test split must give a purpose, and is appended to
``reports/test_touches.jsonl``. The brief allows the test month to be touched once per
reported result; the log makes that auditable.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from fraud.config import Settings, reports_dir
from fraud.lakehouse.tables import GOLD_FEATURES, table_path
from fraud.model.inputs import raw_columns

log = logging.getLogger(__name__)


def test_log_path() -> Path:
    return reports_dir() / "test_touches.jsonl"


def record_test_touch(purpose: str, log_path: Path | None = None) -> None:
    log_path = log_path or test_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if note := os.environ.get("FRAUD_TEST_PURPOSE_NOTE"):
        purpose = f"{purpose} ({note})"
    entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "purpose": purpose}
    # Rewritten whole rather than appended to: Databricks volumes do not support appends.
    before = log_path.read_text() if log_path.exists() else ""
    log_path.write_text(before + json.dumps(entry) + "\n")
    log.warning("test month read: %s", purpose)


def list_test_touches(log_path: Path | None = None) -> list[dict]:
    log_path = log_path or test_log_path()
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def load_frame(
    spark: SparkSession,
    s: Settings,
    splits: list[str],
    *,
    feature_set: str = "all",
    test_purpose: str | None = None,
    extra: list[str] | None = None,
) -> pd.DataFrame:
    """Rows of the given splits, in time order, with the columns a feature set needs."""
    if "test" in splits:
        if not test_purpose:
            raise PermissionError("reading the test month needs a stated purpose")
        record_test_touch(test_purpose)
    cols = list(dict.fromkeys([*raw_columns(feature_set), *(extra or [])]))
    df = (
        spark.read.format("delta")
        .load(table_path(s, GOLD_FEATURES))
        .filter(F.col("split").isin(splits))
        .select(*cols)
    )
    pdf = df.toPandas()
    pdf = pdf.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    log.info("loaded %s rows (%s) with %d columns", f"{len(pdf):,}", "+".join(splits), len(cols))
    return pdf
