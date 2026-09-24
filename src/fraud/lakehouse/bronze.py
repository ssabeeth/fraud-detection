"""Bronze: the Kaggle CSVs as received, every column a string, in Delta.

The CSVs are converted once and then deleted, so the lake holds the only copy of the
data. Bronze adds two lineage columns (source file and load time) and changes nothing
else; typing and checks happen in silver.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from fraud.config import Settings
from fraud.lakehouse.quality import DataQualityError
from fraud.lakehouse.schema import IDENTITY_COLUMNS, TRANSACTION_COLUMNS
from fraud.lakehouse.tables import BRONZE_IDENTITY, BRONZE_TRANSACTION, table_path

log = logging.getLogger(__name__)

SOURCES = {
    "train_transaction.csv": (BRONZE_TRANSACTION, TRANSACTION_COLUMNS),
    "train_identity.csv": (BRONZE_IDENTITY, IDENTITY_COLUMNS),
}


def build_bronze(
    spark: SparkSession, s: Settings, raw_dir: Path, *, delete_source: bool = False
) -> dict[str, int]:
    """Load each CSV into its bronze table. Returns row counts by table."""
    counts: dict[str, int] = {}
    for file_name, (table, expected) in SOURCES.items():
        src = raw_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"{src} not found (run `fraud download` first)")
        df = (
            spark.read.option("header", True)
            .option("inferSchema", False)
            .option("mode", "FAILFAST")
            .csv(str(src))
        )
        if df.columns != expected:
            missing = sorted(set(expected) - set(df.columns))
            extra = sorted(set(df.columns) - set(expected))
            raise DataQualityError(
                f"{file_name}: header differs from the expected layout "
                f"(missing {missing[:5]}, unexpected {extra[:5]})"
            )
        df = df.withColumn("_source_file", F.col("_metadata.file_name")).withColumn(
            "_loaded_at", F.current_timestamp()
        )
        path = table_path(s, table)
        df.write.format("delta").mode("overwrite").option("overwriteSchema", True).save(path)
        n = spark.read.format("delta").load(path).count()
        counts["/".join(table)] = n
        log.info("bronze %s: %s rows from %s", "/".join(table), f"{n:,}", src)

    if delete_source:
        for file_name in SOURCES:
            (raw_dir / file_name).unlink()
            log.info("deleted %s (bronze holds the only copy)", raw_dir / file_name)
    return counts
