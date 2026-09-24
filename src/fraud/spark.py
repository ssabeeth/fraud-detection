"""A local Spark session with Delta Lake, configured the same way everywhere.

On Databricks the platform provides the session (with Delta built in), and
``get_spark`` returns it unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession

# Homebrew's OpenJDK 17 is not registered with /usr/libexec/java_home, so a fresh
# shell has no JAVA_HOME. Fall back to it when nothing else is set.
_JAVA_CANDIDATES = (
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
)


def _ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    for candidate in _JAVA_CANDIDATES:
        if Path(candidate, "bin", "java").exists():
            os.environ["JAVA_HOME"] = candidate
            return


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(app_name: str = "fraud", *, shuffle_partitions: int | None = None) -> SparkSession:
    """Return the active session, or build a local one with Delta."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    if on_databricks():
        return SparkSession.builder.getOrCreate()

    _ensure_java_home()
    from delta import configure_spark_with_delta_pip

    memory = os.environ.get("FRAUD_SPARK_DRIVER_MEMORY", "6g")
    partitions = shuffle_partitions or int(os.environ.get("FRAUD_SPARK_SHUFFLE_PARTITIONS", "16"))
    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("FRAUD_SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.databricks.delta.snapshotPartitions", "2")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
