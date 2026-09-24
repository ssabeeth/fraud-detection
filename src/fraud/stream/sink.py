"""Kafka → Spark Structured Streaming → Delta bronze.

Each topic lands in its own bronze table as received: the message key and value (raw
JSON) with Kafka's topic, partition, offset and timestamp. Checkpoints make the sink
exactly-once per table, so re-running it only appends new messages. ``available_now``
processes what is in the topics and stops (batch-style runs and tests); otherwise it
keeps running with a short trigger interval.
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from fraud.config import Settings
from fraud.lakehouse.tables import lake_dir

log = logging.getLogger(__name__)

KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:{version}"
STREAM_TABLES = {
    "transactions": "stream_transactions",
    "decisions": "stream_decisions",
    "labels": "stream_labels",
}


def stream_table(s: Settings, topic_role: str) -> str:
    return str(lake_dir(s) / "bronze" / STREAM_TABLES[topic_role])


def reset(s: Settings) -> None:
    """Empty the stream landing tables and their checkpoints before a new replay.

    Each replay recreates its topics, so offsets start again at zero; a checkpoint left
    from the previous replay would make the sink skip the new messages.
    """
    import shutil

    for role in STREAM_TABLES:
        shutil.rmtree(stream_table(s, role), ignore_errors=True)
        shutil.rmtree(lake_dir(s) / "_checkpoints" / STREAM_TABLES[role], ignore_errors=True)


def run_sink(spark: SparkSession, s: Settings, *, available_now: bool = True) -> dict[str, int]:
    queries = []
    for role, topic in s.kafka.topics.model_dump().items():
        raw = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", s.kafka.bootstrap_servers)
            .option("subscribe", topic)
            .option("startingOffsets", "earliest")
            .option("failOnDataLoss", "false")
            .load()
        )
        bronze = raw.select(
            F.col("key").cast("string").alias("key"),
            F.col("value").cast("string").alias("value"),
            "topic",
            "partition",
            "offset",
            F.col("timestamp").alias("kafka_timestamp"),
            F.current_timestamp().alias("_loaded_at"),
        )
        writer = (
            bronze.writeStream.format("delta")
            .option("checkpointLocation", str(lake_dir(s) / "_checkpoints" / STREAM_TABLES[role]))
            .outputMode("append")
        )
        writer = (
            writer.trigger(availableNow=True)
            if available_now
            else writer.trigger(processingTime="5 seconds")
        )
        queries.append((role, writer.start(stream_table(s, role))))
    for _, q in queries:
        q.awaitTermination()
    counts = {
        role: spark.read.format("delta").load(stream_table(s, role)).count() for role, _ in queries
    }
    log.info("stream bronze rows: %s", counts)
    return counts
