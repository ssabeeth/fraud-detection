"""Where each Delta table lives.

Locally the lake is a directory under ``data_dir``; on Databricks the same layout sits
under a Unity Catalog volume (``FRAUD_DATA_DIR=/Volumes/...``), so the jobs never
change.
"""

from __future__ import annotations

from pathlib import Path

from fraud.config import Settings

BRONZE_TRANSACTION = ("bronze", "transaction")
BRONZE_IDENTITY = ("bronze", "identity")
SILVER_TRANSACTIONS = ("silver", "transactions")
GOLD_TRANSACTIONS = ("gold", "transactions")
GOLD_FEATURES = ("gold", "features")
QUALITY_DIR = ("_quality",)


def lake_dir(s: Settings) -> Path:
    return s.path("lake")


def table_path(s: Settings, table: tuple[str, ...]) -> str:
    return str(lake_dir(s).joinpath(*table))


def raw_dir(s: Settings) -> Path:
    return s.path("raw")
