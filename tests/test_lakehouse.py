import csv
import json
import shutil

import pandas as pd
import pytest

from fraud.lakehouse.bronze import build_bronze
from fraud.lakehouse.quality import DataQualityError
from fraud.lakehouse.schema import IDENTITY_COLUMNS, TRANSACTION_COLUMNS
from fraud.lakehouse.silver import build_silver
from fraud.lakehouse.tables import (
    BRONZE_TRANSACTION,
    GOLD_TRANSACTIONS,
    SILVER_TRANSACTIONS,
    lake_dir,
    table_path,
)
from tests.conftest import FIXTURES

pytestmark = pytest.mark.spark


def _csv_rows(name: str) -> int:
    with (FIXTURES / name).open() as f:
        return sum(1 for _ in f) - 1


def test_bronze_keeps_every_row_as_a_string(spark, lake):
    df = spark.read.format("delta").load(table_path(lake, BRONZE_TRANSACTION))
    assert df.count() == _csv_rows("train_transaction.csv")
    assert df.columns[: len(TRANSACTION_COLUMNS)] == TRANSACTION_COLUMNS
    assert {t for c, t in df.dtypes if c in TRANSACTION_COLUMNS} == {"string"}
    assert {"_source_file", "_loaded_at"} <= set(df.columns)


def test_silver_is_typed_joined_and_on_the_calendar(spark, lake):
    df = spark.read.format("delta").load(table_path(lake, SILVER_TRANSACTIONS))
    types = dict(df.dtypes)
    assert types["TransactionID"] == "bigint"
    assert types["isFraud"] == "int"
    assert types["TransactionAmt"] == "double"
    assert types["card4"] == "string"
    assert types["event_ts"] == "timestamp"
    assert df.count() == _csv_rows("train_transaction.csv")
    assert df.filter("has_identity").count() == _csv_rows("train_identity.csv")
    first = df.orderBy("TransactionDT").first()
    assert first["event_ts"] == lake.to_datetime(first["TransactionDT"])
    report = json.loads((lake_dir(lake) / "_quality" / "silver.json").read_text())
    assert report["passed"] is True


def test_gold_splits_follow_the_config(spark, lake):
    pdf = (
        spark.read.format("delta")
        .load(table_path(lake, GOLD_TRANSACTIONS))
        .select("TransactionDT", "split")
        .toPandas()
    )
    for name, (lo, hi) in lake.split_bounds_seconds().items():
        rows = pdf[pdf["split"] == name]
        assert len(rows) > 0
        assert rows["TransactionDT"].between(lo, hi - 1).all()
    assert pdf["split"].notna().all()


def _raw_copy(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("train_transaction.csv", "train_identity.csv"):
        shutil.copy(FIXTURES / name, raw / name)
    return raw


def _edit_transactions(raw, edit):
    path = raw / "train_transaction.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = edit(df)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


@pytest.mark.parametrize(
    ("edit", "check"),
    [
        (
            lambda d: d.assign(TransactionAmt=d["TransactionAmt"].where(d.index != 5, "-3.0")),
            "TransactionAmt_positive",
        ),
        (
            lambda d: d.assign(ProductCD=d["ProductCD"].where(d.index != 5, "Z")),
            "ProductCD_allowed",
        ),
        (lambda d: d.assign(isFraud=d["isFraud"].where(d.index != 5, "2")), "isFraud_binary"),
        (
            lambda d: d.assign(TransactionAmt=d["TransactionAmt"].where(d.index != 5, "abc")),
            "transaction_casts",
        ),
        # the same TransactionID twice with different content
        (
            lambda d: pd.concat([d, d.iloc[[3]].assign(TransactionAmt="1.0")]),
            "TransactionID_unique",
        ),
        # drop a transaction that has an identity row: the identity row becomes an orphan
        (
            lambda d: d[d["TransactionID"] != d["TransactionID"].iloc[_first_identity_row()]],
            "identity_orphans",
        ),
    ],
)
def test_silver_fails_the_job_on_bad_data(spark, lake_settings, tmp_path, edit, check):
    s = lake_settings.model_copy(update={"data_dir": tmp_path / "data"})
    raw = _raw_copy(tmp_path)
    _edit_transactions(raw, edit)
    build_bronze(spark, s, raw)
    with pytest.raises(DataQualityError, match=check):
        build_silver(spark, s)


def _first_identity_row() -> int:
    ids = pd.read_csv(FIXTURES / "train_identity.csv", usecols=["TransactionID"])
    tx = pd.read_csv(FIXTURES / "train_transaction.csv", usecols=["TransactionID"])
    return int(tx.index[tx["TransactionID"] == ids["TransactionID"].iloc[0]][0])


def test_exact_duplicates_are_dropped_not_failed(spark, lake_settings, tmp_path):
    s = lake_settings.model_copy(update={"data_dir": tmp_path / "data"})
    raw = _raw_copy(tmp_path)
    _edit_transactions(raw, lambda d: pd.concat([d, d.iloc[:3]]))
    build_bronze(spark, s, raw)
    assert build_silver(spark, s) == _csv_rows("train_transaction.csv")


def test_bronze_rejects_an_unexpected_header(spark, lake_settings, tmp_path):
    s = lake_settings.model_copy(update={"data_dir": tmp_path / "data"})
    raw = _raw_copy(tmp_path)
    _edit_transactions(raw, lambda d: d.drop(columns=["V339"]))
    with pytest.raises(DataQualityError, match="header"):
        build_bronze(spark, s, raw)


def test_bronze_can_delete_the_source(spark, lake_settings, tmp_path):
    s = lake_settings.model_copy(update={"data_dir": tmp_path / "data"})
    raw = _raw_copy(tmp_path)
    build_bronze(spark, s, raw, delete_source=True)
    assert not list(raw.glob("*.csv"))


def test_fixture_layout_matches_the_schema():
    with (FIXTURES / "train_transaction.csv").open() as f:
        assert next(csv.reader(f)) == TRANSACTION_COLUMNS
    with (FIXTURES / "train_identity.csv").open() as f:
        assert next(csv.reader(f)) == IDENTITY_COLUMNS
