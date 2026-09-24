"""The online (Python) entity keys must equal the offline (Spark) ones on every row."""

import math

import pytest

from fraud.features.keys import card_key, device_key, email_key
from fraud.lakehouse.tables import GOLD_TRANSACTIONS, table_path


def test_card_key_examples():
    assert card_key(13926, 315.0, 86_400 * 10 + 5, 3.0) == "13926_315_7"
    assert card_key(13926, math.nan, 86_400 * 10, 3.0) == "13926_NA_7"
    assert card_key(13926, 315.0, 86_400 * 10, None) == "13926_315_NA"
    # a card first seen before the data starts gives a negative first-seen day
    assert card_key(1000, 100.0, 86_400 * 2, 300.0) == "1000_100_-298"


def test_device_and_email_keys():
    row = {"DeviceType": "mobile", "DeviceInfo": "iOS Device", "id_30": None, "id_31": "x"}
    assert device_key(row) == "mobile|iOS Device||x|"
    assert device_key({"DeviceType": "mobile", "DeviceInfo": None}) is None
    assert email_key({"P_emaildomain": math.nan}) is None
    assert email_key({"P_emaildomain": "gmail.com"}) == "gmail.com"


@pytest.mark.spark
def test_online_keys_equal_offline_keys_on_every_fixture_row(spark, lake):
    rows = (
        spark.read.format("delta").load(table_path(lake, GOLD_TRANSACTIONS)).toPandas()
    ).to_dict("records")
    assert rows
    for r in rows:
        assert card_key(r["card1"], r["addr1"], r["TransactionDT"], r["D1"]) == r["card_key"]
        assert device_key(r) == r["device_key"]
        assert email_key(r) == r["email_key"]
