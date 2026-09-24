"""Entity keys: the pseudo-card, the device fingerprint and the email domain.

There is no card ID in the data. The pseudo-card key is the usual construction:
``card1`` (an issuer/card field) + ``addr1`` (billing region) + the day the card was
first seen, which is the transaction day minus ``D1`` ("days since the card's first
transaction"). All three are fields of the transaction itself, so the key is known at
the moment of the transaction and needs no lookahead.

Each key has a Spark implementation (offline) and a plain-Python one (online). They
must produce identical strings; ``tests/test_keys.py`` checks that on every fixture row.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import Column

SECONDS_PER_DAY = 86_400
NA = "NA"


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# --- online (plain Python) -------------------------------------------------------------


def card_key(card1: object, addr1: object, transaction_dt: int, d1: object) -> str:
    c = NA if _missing(card1) else str(int(card1))  # type: ignore[arg-type]
    a = NA if _missing(addr1) else str(int(addr1))  # type: ignore[arg-type]
    first_seen = (
        NA if _missing(d1) else str(int(transaction_dt // SECONDS_PER_DAY - float(d1)))  # type: ignore[arg-type]
    )
    return f"{c}_{a}_{first_seen}"


DEVICE_PARTS = ("DeviceType", "DeviceInfo", "id_30", "id_31", "id_33")


def device_key(row: dict) -> str | None:
    """A coarse device fingerprint: type, model/OS string, OS version, browser, screen.

    ``None`` when the transaction has no ``DeviceInfo`` (no identity record, or the
    field is blank), so those rows get no device features rather than one giant bucket.
    """
    if _missing(row.get("DeviceInfo")):
        return None
    return "|".join("" if _missing(row.get(p)) else str(row.get(p)) for p in DEVICE_PARTS)


def email_key(row: dict) -> str | None:
    v = row.get("P_emaildomain")
    return None if _missing(v) else str(v)


# --- offline (Spark; imported lazily so the scoring image needs no Spark) ----------------


def card_key_col() -> Column:
    from pyspark.sql import functions as F

    first_seen = F.floor(F.col("TransactionDT") / SECONDS_PER_DAY) - F.col("D1")
    return F.concat_ws(
        "_",
        F.coalesce(F.col("card1").cast("int").cast("string"), F.lit(NA)),
        F.coalesce(F.col("addr1").cast("int").cast("string"), F.lit(NA)),
        F.coalesce(first_seen.cast("bigint").cast("string"), F.lit(NA)),
    )


def device_key_col() -> Column:
    from pyspark.sql import functions as F

    parts = [F.coalesce(F.col(p).cast("string"), F.lit("")) for p in DEVICE_PARTS]
    return F.when(F.col("DeviceInfo").isNotNull(), F.concat_ws("|", *parts))


def email_key_col() -> Column:
    from pyspark.sql import functions as F

    return F.col("P_emaildomain")
