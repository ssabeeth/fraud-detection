"""The IEEE-CIS column layout and the types silver casts each column to.

The Kaggle CSVs are read into bronze as strings, exactly as received; silver applies
these types. The synthetic fixture generator uses the same lists, so CI exercises
every column the real data has.
"""

from __future__ import annotations

KEY = "TransactionID"
LABEL = "isFraud"
TIME = "TransactionDT"
AMOUNT = "TransactionAmt"

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
V_COLS = [f"V{i}" for i in range(1, 340)]

TRANSACTION_COLUMNS: list[str] = [
    KEY,
    LABEL,
    TIME,
    AMOUNT,
    "ProductCD",
    *(f"card{i}" for i in range(1, 7)),
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *C_COLS,
    *D_COLS,
    *M_COLS,
    *V_COLS,
]

ID_COLS = [f"id_{i:02d}" for i in range(1, 39)]
IDENTITY_COLUMNS: list[str] = [KEY, *ID_COLS, "DeviceType", "DeviceInfo"]

# Identity columns that hold categories rather than numbers.
ID_STRING_COLS = {
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
}

TRANSACTION_STRING_COLS = {
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *M_COLS,
}

LONG_COLS = {KEY, TIME}
INT_COLS = {LABEL, "card1"}


def spark_type(column: str) -> str:
    """The Spark SQL type silver casts a column to."""
    if column in LONG_COLS:
        return "bigint"
    if column in INT_COLS:
        return "int"
    if column in TRANSACTION_STRING_COLS or column in ID_STRING_COLS:
        return "string"
    if column in ("DeviceType", "DeviceInfo"):
        return "string"
    return "double"


# Allowed values for the categorical columns the pipeline relies on. Anything else
# fails the silver data-quality checks.
ALLOWED = {
    "ProductCD": {"W", "H", "C", "S", "R"},
    "card4": {"visa", "mastercard", "american express", "discover"},
    "card6": {"debit", "credit", "debit or credit", "charge card"},
    "DeviceType": {"mobile", "desktop"},
}

assert len(TRANSACTION_COLUMNS) == 394, len(TRANSACTION_COLUMNS)
assert len(IDENTITY_COLUMNS) == 41, len(IDENTITY_COLUMNS)
