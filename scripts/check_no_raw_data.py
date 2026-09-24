"""Refuse to commit anything that looks like IEEE-CIS rows.

The competition rules forbid redistributing the data, so only aggregates may leave
the machine. This check runs as a pre-commit hook and in CI over every tracked file:

- a zip archive anywhere is rejected;
- a CSV, JSON(L) or Parquet file with a ``TransactionID`` column is rejected, unless it
  sits under ``tests/fixtures/`` and every ID is below 1,000,000 (the real IDs start
  at 2,987,000, and the synthetic generator numbers rows from 1).

Usage: ``python scripts/check_no_raw_data.py FILE [FILE ...]``
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

FIXTURE_DIR = "tests/fixtures/"
SYNTHETIC_ID_LIMIT = 1_000_000
DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}


def _ids_from_csv(path: Path) -> list[int] | None:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if "TransactionID" not in header:
            return None
        i = header.index("TransactionID")
        return [int(float(row[i])) for row in reader if row and row[i]]


def _ids_from_json(path: Path) -> list[int] | None:
    ids: list[int] = []
    text = path.read_text()
    records: list = []
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        obj = json.loads(text)
        records = obj if isinstance(obj, list) else [obj]
    found = False
    for r in records:
        if isinstance(r, dict) and "TransactionID" in r:
            found = True
            ids.append(int(r["TransactionID"]))
    return ids if found else None


def _ids_from_parquet(path: Path) -> list[int] | None:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if "TransactionID" not in table.column_names:
        return None
    return [int(v) for v in table.column("TransactionID").to_pylist() if v is not None]


def problem(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    if path.suffix == ".zip":
        return "zip archives are not allowed in the repository"
    if path.suffix not in DATA_SUFFIXES:
        return None
    reader = {
        ".csv": _ids_from_csv,
        ".json": _ids_from_json,
        ".jsonl": _ids_from_json,
        ".parquet": _ids_from_parquet,
    }[path.suffix]
    try:
        ids = reader(path)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if ids is None:
        return None
    if not path_str.replace("\\", "/").startswith(FIXTURE_DIR):
        return "has a TransactionID column outside tests/fixtures (row-level data)"
    if any(i >= SYNTHETIC_ID_LIMIT for i in ids):
        return f"has TransactionID >= {SYNTHETIC_ID_LIMIT:,}: looks like real competition data"
    return None


def main(argv: list[str]) -> int:
    failures = [(p, msg) for p in argv if (msg := problem(p))]
    for p, msg in failures:
        print(f"{p}: {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
