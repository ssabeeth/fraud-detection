"""Data-quality checks that fail the job.

Each check is a named condition that marks *failing* rows. All row-level checks run
in one aggregation pass. The results are written as JSON next to the lake, and if any
check fails the job raises ``DataQualityError`` before anything downstream is built.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

log = logging.getLogger(__name__)


class DataQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckResult:
    name: str
    failed_rows: int
    description: str

    @property
    def passed(self) -> bool:
        return self.failed_rows == 0


@dataclass(frozen=True)
class RowCheck:
    """A check that counts rows where ``fails`` is true."""

    name: str
    fails: Column
    description: str


def run_row_checks(df: DataFrame, checks: list[RowCheck]) -> list[CheckResult]:
    exprs = [
        F.sum(F.when(c.fails, 1).otherwise(0)).cast("long").alias(f"c{i}")
        for i, c in enumerate(checks)
    ]
    row = df.agg(*exprs).collect()[0]
    return [
        CheckResult(c.name, int(row[f"c{i}"] or 0), c.description) for i, c in enumerate(checks)
    ]


def unique_check(df: DataFrame, key: str) -> CheckResult:
    agg = df.agg(F.count(F.lit(1)).alias("n"), F.countDistinct(key).alias("d")).collect()[0]
    return CheckResult(f"{key}_unique", int(agg["n"] - agg["d"]), f"{key} is unique")


def enforce(results: list[CheckResult], stage: str, report_dir: Path | None = None) -> None:
    """Write the results, then raise if any check failed."""
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": stage,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "passed": all(r.passed for r in results),
            "checks": [asdict(r) for r in results],
        }
        (report_dir / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    failed = [r for r in results if not r.passed]
    for r in results:
        log.info("%s check %-32s %s", stage, r.name, "ok" if r.passed else f"{r.failed_rows} rows")
    if failed:
        detail = "; ".join(f"{r.name}: {r.failed_rows} rows ({r.description})" for r in failed)
        raise DataQualityError(f"{stage} data-quality checks failed: {detail}")
