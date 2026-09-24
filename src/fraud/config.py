"""Project configuration, loaded once from ``configs/base.yaml``.

Every tunable decision lives in the YAML file so that code never hard-codes a date,
a delay or a boundary. ``FRAUD_DATA_DIR`` and ``FRAUD_CONFIG`` in the environment
override the data directory and the config file.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """``configs/`` in a checkout; the copy packaged in the wheel otherwise (Databricks)."""
    if env := os.environ.get("FRAUD_CONFIG_DIR"):
        return Path(env)
    repo = REPO_ROOT / "configs"
    return repo if repo.exists() else Path(__file__).parent / "_configs"


def reports_dir() -> Path:
    """Where generated reports go: ``reports/`` in a checkout, or ``FRAUD_REPORTS_DIR``."""
    return Path(os.environ.get("FRAUD_REPORTS_DIR") or REPO_ROOT / "reports")


def exports_dir() -> Path:
    """Aggregated exports for the dashboard: ``exports/``, or ``FRAUD_EXPORTS_DIR``."""
    return Path(os.environ.get("FRAUD_EXPORTS_DIR") or REPO_ROOT / "exports")


DEFAULT_CONFIG = config_dir() / "base.yaml"

SECONDS_PER_DAY = 86_400


class Interval(BaseModel):
    """A half-open date interval ``[start, end)``."""

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Interval:
        if self.start >= self.end:
            raise ValueError(f"interval start {self.start} must be before end {self.end}")
        return self


class Split(BaseModel):
    train: Interval
    valid: Interval
    test: Interval

    @field_validator("train", "valid", "test", mode="before")
    @classmethod
    def _from_pair(cls, v: object) -> object:
        if isinstance(v, list | tuple):
            start, end = v
            return {"start": start, "end": end}
        return v

    @model_validator(mode="after")
    def _contiguous(self) -> Split:
        if not (self.train.end == self.valid.start and self.valid.end == self.test.start):
            raise ValueError("train, valid and test must be contiguous and in that order")
        return self

    def names(self) -> tuple[str, ...]:
        return ("train", "valid", "test")


class KafkaTopics(BaseModel):
    transactions: str
    decisions: str
    labels: str


class Kafka(BaseModel):
    bootstrap_servers: str
    topics: KafkaTopics


class Settings(BaseModel):
    data_dir: Path
    anchor: datetime
    split: Split
    label_delay_days: int
    kafka: Kafka

    @field_validator("label_delay_days")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("label_delay_days must be >= 0")
        return v

    @property
    def label_delay_seconds(self) -> int:
        return self.label_delay_days * SECONDS_PER_DAY

    def to_datetime(self, transaction_dt: int) -> datetime:
        """Map a raw ``TransactionDT`` (seconds) to a naive UTC datetime."""
        return self.anchor + timedelta(seconds=int(transaction_dt))

    def to_seconds(self, when: date | datetime) -> int:
        """Map a date or datetime to seconds since the anchor (the ``TransactionDT`` scale)."""
        if not isinstance(when, datetime):
            when = datetime(when.year, when.month, when.day)
        return int((when - self.anchor).total_seconds())

    def split_bounds_seconds(self) -> dict[str, tuple[int, int]]:
        """Split boundaries on the ``TransactionDT`` scale, as ``[start, end)`` pairs."""
        return {
            name: (self.to_seconds(iv.start), self.to_seconds(iv.end))
            for name, iv in (
                ("train", self.split.train),
                ("valid", self.split.valid),
                ("test", self.split.test),
            )
        }

    def path(self, *parts: str) -> Path:
        """A path under the data directory."""
        return self.data_dir.joinpath(*parts)


def load_settings(config_path: Path | str | None = None) -> Settings:
    path = Path(config_path or os.environ.get("FRAUD_CONFIG") or config_dir() / "base.yaml")
    raw = yaml.safe_load(path.read_text())
    data_dir = Path(os.environ.get("FRAUD_DATA_DIR") or raw.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    raw["data_dir"] = data_dir
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def settings() -> Settings:
    """The process-wide settings (cached)."""
    return load_settings()
