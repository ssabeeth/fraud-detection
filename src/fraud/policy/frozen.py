"""The policy frozen in phase 5, as the online path and later phases load it."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from fraud.config import reports_dir
from fraud.policy.costs import CostModel


@dataclass(frozen=True)
class FrozenPolicy:
    review_threshold: float
    costs: CostModel

    @classmethod
    def load(cls, path: Path | None = None) -> FrozenPolicy:
        """From ``FRAUD_POLICY`` or ``reports/policy_frozen.json``."""
        path = Path(path or os.environ.get("FRAUD_POLICY") or reports_dir() / "policy_frozen.json")
        d = json.loads(path.read_text())
        if d["policy"]["type"] != "ExpectedLossPolicy":
            raise ValueError(f"the online path implements ExpectedLossPolicy, not {d['policy']}")
        return cls(d["policy"]["review_threshold"], CostModel(**d["costs"]))
