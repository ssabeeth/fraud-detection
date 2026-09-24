"""Feature sets and the preprocessing every model and the scorer share.

Two feature sets:

- ``explainable``: the point-in-time history aggregates and the readable transaction
  fields (``docs/features.md``). A reviewer can be told what each one means.
- ``all``: those plus Vesta's anonymised columns (C, D, M, V and the identity fields).

``Preprocessor`` is fitted on the training months only. It fixes the category levels
(the most frequent levels in training; anything else becomes ``__other__``), maps the
T/F flags to 1/0 and casts numbers to float32. Its state is plain JSON, so the scoring
service rebuilds exactly the same model matrix from a single event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.features.definitions import AGGREGATE_NAMES, CATEGORICAL, LOCAL_NAMES
from fraud.lakehouse.schema import C_COLS, D_COLS, ID_COLS, ID_STRING_COLS, M_COLS, V_COLS

OTHER = "__other__"
MAX_LEVELS = 50

M_FLAGS = [c for c in M_COLS if c != "M4"]
ANON_CATEGORICAL = ["M4", *sorted(ID_STRING_COLS), "DeviceInfo"]
ANON_NUMERIC = [
    *C_COLS,
    *D_COLS,
    *M_FLAGS,
    *V_COLS,
    *(c for c in ID_COLS if c not in ID_STRING_COLS),
]

FEATURE_SETS: dict[str, list[str]] = {
    "explainable": [*AGGREGATE_NAMES, *LOCAL_NAMES],
    "all": [*AGGREGATE_NAMES, *LOCAL_NAMES, *ANON_NUMERIC, *ANON_CATEGORICAL],
}
CATEGORICAL_COLUMNS = set(CATEGORICAL) | set(ANON_CATEGORICAL)

# Columns the modelling frame carries besides the features.
META = ["TransactionID", "TransactionDT", "event_date", "split", "isFraud", "card_key"]


def raw_columns(feature_set: str) -> list[str]:
    """Columns to read from the feature table for a feature set (features + meta)."""
    cols = FEATURE_SETS[feature_set]
    return list(dict.fromkeys([*META, "TransactionAmt", *cols]))


@dataclass
class Preprocessor:
    features: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def categorical(self) -> list[str]:
        return [c for c in self.features if c in CATEGORICAL_COLUMNS]

    def fit(self, train: pd.DataFrame) -> Preprocessor:
        for c in self.categorical:
            counts = train[c].dropna().astype(str).value_counts()
            self.levels[c] = [*sorted(counts.index[:MAX_LEVELS].tolist()), OTHER]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.features:
            col = df[c] if c in df else pd.Series(np.nan, index=df.index)
            if c in self.levels:
                s = col.astype("object")
                known = set(self.levels[c])
                s = s.where(s.isna(), s.astype(str))
                s = s.where(s.isna() | s.isin(known), OTHER)
                out[c] = pd.Categorical(s, categories=self.levels[c])
            elif c in M_FLAGS:
                out[c] = col.map({"T": 1.0, "F": 0.0}).astype("float32")
            else:
                out[c] = pd.to_numeric(col, errors="coerce").astype("float32")
        return pd.DataFrame(out, index=df.index)

    def transform_one(self, event: dict) -> np.ndarray:
        """One event as a (1, n) float array, without pandas: the online scoring path.

        Categories become their index in the fitted levels, which is exactly the code
        pandas gives them in ``transform``, so the booster sees the same numbers.
        """
        index = self._index()
        row = np.empty(len(self.features), dtype=np.float64)
        for i, c in enumerate(self.features):
            v = event.get(c)
            missing = v is None or (isinstance(v, float) and v != v)
            if c in index:
                row[i] = np.nan if missing else index[c].get(str(v), index[c][OTHER])
            elif c in M_FLAGS:
                row[i] = {"T": 1.0, "F": 0.0}.get(v, np.nan)
            else:
                row[i] = np.nan if missing else float(v)
        return row.reshape(1, -1)

    def _index(self) -> dict[str, dict[str, int]]:
        if getattr(self, "_idx", None) is None:
            self._idx = {c: {v: i for i, v in enumerate(lv)} for c, lv in self.levels.items()}
        return self._idx

    def to_json(self) -> str:
        return json.dumps({"features": self.features, "levels": self.levels})

    @classmethod
    def from_json(cls, text: str) -> Preprocessor:
        d = json.loads(text)
        return cls(features=d["features"], levels=d["levels"])

    def save(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        return cls.from_json(path.read_text())
