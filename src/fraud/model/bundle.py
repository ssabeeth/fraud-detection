"""A trained model with everything needed to score: preprocessing, model, calibration.

Bundles are saved under ``data/models/<name>/`` (and logged to MLflow). The scoring
service and the stream processor load a bundle and call ``predict`` and ``reasons``;
they never re-implement preprocessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud.model.calibration import Calibrator
from fraud.model.inputs import Preprocessor
from fraud.model.rules import Rules


def as_strings(x: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Logistic regression one-hot encodes these columns, so it needs them as strings."""
    x = x.copy()
    for c in columns:
        x[c] = x[c].astype("object").where(x[c].notna(), None).astype(str)
    return x


@dataclass
class ModelBundle:
    name: str
    kind: str  # rules | logreg | lightgbm
    feature_set: str
    model: object
    preprocessor: Preprocessor | None = None
    calibrator: Calibrator | None = None
    meta: dict = field(default_factory=dict)

    # -- scoring -------------------------------------------------------------------------

    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.preprocessor is not None
        return self.preprocessor.transform(df)

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "rules":
            return self.model.score(df)
        x = self.matrix(df)
        if self.kind == "lightgbm":
            return self.model.predict(x)
        return self.model.predict_proba(as_strings(x, self.meta["string_columns"]))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated P(fraud), or the rules' points score for the rules baseline."""
        raw = self.raw_score(df)
        return self.calibrator(raw) if self.calibrator is not None else raw

    @property
    def is_probability(self) -> bool:
        return self.kind != "rules"

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature SHAP values (log-odds) from LightGBM's TreeSHAP."""
        if self.kind != "lightgbm":
            raise TypeError("contributions are available for LightGBM bundles")
        x = self.matrix(df)
        contrib = self.model.predict(x, pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=x.columns, index=df.index)

    def score_one(self, event: dict, k: int = 3) -> tuple[float, list[dict]]:
        """Calibrated P(fraud) and the top ``k`` reasons for one event (LightGBM only).

        One TreeSHAP pass gives both: the contributions plus the bias sum to the raw
        log-odds, so no separate prediction is needed.
        """
        from fraud.explain.reasons import reasons_for_row

        if self.kind != "lightgbm":
            raise TypeError("single-event scoring is implemented for LightGBM bundles")
        x = self.preprocessor.transform_one(event)
        contrib = self.model.predict(x, pred_contrib=True)
        raw = 1.0 / (1.0 + np.exp(-contrib[0].sum()))
        p = float(self.calibrator(np.array([raw]))[0]) if self.calibrator else float(raw)
        return p, reasons_for_row(contrib[0, :-1], self.preprocessor.features, event, k)

    # -- persistence ---------------------------------------------------------------------

    def save(self, root: Path) -> Path:
        d = root / self.name
        d.mkdir(parents=True, exist_ok=True)
        if self.kind == "lightgbm":
            self.model.save_model(str(d / "model.lgb"))
        elif self.kind == "logreg":
            joblib.dump(self.model, d / "model.joblib")
        else:
            (d / "rules.json").write_text(json.dumps(self.model.__dict__, indent=2))
        if self.preprocessor is not None:
            self.preprocessor.save(d / "preprocessor.json")
        if self.calibrator is not None:
            (d / "calibrator.json").write_text(self.calibrator.to_json())
        meta = {"name": self.name, "kind": self.kind, "feature_set": self.feature_set, **self.meta}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return d

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        meta = json.loads((path / "meta.json").read_text())
        kind = meta["kind"]
        if kind == "lightgbm":
            import lightgbm as lgb

            model = lgb.Booster(model_file=str(path / "model.lgb"))
        elif kind == "logreg":
            model = joblib.load(path / "model.joblib")
        else:
            model = Rules(**json.loads((path / "rules.json").read_text()))
        pre = path / "preprocessor.json"
        cal = path / "calibrator.json"
        return cls(
            name=meta["name"],
            kind=kind,
            feature_set=meta["feature_set"],
            model=model,
            preprocessor=Preprocessor.load(pre) if pre.exists() else None,
            calibrator=Calibrator.from_json(cal.read_text()) if cal.exists() else None,
            meta={k: v for k, v in meta.items() if k not in ("name", "kind", "feature_set")},
        )
