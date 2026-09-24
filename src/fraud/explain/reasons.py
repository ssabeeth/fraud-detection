"""Reason codes: the features that pushed a decision towards fraud, in plain names.

LightGBM's TreeSHAP (``pred_contrib``) splits each prediction's log-odds into one term per
feature. The reasons are the features with the largest positive terms. Engineered and
readable fields have reviewer names from ``features/definitions.py``; Vesta's masked
columns get the family name Vesta published for them (for example "C13 — count of linked
entities (masked)"), which is as much as anyone outside Vesta can say.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fraud.features.definitions import reviewer_names

FAMILIES = {
    "C": "count of linked entities (masked)",
    "D": "days since an earlier event (masked)",
    "M": "match check (masked)",
    "V": "Vesta risk signal (masked)",
}
IDENTITY = {
    "id_30": "Operating system",
    "id_31": "Browser",
    "id_33": "Screen resolution",
    "DeviceInfo": "Device model",
    "id_12": "Identity check (masked)",
    "id_15": "Identity check (masked)",
    "id_19": "Network detail (masked)",
    "id_20": "Network detail (masked)",
}


def plain_name(feature: str) -> str:
    names = reviewer_names()
    if feature in names:
        return names[feature]
    if feature in IDENTITY:
        return IDENTITY[feature]
    if feature.startswith("id_"):
        return f"{feature} — identity or network detail (masked)"
    head = feature[0]
    if head in FAMILIES and feature[1:].isdigit():
        return f"{feature} — {FAMILIES[head]}"
    return feature


def _display(v) -> object:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, bool | np.bool_):
        return int(v)
    if isinstance(v, float | np.floating):
        return round(float(v), 4)
    if isinstance(v, int | np.integer):
        return int(v)
    return str(v)


def reasons_for_row(contrib, features, values, k: int = 3) -> list[dict]:
    """The ``k`` largest positive contributions of one row, with names and raw values."""
    contrib = np.asarray(contrib, dtype=float)
    order = np.argsort(-contrib)[:k]
    return [
        {
            "feature": features[j],
            "reason": plain_name(features[j]),
            "value": _display(values.get(features[j])),
            "contribution": round(float(contrib[j]), 4),
        }
        for j in order
        if contrib[j] > 0
    ]


def top_reasons(contrib: pd.DataFrame, values: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """For each row, the ``k`` features with the largest positive SHAP contributions."""
    cols = list(contrib.columns)
    arr = contrib.to_numpy()
    records = values.to_dict("records")
    return [reasons_for_row(arr[i], cols, records[i], k) for i in range(len(arr))]
