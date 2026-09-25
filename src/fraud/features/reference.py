"""A deliberately naive recomputation of every aggregate, for the point-in-time test.

For a transaction at time ``t`` it filters the raw history to the same entity and
``t - W <= t_e < t`` and computes each aggregate from scratch with pandas. It shares
nothing with the Spark or streaming code except the definitions, so a leak in either
of those shows up as a difference here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fraud.features.definitions import (
    AGGREGATES,
    LABEL_DELAY,
    LABEL_KINDS,
    STD_EPSILON,
    Aggregate,
)


def _one(a: Aggregate, prior: pd.DataFrame, row: pd.Series):
    """One aggregate from ``prior``: the entity's transactions strictly before ``row``."""
    t = row["TransactionDT"]
    if a.kind in LABEL_KINDS:
        known = prior[prior["TransactionDT"] < t - LABEL_DELAY]  # labels that had arrived
        frauds, labelled = int(known["isFraud"].sum()), len(known)
        if a.kind == "known_frauds":
            return frauds
        if a.kind == "known_labelled":
            return labelled
        return frauds / labelled if labelled else None
    h = prior if a.window is None else prior[prior["TransactionDT"] >= t - a.window]
    amounts = h["TransactionAmt"].to_numpy(dtype=float)
    if a.kind == "count":
        return len(h)
    if a.kind == "sum_amount":
        return float(amounts.sum()) if len(h) else 0.0
    if a.kind == "distinct":
        return int(h[a.column].dropna().nunique())
    if a.kind == "seconds_since_prev":
        return None if h.empty else int(t - h["TransactionDT"].max())
    if a.kind == "amount_zscore":
        if len(h) < 2:
            return None
        std = float(np.std(amounts, ddof=1))
        return (row["TransactionAmt"] - amounts.mean()) / std if std > STD_EPSILON else None
    if a.kind == "amount_ratio":
        if h.empty:
            return None
        mean = amounts.mean()
        return row["TransactionAmt"] / mean if mean > 0 else None
    if a.kind == "is_new":
        v = row[a.column]
        if _missing(v):
            return None
        return int(v not in set(h[a.column].dropna()))
    raise ValueError(a.kind)


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def recompute(raw: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute every aggregate for ``rows`` from ``raw`` history.

    ``raw`` needs TransactionDT, TransactionAmt and the entity key columns; ``rows`` is a
    subset of it (with TransactionID) whose features are wanted.
    """
    entities = sorted({a.entity for a in AGGREGATES})
    # Only the keys the sampled rows need, each with its whole history: splitting all of
    # raw into one frame per key (217,850 card keys) is what made this slow.
    groups = {}
    for e in entities:
        wanted = raw[raw[e].isin(set(rows[e].dropna()))]
        groups[e] = dict(tuple(wanted.groupby(e, sort=False)))
    out = []
    for _, row in rows.iterrows():
        rec = {"TransactionID": row["TransactionID"]}
        for e in entities:
            aggs = [a for a in AGGREGATES if a.entity == e]
            key = row[e]
            if _missing(key):
                rec.update({a.name: None for a in aggs})
                continue
            g = groups[e][key]
            prior = g[g["TransactionDT"] < row["TransactionDT"]]
            for a in aggs:
                rec[a.name] = _one(a, prior, row)
        out.append(rec)
    return pd.DataFrame(out)


def differences(expected: pd.DataFrame, actual: pd.DataFrame, rel_tol: float = 1e-6):
    """Rows where ``actual`` differs from ``expected``, as (TransactionID, feature, e, a)."""
    merged = expected.merge(actual, on="TransactionID", suffixes=("_exp", "_act"))
    if len(merged) != len(expected):
        raise AssertionError("some rows are missing from the actual features")
    diffs = []
    for a in AGGREGATES:
        for tid, e, v in zip(
            merged["TransactionID"], merged[f"{a.name}_exp"], merged[f"{a.name}_act"], strict=True
        ):
            if not _close(e, v, rel_tol):
                diffs.append((tid, a.name, e, v))
    return diffs


def _close(e, v, rel_tol: float) -> bool:
    e_missing = e is None or (isinstance(e, float) and math.isnan(e))
    v_missing = v is None or (isinstance(v, float) and math.isnan(v))
    if e_missing or v_missing:
        return e_missing and v_missing
    return math.isclose(float(e), float(v), rel_tol=rel_tol, abs_tol=1e-9)
