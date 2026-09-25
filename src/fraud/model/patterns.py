"""Patterns in the training and validation months, and what each one changed.

Computed from December 2017 to April 2018 only, so the patterns can shape experiments
without looking at the month those experiments are finally judged on (May is not read).
Writes ``reports/data_patterns.json`` and ``reports/data_patterns.md`` with four figures.
Every consequence the report names is a decision already in DECISIONS.md or one of the
pre-registered experiments (``fraud.model.experiments``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import Settings, reports_dir
from fraud.lakehouse.schema import C_COLS, D_COLS, ID_COLS, M_COLS, V_COLS
from fraud.model.data import load_frame
from fraud.model.inputs import FEATURE_SETS, MAX_LEVELS, Preprocessor

log = logging.getLogger(__name__)

DAY = 86_400
FAMILIES = {"C": C_COLS, "D": D_COLS, "M": M_COLS, "V": V_COLS, "identity": ID_COLS}
CARDINAL = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31"]
D_SHOWN = ["D1", "D4", "D10", "D15"]
# What each feature measures, where that explains why its distribution moves with time.
DRIFT_REASONS = {
    "email_txn_7d": "it counts every transaction from the same email domain in the past "
    "week, so for a large domain it tracks the site's overall traffic, which changes by "
    "month.",
    "id_31": "browser and version; new versions appear as the months pass.",
    "id_30": "operating system and version; new versions appear as the months pass.",
    "card_txn_prior": "the length of the card's history, which can only grow while the data "
    "runs (the data starts on 1 December, so early histories are cut short).",
    "card_secs_since_prev": "depends on how much history exists, which grows with time.",
    "D10": "a raw day count (section 10).",
    "D11": "a raw day count (section 10).",
    "D15": "a raw day count (section 10).",
    "D4": "a raw day count (section 10).",
    "D1": "a raw day count (section 10).",
}


def run(spark, s: Settings, out_dir: Path | None = None) -> dict:
    df = load_frame(spark, s, ["train", "valid"], extra=["card1", "addr1"])  # never May
    result = compute(df, s.label_delay_days * DAY)
    out_dir = out_dir or reports_dir()
    (out_dir / "data_patterns.json").write_text(json.dumps(result, indent=2) + "\n")
    _figures(result, out_dir / "figures")
    write_report(result, out_dir)
    return result


def compute(df: pd.DataFrame, delay_seconds: int) -> dict:
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    y = df["isFraud"].to_numpy().astype(int)
    amt = df["TransactionAmt"].to_numpy(dtype=float)
    month = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m")
    out: dict = {"rows": len(df), "first_day": str(df["event_date"].min())}
    out["last_day"] = str(df["event_date"].max())
    out["imbalance"] = {
        "fraud": int(y.sum()),
        "rate": float(y.mean()),
        "value_share": float(amt[y == 1].sum() / amt.sum()),
    }
    out["months"] = [
        {
            "month": m,
            "rows": len(g),
            "fraud_rate": float(g["isFraud"].mean()),
            "fraud_value": float(g.loc[g["isFraud"] == 1, "TransactionAmt"].sum()),
        }
        for m, g in df.groupby(month, sort=True)
    ]
    out["products"] = [
        {"product": p, "share": float(len(g) / len(df)), "fraud_rate": float(g["isFraud"].mean())}
        for p, g in df.groupby("ProductCD")
    ]
    ident = df["has_identity"].fillna(0).astype(int)
    out["identity"] = {
        "share": float(ident.mean()),
        "fraud_rate_with": float(y[ident == 1].mean()),
        "fraud_rate_without": float(y[ident == 0].mean()),
    }
    out["missing"] = _missingness(df)
    out["cardinality"] = _cardinality(df)
    out["amounts"] = _amounts(df, y, amt)
    hour_rate = df.groupby("hour")["isFraud"].mean()
    out["hours"] = {
        "highest": {"hour": int(hour_rate.idxmax()), "fraud_rate": float(hour_rate.max())},
        "lowest": {"hour": int(hour_rate.idxmin()), "fraud_rate": float(hour_rate.min())},
    }
    out["clients"] = _clients(df, delay_seconds)
    out["d_columns"] = _d_columns(df)
    out["adversarial"] = _adversarial(df, month)
    return out


# --- the pieces --------------------------------------------------------------------------


def _missingness(df: pd.DataFrame) -> dict:
    fam = {}
    for name, cols in FAMILIES.items():
        cols = [c for c in cols if c in df]
        miss = df[cols].isna()
        fam[name] = {
            "columns": len(cols),
            "mean_missing": float(miss.to_numpy().mean()),
            "columns_over_half_missing": int((miss.mean() > 0.5).sum()),
        }
    v = [c for c in V_COLS if c in df]
    sample = df[v].sample(n=min(50_000, len(df)), random_state=0).isna()
    patterns = sample.T.drop_duplicates()
    fam["V"]["missingness_groups"] = len(patterns)
    return fam


def _cardinality(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in CARDINAL:
        if c not in df:
            continue
        counts = df[c].dropna().astype(str).value_counts()
        present = df[c].notna().sum()
        out.append(
            {
                "column": c,
                "distinct": len(counts),
                "present_share": float(present / len(df)),
                "top_levels_cover": float(counts.head(MAX_LEVELS).sum() / max(present, 1)),
            }
        )
    return out


def _amounts(df: pd.DataFrame, y: np.ndarray, amt: np.ndarray) -> dict:
    cents = (np.round(amt * 100) % 100) != 0
    q = [0.5, 0.9, 0.99]
    return {
        "legit_quantiles": [float(v) for v in np.quantile(amt[y == 0], q)],
        "fraud_quantiles": [float(v) for v in np.quantile(amt[y == 1], q)],
        "cents_share_legit": float(cents[y == 0].mean()),
        "cents_share_fraud": float(cents[y == 1].mean()),
        "fraud_rate_with_cents": float(y[cents].mean()),
        "fraud_rate_whole_dollars": float(y[~cents].mean()),
    }


def known_fraud_counts(df: pd.DataFrame, delay_seconds: int) -> np.ndarray:
    """For each row, how many earlier frauds on the same card key had a label that had
    arrived: fraud transactions at least ``delay_seconds`` before this one (strictly)."""
    out = np.zeros(len(df), dtype=np.int64)
    keyed = df["card_key"].notna().to_numpy()
    left = pd.DataFrame(
        {
            "key": df.loc[keyed, "card_key"].to_numpy(),
            "t": df.loc[keyed, "TransactionDT"].to_numpy() - delay_seconds,
            "i": np.flatnonzero(keyed),
        }
    ).sort_values("t", kind="stable")
    fraud = df.loc[keyed & (df["isFraud"].to_numpy() == 1), ["card_key", "TransactionDT"]]
    right = fraud.rename(columns={"card_key": "key", "TransactionDT": "t"}).sort_values(
        "t", kind="stable"
    )
    right["n"] = right.groupby("key").cumcount() + 1
    right = right.groupby(["key", "t"], as_index=False)["n"].max().sort_values("t", kind="stable")
    merged = pd.merge_asof(
        left, right, on="t", by="key", direction="backward", allow_exact_matches=False
    )
    out[merged["i"].to_numpy()] = merged["n"].fillna(0).to_numpy(dtype=np.int64)
    return out


def _clients(df: pd.DataFrame, delay_seconds: int) -> dict:
    keyed = df[df["card_key"].notna()]
    per_key = keyed.groupby("card_key")["isFraud"].agg(["size", "sum"])
    repeat = per_key[per_key["size"] >= 2]
    with_fraud = repeat[repeat["sum"] >= 1]
    y = df["isFraud"].to_numpy().astype(int)
    out = {
        "keys": len(per_key),
        "rows_in_repeat_keys": float(repeat["size"].sum() / len(keyed)),
        "repeat_keys_with_fraud": len(with_fraud),
        "fraud_share_within_fraud_keys": float(with_fraud["sum"].sum() / with_fraud["size"].sum()),
        "all_fraud_keys_share": float((with_fraud["sum"] == with_fraud["size"]).mean()),
    }
    for label, delay in (("delayed", delay_seconds), ("instant", 0)):
        known = known_fraud_counts(df, delay) > 0
        out[label] = {
            "rows_with_known_fraud": float(known.mean()),
            "fraud_rate_known": float(y[known].mean()) if known.any() else None,
            "fraud_rate_not_known": float(y[~known].mean()),
            "fraud_rows_with_known_fraud": float(known[y == 1].mean()),
        }
    return out


def _d_columns(df: pd.DataFrame) -> list[dict]:
    day = df["TransactionDT"] // DAY
    month = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m")
    keyed = df["card_key"].notna()
    out = []
    for c in D_SHOWN:
        raw = df[c]
        norm = day - raw
        sub = pd.DataFrame({"key": df["card_key"], "raw": raw, "norm": norm})[keyed & raw.notna()]
        sizes = sub.groupby("key")["raw"].transform("size")
        sub = sub[sizes >= 3]
        const = sub.groupby("key").agg(raw=("raw", "nunique"), norm=("norm", "nunique"))
        by_month = raw.groupby(month).mean()
        out.append(
            {
                "column": c,
                "present_share": float(raw.notna().mean()),
                "cards_constant_raw": float((const["raw"] == 1).mean()) if len(const) else None,
                "cards_constant_normalised": float((const["norm"] == 1).mean())
                if len(const)
                else None,
                "mean_by_month": {m: float(v) for m, v in by_month.items()},
            }
        )
    return out


def _adversarial(df: pd.DataFrame, month: pd.Series, top: int = 15) -> dict:
    """Can a model tell the training months (December to March) from April? AUC over
    three random folds, and the features it leans on most."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    target = (month == month.max()).astype(int).to_numpy()
    pre = Preprocessor(FEATURE_SETS["all"]).fit(df[target == 0])
    x = pre.transform(df)
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "feature_fraction": 0.5,
        "verbosity": -1,
        "seed": 7,
    }
    aucs, gain = [], np.zeros(x.shape[1])
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    for fit_idx, eval_idx in folds.split(x, target):
        b = lgb.train(params, lgb.Dataset(x.iloc[fit_idx], target[fit_idx]), num_boost_round=200)
        aucs.append(roc_auc_score(target[eval_idx], b.predict(x.iloc[eval_idx])))
        gain += b.feature_importance("gain")
    order = np.argsort(-gain)[:top]
    share = gain / gain.sum()
    return {
        "auc": float(np.mean(aucs)),
        "auc_folds": [float(a) for a in aucs],
        "top": [{"feature": x.columns[i], "gain_share": float(share[i])} for i in order],
    }


# --- figures and report ------------------------------------------------------------------


def _figures(r: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    months = [m["month"] for m in r["months"]]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(months, [m["fraud_rate"] * 100 for m in r["months"]], color="#2553a8")
    ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud rate by month, December to April")
    fig.tight_layout()
    fig.savefig(fig_dir / "patterns_monthly.png", dpi=120)
    plt.close(fig)

    fam = r["missing"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(list(fam), [v["mean_missing"] * 100 for v in fam.values()], color="#7c7c7c")
    ax.set_ylabel("cells missing (%)")
    ax.set_title("Missing values by column family")
    fig.tight_layout()
    fig.savefig(fig_dir / "patterns_missing.png", dpi=120)
    plt.close(fig)

    c = r["clients"]
    labels = ["no known\nchargeback", "known chargeback\n(30-day delay)"]
    rates = [c["delayed"]["fraud_rate_not_known"], c["delayed"]["fraud_rate_known"] or 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, [v * 100 for v in rates], color=["#7c7c7c", "#b5452f"])
    ax.set_ylabel("fraud rate (%)")
    ax.set_title("Fraud rate by the card's known history")
    fig.tight_layout()
    fig.savefig(fig_dir / "patterns_known_fraud.png", dpi=120)
    plt.close(fig)

    top = r["adversarial"]["top"][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([t["feature"] for t in top], [t["gain_share"] * 100 for t in top], color="#c07d1a")
    ax.set_xlabel("share of the adversarial model's gain (%)")
    ax.set_title("What separates the training months from April")
    fig.tight_layout()
    fig.savefig(fig_dir / "patterns_adversarial.png", dpi=120)
    plt.close(fig)


def _pct(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v:.{digits}%}"


def write_report(r: dict, out_dir: Path) -> Path:
    imb, ident, amts, cl = r["imbalance"], r["identity"], r["amounts"], r["clients"]
    months = r["months"]
    rates = [m["fraud_rate"] for m in months]
    hi = max(months, key=lambda m: m["fraud_rate"])
    lo = min(months, key=lambda m: m["fraud_rate"])
    prods = sorted(r["products"], key=lambda p: -p["fraud_rate"])
    fam = r["missing"]
    d = {x["column"]: x for x in r["d_columns"]}
    adv = r["adversarial"]
    dl, inst = cl["delayed"], cl["instant"]
    lift = (dl["fraud_rate_known"] or 0) / dl["fraud_rate_not_known"]
    lines = [
        "# Data patterns and what they changed",
        "",
        f"Generated by `fraud patterns` from {r['first_day']} to {r['last_day']} "
        f"({r['rows']:,} transactions): the training months and April. **May, the test "
        "month, is not read.** Each pattern ends with what it changed: a decision already "
        "made (DECISIONS.md) or one of the pre-registered experiments "
        "([experiments.md](experiments.md)).",
        "",
        "## 1. Fraud is rare, and larger than average",
        "",
        f"{imb['fraud']:,} frauds, **{imb['rate']:.2%}** of transactions but "
        f"**{imb['value_share']:.2%} of the money**. A model that flags nothing is "
        f"{1 - imb['rate']:.1%} accurate and useless.",
        "",
        "→ Judged by PR-AUC, recall at fixed false-positive rates and money, never accuracy "
        "or ROC-AUC alone. Imbalance is handled by weighting, not oversampling (which would "
        "distort the calibrated probabilities the money policy needs). *Experiment 9* tests "
        "a five-fold positive weight.",
        "",
        "## 2. The fraud rate moves month to month",
        "",
        "| Month | Transactions | Fraud rate | Fraud value |",
        "|---|---|---|---|",
        *(
            f"| {m['month']} | {m['rows']:,} | {m['fraud_rate']:.2%} | ${m['fraud_value']:,.0f} |"
            for m in months
        ),
        "",
        f"From {lo['fraud_rate']:.2%} ({lo['month']}) to {hi['fraud_rate']:.2%} "
        f"({hi['month']}): a relative swing of {(max(rates) - min(rates)) / min(rates):.0%}. "
        "Volume moves too: four of the five busiest days fall 20 to 24 December "
        "([data.md](data.md)).",
        "",
        "![Fraud rate by month](figures/patterns_monthly.png)",
        "",
        "→ Splits are by time only. Because one month can mislead, the model library and "
        "every experiment are judged on three scored months, not April alone "
        "([model_comparison.md](model_comparison.md)). *Experiment 8* asks whether older "
        "months still help or whether a recent window is better.",
        "",
        "## 3. Products differ sharply",
        "",
        "| Product | Share of transactions | Fraud rate |",
        "|---|---|---|",
        *(f"| {p['product']} | {p['share']:.1%} | {p['fraud_rate']:.2%} |" for p in prods),
        "",
        "→ ProductCD is a feature, and the segment checks in phase 6 report alert rates and "
        "precision per product.",
        "",
        "## 4. Having an identity record is itself a signal",
        "",
        f"{ident['share']:.1%} of transactions have an identity record. Their fraud rate is "
        f"{ident['fraud_rate_with']:.2%}, against {ident['fraud_rate_without']:.2%} without.",
        "",
        "→ `has_identity` is a feature, and identity columns are left missing rather than "
        "filled: the absence is informative. The monitoring's feed-health check watches the "
        "share of rows carrying these fields, since a silent identity feed would shift scores.",
        "",
        "## 5. Most masked columns are mostly missing, in blocks",
        "",
        "| Family | Columns | Cells missing | Columns over half missing |",
        "|---|---|---|---|",
        *(
            f"| {k} | {v['columns']} | {v['mean_missing']:.1%} | {v['columns_over_half_missing']} |"
            for k, v in fam.items()
        ),
        "",
        f"The {fam['V']['columns']} V columns fall into {fam['V']['missingness_groups']} groups "
        "that are missing together: they come from a handful of sources and are heavily "
        "redundant.",
        "",
        "![Missing values by family](figures/patterns_missing.png)",
        "",
        "→ No imputation: LightGBM routes missing values to their own branch, and the same "
        "matrix is built for one streamed event. *Experiment 2* asks whether the V columns "
        "earn their place at all.",
        "",
        "## 6. Several categories have long tails",
        "",
        "| Column | Distinct values | Present | Rows covered by the top 50 |",
        "|---|---|---|---|",
        *(
            f"| {c['column']} | {c['distinct']:,} | {c['present_share']:.1%} | "
            f"{c['top_levels_cover']:.1%} |"
            for c in r["cardinality"]
        ),
        "",
        f"→ Categorical columns keep their {MAX_LEVELS} most frequent training levels; the "
        "rest share one `__other__` level, so an unseen value in the stream cannot break the "
        "model. card1 and addr1 have too many values for that and are used as numbers and, "
        "above all, through the pseudo-card key.",
        "",
        "## 7. Amounts: fraud is a little larger in the middle, not in the tail",
        "",
        "| | Median | 90th percentile | 99th percentile | Amounts with cents |",
        "|---|---|---|---|---|",
        f"| Legitimate | ${amts['legit_quantiles'][0]:,.2f} | ${amts['legit_quantiles'][1]:,.2f} | "
        f"${amts['legit_quantiles'][2]:,.2f} | {amts['cents_share_legit']:.1%} |",
        f"| Fraud | ${amts['fraud_quantiles'][0]:,.2f} | ${amts['fraud_quantiles'][1]:,.2f} | "
        f"${amts['fraud_quantiles'][2]:,.2f} | {amts['cents_share_fraud']:.1%} |",
        "",
        f"The fraud rate is {amts['fraud_rate_with_cents']:.2%} for amounts with cents and "
        f"{amts['fraud_rate_whole_dollars']:.2%} for whole dollars: the amount alone separates "
        "the classes weakly.",
        "",
        "→ The amount matters less on its own than against the card's own history, so the "
        "per-card z-score and ratio are features. And because a missed large fraud costs "
        "more than a missed small one, the policy ranks the review queue by expected money, "
        "not probability.",
        "",
        "## 8. Fraud clusters by hour",
        "",
        f"The fraud rate is highest at hour {r['hours']['highest']['hour']} "
        f"({r['hours']['highest']['fraud_rate']:.2%}) and lowest at hour "
        f"{r['hours']['lowest']['hour']} ({r['hours']['lowest']['fraud_rate']:.2%}), relative "
        "to the time anchor.",
        "",
        "→ Hour of day and weekday are features.",
        "",
        "## 9. Fraud labels cover whole clients",
        "",
        f"{cl['keys']:,} pseudo-card keys; {cl['rows_in_repeat_keys']:.1%} of keyed "
        f"transactions belong to a key seen more than once. Among repeat keys with any fraud "
        f"({cl['repeat_keys_with_fraud']:,}), **{cl['fraud_share_within_fraud_keys']:.1%} of "
        f"their transactions are fraud**, and {cl['all_fraud_keys_share']:.1%} of them are "
        "fraud throughout: once a client is marked, later transactions tend to be marked "
        "too.",
        "",
        "| The card's earlier frauds are... | Rows with one | Fraud rate if so "
        "| Fraud rate if not | Frauds that had one |",
        "|---|---|---|---|---|",
        f"| known after the 30-day label delay | {dl['rows_with_known_fraud']:.2%} | "
        f"{_pct(dl['fraud_rate_known'])} | {dl['fraud_rate_not_known']:.2%} | "
        f"{dl['fraud_rows_with_known_fraud']:.1%} |",
        f"| known instantly (not realistic) | {inst['rows_with_known_fraud']:.2%} | "
        f"{_pct(inst['fraud_rate_known'])} | {inst['fraud_rate_not_known']:.2%} | "
        f"{inst['fraud_rows_with_known_fraud']:.1%} |",
        "",
        "![Fraud rate by the card's known history](figures/patterns_known_fraud.png)",
        "",
        f"With the realistic delay, a card with a known chargeback is {lift:.0f} times as "
        "likely to be fraud, but few frauds have one: the delay hides most of this "
        "signal. The top Kaggle solutions exploited the same client-level labelling using "
        "each client's transactions across the whole dataset, future ones included, which a "
        "live system cannot do.",
        "",
        "→ The pseudo-card key and its point-in-time aggregates exist because of this. "
        "*Experiment 5* adds the delayed-label history as features, and *experiment 4* "
        "per-card means of earlier C and D values, a softer way of recognising a client.",
        "",
        "## 10. D columns are day counts that grow with time",
        "",
        "| Column | Present | Repeat cards with a constant raw value "
        "| Constant after normalising (day − D) |",
        "|---|---|---|---|",
        *(
            f"| {c} | {d[c]['present_share']:.1%} | {_pct(d[c]['cards_constant_raw'])} | "
            f"{_pct(d[c]['cards_constant_normalised'])} |"
            for c in D_SHOWN
            if c in d
        ),
        "",
        "D1 is constant after normalising by construction: the pseudo-card key includes "
        "day − D1. For the others, normalising makes "
        + ", ".join(
            f"{c} constant for {_pct(d[c]['cards_constant_normalised'], 0)} of repeat cards "
            f"(raw {_pct(d[c]['cards_constant_raw'], 0)})"
            for c in D_SHOWN[1:]
            if c in d
        )
        + ": a partial client fingerprint, not a fixed one.",
        "",
        "Mean raw value by month: "
        + "; ".join(
            f"{c} " + ", ".join(f"{v:.0f}" for v in d[c]["mean_by_month"].values())
            for c in D_SHOWN
            if c in d
        )
        + ". Raw D values rise as the data goes on.",
        "",
        "→ The pseudo-card key already uses day − D1. *Experiment 3* normalises the other D "
        "columns the same way.",
        "",
        "## 11. Adversarial validation: what tells April from the training months",
        "",
        f"A model asked to tell April from December to March reaches **AUC "
        f"{adv['auc']:.3f}** (0.5 would mean no difference). Its top features:",
        "",
        "| Feature | Share of its gain |",
        "|---|---|",
        *(f"| `{t['feature']}` | {t['gain_share']:.1%} |" for t in adv["top"]),
        "",
        "![What separates the training months from April](figures/patterns_adversarial.png)",
        "",
        "Why the leading ones drift:",
        "",
        *(
            f"- `{t['feature']}`: {DRIFT_REASONS[t['feature']]}"
            for t in adv["top"][:10]
            if t["feature"] in DRIFT_REASONS
        ),
        "",
        "→ Features that mainly encode *when* a transaction happened can help a model on "
        "the months it saw and hurt it later. *Experiment 7* drops the ten most "
        "time-dependent; *experiment 3* removes the time trend from the D columns instead.",
        "",
    ]
    path = out_dir / "data_patterns.md"
    path.write_text("\n".join(lines))
    return path
