"""Write ``reports/model.md`` and its figures from the evaluation results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud.features.definitions import reviewer_names
from fraud.model import rules as rules_module
from fraud.model import train as train_module
from fraud.model.bundle import ModelBundle
from fraud.model.data import list_test_touches

log = logging.getLogger(__name__)

LABELS = {
    "rules": "Rules baseline",
    "logreg": "Logistic regression",
    "lightgbm": "LightGBM",
    "lightgbm_explainable": "LightGBM, explainable features",
}


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _plots(bundles, scored, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ("valid", "test"), strict=True):
        frame = scored[split]
        for name in bundles:
            p, r, _ = precision_recall_curve(frame["isFraud"], frame[f"score_{name}"])
            ax.plot(r, p, lw=1.3, label=LABELS.get(name, name))
        ax.axhline(frame["isFraud"].mean(), color="grey", ls=":", lw=0.8, label="fraud rate")
        ax.set_xlabel("recall")
        ax.set_title({"valid": "Validation (Apr 2018)", "test": "Test (May 2018)"}[split])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("precision")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    frame = scored["test"]
    for name, b in bundles.items():
        if not b.is_probability:
            continue
        p = frame[f"score_{name}"].to_numpy()
        y = frame["isFraud"].to_numpy()
        order = np.argsort(p)
        bins = np.array_split(order, 15)
        ax.plot(
            [p[g].mean() for g in bins],
            [y[g].mean() for g in bins],
            "o-",
            ms=3,
            lw=1,
            label=LABELS.get(name, name),
        )
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("predicted P(fraud)")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("Calibration on test (equal-count bins)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png", dpi=120)
    plt.close(fig)

    lgbm = bundles.get("lightgbm")
    if lgbm is not None:
        gain = (
            pd.Series(lgbm.model.feature_importance("gain"), index=lgbm.model.feature_name())
            .sort_values(ascending=False)
            .head(25)
        )
        names = reviewer_names()
        labels = [f"{names[c]} ({c})" if c in names else c for c in gain.index]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(range(len(gain))[::-1], gain.to_numpy() / gain.sum(), color="#4c72b0")
        ax.set_yticks(range(len(gain))[::-1], labels, fontsize=8)
        ax.set_xlabel("share of total gain among the top 25")
        ax.set_title("LightGBM: top 25 features by gain", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "lightgbm_importance.png", dpi=120)
        plt.close(fig)


def metrics_table(results: dict, split: str) -> list[str]:
    lines = [
        "| Model | PR-AUC | ROC-AUC | Recall at FPR 0.1% / 0.5% / 1% / 5% | Fraud $ caught at "
        "FPR 1% | Brier | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, by_split in results.items():
        m = by_split[split]
        rec = " / ".join(_pct(r["recall"]) for r in m["recall_at_fpr"])
        val = next(r for r in m["recall_at_fpr"] if r["fpr_target"] == 0.01)["value_recall"]
        brier = f"{m['brier']:.4f}" if "brier" in m else "—"
        ece = f"{m['ece']:.4f}" if "ece" in m else "—"
        lines.append(
            f"| {LABELS.get(name, name)} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {rec} "
            f"| {_pct(val)} | {brier} | {ece} |"
        )
    return lines


def _diagnostic_lines(d: dict, results: dict) -> list[str]:
    top = d["top_1pct"]
    share = d["share_card_txn_30d_over_100"]
    lr, lg = results["logreg"], results["lightgbm"]
    lr_drop = 1 - lr["test"]["pr_auc"] / lr["valid"]["pr_auc"]
    lg_drop = 1 - lg["test"]["pr_auc"] / lg["valid"]["pr_auc"]
    return [
        "### Why logistic regression falls on the test month",
        "",
        f"From April to May, logistic regression's PR-AUC falls by {lr_drop:.0%} "
        f"({lr['valid']['pr_auc']:.3f} to {lr['test']['pr_auc']:.3f}) and LightGBM's by "
        f"{lg_drop:.0%} ({lg['valid']['pr_auc']:.3f} to {lg['test']['pr_auc']:.3f}). "
        f"In May one pseudo-card key has {d['busiest_key_rows']:,} transactions "
        f"({d['busiest_key_fraud_rate']:.1%} fraud), probably a business account or several "
        f"cards sharing a key. {share['test']:.1%} of May's transactions come from a card with "
        f"more than 100 transactions in the previous 30 days, against {share['valid']:.2%} in "
        "April. Logistic regression extends its velocity terms linearly beyond anything it "
        "was trained on, so that one legitimate key fills its alert list; LightGBM's trees "
        "stop at the largest split they learnt:",
        "",
        "| Model | Top 1% of test scores | From the busiest key | Fraud among them |",
        "|---|---|---|---|",
        *(
            f"| {LABELS.get(n, n)} | {v['rows']:,} | {v['from_busiest_key']:,} "
            f"| {v['fraud_share']:.1%} |"
            for n, v in top.items()
        ),
        "",
        "The monitoring phase is where a shift like this should be caught before it costs "
        "money (reports/monitoring.md).",
        "",
        *_latency_lines(d.get("single_event_ms")),
    ]


def _latency_lines(t: dict | None) -> list[str]:
    if not t:
        return []
    return [
        "### Cost of one decision",
        "",
        f"One transaction at a time (median / 99th percentile over {t['rows']} validation "
        f"rows, on the development laptop): score {t['score_p50']:.2f} / {t['score_p99']:.2f} "
        "ms; score with exact TreeSHAP reasons "
        f"{t['score_and_reasons_p50']:.1f} / "
        f"{t['score_and_reasons_p99']:.1f} ms. TreeSHAP costs trees × leaves × depth², which "
        "is why the trees are limited to depth 8 (DECISIONS.md).",
        "",
    ]


def write_model_report(
    bundles: dict[str, ModelBundle],
    results: dict,
    scored: dict,
    valid: pd.DataFrame,
    out_dir: Path,
    diagnostic: dict | None = None,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if scored is not None:  # None: re-render text from saved metrics, keep the figures
        _plots(bundles, scored, fig_dir)
    lgbm, lr, rules = bundles["lightgbm"], bundles["logreg"], bundles["rules"]
    test_fraud_rate = results["lightgbm"]["test"]["fraud"] / results["lightgbm"]["test"]["rows"]
    test_ap = {n: results[n]["test"]["pr_auc"] for n in results}
    touches = list_test_touches()

    lines = [
        "# Models",
        "",
        "Generated by `fraud evaluate`. Every number comes from the frozen bundles in",
        "`data/models/`; tuning used the validation month only. Amounts are USD.",
        "",
        "**Split.** Train December 2017 to March 2018 (417,559 transactions), validation",
        "April 2018 (83,655), test May 2018 (89,326). No shuffling across time. The test",
        f"month has been read {len(touches)} time(s) so far; each read and its purpose is",
        "logged in `reports/test_touches.jsonl`.",
        "",
        "## Test month (May 2018)",
        "",
        *metrics_table(results, "test"),
        "",
        f"LightGBM's test PR-AUC is {test_ap['lightgbm'] / test_ap['rules']:.1f}x the rules "
        f"baseline's and {test_ap['lightgbm'] / test_ap['logreg']:.2f}x logistic "
        "regression's. PR-AUC of a random score equals the fraud rate "
        f"({test_fraud_rate:.2%}). Recall at a false-positive rate counts "
        "fraud caught when that share of legitimate transactions is flagged. The rules "
        "produce points, not probabilities, so they have no Brier score or ECE.",
        "",
        *(_diagnostic_lines(diagnostic, results) if diagnostic else []),
        "## Validation month (April 2018)",
        "",
        *metrics_table(results, "valid"),
        "",
        "![PR curves](figures/pr_curves.png)",
        "",
        "## How each model was fitted",
        "",
        f"- **Rules baseline** — five hand-written rules with points: amount ≥ "
        f"${rules.model.amount:,.0f} (2), ≥ {rules.model.txn_1h} card transactions in the "
        f"last hour (2), ≥ {rules.model.txn_24h} in 24 hours (1), 24-hour spend including this "
        f"one ≥ ${rules.model.spend_24h:,.0f} (1), and a new device for the card with amount ≥ "
        f"${rules.model.new_device_amount:,.0f} (1). Ties are broken by amount. The five "
        "thresholds were chosen from a grid of round values "
        f"({np.prod([len(v) for v in rules_module.GRID.values()]):,} combinations) by "
        "validation PR-AUC.",
        f"- **Logistic regression** — the explainable features without the masked card and "
        f"address codes; signed log transform, standardised, one-hot categories. "
        f"C = {lr.meta['params']['C']}, class weight = {lr.meta['params']['class_weight']}, "
        f"chosen from {len(train_module.LOGREG_GRID)} settings by validation PR-AUC. "
        "Calibration: "
        f"{lr.calibrator.method}.",
        f"- **LightGBM** — all {len(lgbm.preprocessor.features)} features (history aggregates,"
        " readable fields and Vesta's anonymised columns). "
        + ", ".join(
            f"{k} = {lgbm.meta['params'][k]}"
            for k in ("num_leaves", "min_child_samples", "scale_pos_weight")
        )
        + f", {lgbm.meta['trees']} trees (early stopping on validation PR-AUC), chosen from "
        f"{len(train_module.LGBM_GRID)} settings. Calibration: {lgbm.calibrator.method}.",
        "",
        "**Imbalance** is handled by weighting, tuned as a hyperparameter "
        "(`class_weight`, `scale_pos_weight`), not by oversampling. See DECISIONS.md.",
        "",
        "**Calibration.** The policy multiplies P(fraud) by the amount, so probabilities must "
        "be right, not just well ordered. The method was chosen by a two-fold time split "
        "inside the validation month (fit on one half, Brier score on the other):",
        "",
        "| Model | none | Platt | isotonic | chosen |",
        "|---|---|---|---|---|",
    ]
    for name in ("logreg", "lightgbm"):
        b = bundles[name]
        cb = b.meta["calibration_brier"]
        lines.append(
            f"| {LABELS[name]} | {cb['none']:.5f} | {cb['platt']:.5f} | {cb['isotonic']:.5f} "
            f"| {b.calibrator.method} |"
        )
    lines += [
        "",
        "![calibration](figures/calibration.png)",
        "",
        "![importance](figures/lightgbm_importance.png)",
        "",
        "## Not comparable with the Kaggle leaderboard",
        "",
        "The competition was scored by ROC-AUC on a later, unlabelled period, and the winning",
        "solutions identified clients across the training and test files and aggregated",
        "features over both, including transactions after the one being scored. Here the",
        "model is trained on four months, tuned on the fifth and tested on the sixth, and every",
        "history feature uses only earlier transactions. The numbers answer a different,",
        "stricter question and should not be ranked against the leaderboard.",
        "",
    ]
    (out_dir / "model.md").write_text("\n".join(lines))
    log.info("wrote %s", out_dir / "model.md")
