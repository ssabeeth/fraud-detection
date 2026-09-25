"""Write ``reports/experiments.md`` from the pre-registered experiments."""

from __future__ import annotations

from pathlib import Path

MONTHS = {"2018-02": "Feb", "2018-03": "Mar", "2018-04": "Apr"}


def _usd(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _signed(v: float) -> str:
    return f"+{_usd(v)}" if v > 0 else _usd(v)


def verdict(e: dict) -> str:
    """Why an experiment was or was not adopted, in the rule's terms."""
    if e.get("post_hoc"):
        return "Post hoc, added after the results were seen, so not eligible for adoption."
    months = len(e["saving_by_month"])
    cheaper = sum(x > 0 for x in e["saving_by_month"])
    if e["adopted"]:
        return f"**Adopted**: cheaper in all {months} months, interval above zero."
    if cheaper == 0:
        return f"Not adopted: costs more in all {months} months."
    if cheaper < months:
        return f"Not adopted: cheaper in {cheaper} of {months} months."
    return "Not adopted: cheaper in every month, but the interval includes zero."


def finding(e: dict) -> str:
    per_month = e["pooled_saving"] / len(e["saving_by_month"])
    lo, hi = (v / len(e["saving_by_month"]) for v in e["saving_interval"])
    pr = sum(e["pr_auc_change"]) / len(e["pr_auc_change"])
    direction = "saves" if per_month > 0 else "costs"
    return (
        f"**{e['question']}**: {direction} {_usd(abs(per_month))} a month on average "
        f"(95% interval {_signed(lo)} to {_signed(hi)} saved a month) and changes mean PR-AUC "
        f"by {pr:+.3f}. {verdict(e)}"
    )


def what_changes(result: dict) -> list[str]:
    passed = [e for e in result["experiments"] if e["adopted"]]
    if not passed:
        return [
            "Nothing passed the rule, so the model is unchanged. The ablations still say what "
            "each part of the feature set is worth."
        ]
    best = max(passed, key=lambda e: (e["pooled_saving"], -e["folds"][0]["features"]))
    out = []
    if len(passed) > 1:
        others = [e for e in passed if e is not best]
        rest = ", ".join(f'{_signed(e["pooled_saving"])} for "{e["question"]}"' for e in others)
        out.append(
            f"{len(passed)} changes pass the rule, and they overlap. "
            f"**{best['question']}** saves the most over the three months "
            f"({_signed(best['pooled_saving'])}, against "
            + rest
            + f") with {best['folds'][0]['features']} features, so it alone is adopted. The "
            "rule did not say what to do when overlapping changes both pass; this tie-break "
            "(most money saved, then fewest features) was decided after the results were seen, "
            "and the other additions each fail the rule on their own."
        )
    else:
        out.append(f"Adopted: **{best['question']}**.")
    out.append(
        "It is then built in the Spark and stream feature code with point-in-time and parity "
        "tests, the model is retrained and tuned on April by the usual pipeline, the policy "
        "re-frozen, and May scored once more as a new, logged result."
    )
    return out


def blocklist_check(exps: dict) -> list[str]:
    """Does the model with the label history do more than the current model plus a block
    list? Only when both were run."""
    if not {"label_history", "baseline_blocklist"} <= set(exps):
        return []
    from fraud.model.experiments import paired_saving

    lh, bl = exps["label_history"], exps["baseline_blocklist"]
    saving, lo, hi = paired_saving(lh["daily_costs"], bl["daily_costs"])
    months = len(saving)
    return [
        "## Is it more than a block list?",
        "",
        "Vesta's labels mark transactions that follow a reported chargeback on the same card "
        "as fraud too, so a model that knows about earlier chargebacks is partly learning "
        "that labelling rule. In production the rule's equivalent is a block list. So the "
        "post-hoc question is whether the model does better than the current model with a "
        "block list that declines any card with a known chargeback.",
        "",
        f"The block list alone saves {_signed(bl['pooled_saving'])} over the three months "
        f"({' / '.join(_signed(x) for x in bl['saving_by_month'])}). The model with the label "
        f"history saves a further {_signed(sum(saving))} on top of it "
        f"({' / '.join(_signed(x) for x in saving)}; 95% interval {_signed(lo)} to "
        f"{_signed(hi)}), "
        + (
            f"cheaper in all {months} months: it learns more from the history than the rule does."
            if all(x > 0 for x in saving) and lo > 0
            else "which does not clearly beat the rule."
        ),
        "",
    ]


def write_report(result: dict, out_dir: Path) -> Path:
    exps = {e["name"]: e for e in result["experiments"]}
    base = exps["baseline"]
    months = [MONTHS.get(m, m) for m in result["folds"]]
    setting = ", ".join(f"{k}={v}" for k, v in result["setting"].items())
    lines = [
        "# Experiments: what the data patterns suggested, and what held up",
        "",
        "Generated by `fraud experiments`. Pre-registered in DECISIONS.md before the run, "
        "with the adoption rule. The same three folds as the model comparison score "
        f"{', '.join(months)} once each; early stopping and calibration use the last "
        f"{result['inner_days']} days of each training window; **May is not read**. Every "
        f"experiment uses LightGBM with the same setting ({setting}, "
        f"learning rate {result['learning_rate']}), so only the thing under test changes. Each "
        "experiment's pattern is the section of [data_patterns.md](data_patterns.md) that "
        "raised it.",
        "",
        "Money is the cost of the untuned expected-loss policy (review whenever the expected "
        "saving is positive, within 50 a day). The interval for the saving resamples whole "
        f"days within each month ({result['bootstrap_draws']:,} draws). **Rule:** "
        f"{result['rule']}.",
        "",
        "| Experiment | Pattern | Features | PR-AUC " + " / ".join(months) + " | Mean PR-AUC | "
        "Saving vs current, " + " / ".join(months) + " | Pooled saving (95% interval) | "
        "Decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in result["experiments"]:
        f = e["folds"]
        if e["name"] == "baseline":
            saving, pooled, decision = "—", "—", "the reference"
        else:
            saving = " / ".join(_signed(x) for x in e["saving_by_month"])
            lo, hi = e["saving_interval"]
            pooled = f"{_signed(e['pooled_saving'])} ({_signed(lo)} to {_signed(hi)})"
            decision = (
                "post hoc"
                if e.get("post_hoc")
                else ("**passes the rule**" if e["adopted"] else "fails the rule")
            )
        lines.append(
            f"| {e['question']} | {e['pattern']} | {f[0]['features']} | "
            + " / ".join(f"{r['pr_auc']:.3f}" for r in f)
            + f" | {e['mean_pr_auc']:.3f} | {saving} | {pooled} | {decision} |"
        )
    lines += [
        "",
        f"The current features cost {_usd(base['mean_cost'])} a month on average over these "
        "months under this policy.",
        "",
        "## Findings",
        "",
    ]
    for e in result["experiments"]:
        if e["name"] != "baseline":
            lines.append(f"- {finding(e)}")
    lines += ["", "## What changes", "", *what_changes(result), ""]
    lines += blocklist_check(exps)
    lines.append("")
    path = out_dir / "experiments.md"
    path.write_text("\n".join(lines))
    return path
