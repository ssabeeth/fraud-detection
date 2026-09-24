"""``fraud policy``: choose the decision policy on validation, freeze it, report it on test.

1. Score the validation month with every frozen model.
2. Tune each candidate policy on validation for money under the default cost model:
   the rules baseline, probability cut-offs on LightGBM, and the expected-loss policy
   on logistic regression and on LightGBM. Compare the two review rankings (rule 8).
3. Choose the cheapest model policy on validation and write it to
   ``reports/policy_frozen.json`` *before* the test month is read.
4. Read the test month once, report every policy there, and the sensitivity table: for
   each cost assumption at the ends of its range, re-tune the rules and the chosen policy
   type on validation and cost both on test.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fraud.config import Settings, exports_dir, reports_dir
from fraud.model.data import load_frame
from fraud.model.pipeline import load_bundles, score_frame
from fraud.policy import policy as P
from fraud.policy.costs import CostModel

NEVER = None  # a threshold of infinity (the level is never reached), stored as null


def _plain(d: dict) -> dict:
    """Policy parameters as strict JSON: an infinite threshold becomes null."""
    return {k: (NEVER if isinstance(v, float) and v == float("inf") else v) for k, v in d.items()}


def describe(name: str, pol) -> str:
    if name == "approve_all":
        return "no parameters"
    if name == "rules":
        decline = (
            "never declines"
            if pol.decline_at == float("inf")
            else f"declines at ≥ {pol.decline_at:.2f} points"
        )
        return f"{decline}; reviews at ≥ {pol.review_at:.2f} (points plus the amount tie-break)"
    if name.endswith("cutoffs"):
        return f"decline at P ≥ {pol.decline_at:.3f}, review at P ≥ {pol.review_at:.3f}"
    return (
        "decline when its expected cost is lowest; review when the expected saving "
        f"exceeds ${pol.review_threshold:,.2f}"
    )


log = logging.getLogger(__name__)

MODEL_POLICIES = ("logreg_expected_loss", "lightgbm_cutoffs", "lightgbm_expected_loss")


def tune_all(vs: pd.DataFrame, costs: CostModel) -> dict:
    return {
        "approve_all": P.ApproveAll(),
        "rules": P.tune_rules(vs, costs)[0],
        "logreg_expected_loss": P.tune_expected_loss(vs, costs, "score_logreg")[0],
        "lightgbm_cutoffs": P.tune_cutoffs(vs, costs, "score_lightgbm")[0],
        "lightgbm_expected_loss": P.tune_expected_loss(vs, costs, "score_lightgbm")[0],
    }


def retune(name: str, vs: pd.DataFrame, costs: CostModel):
    return {
        "rules": lambda: P.tune_rules(vs, costs)[0],
        "logreg_expected_loss": lambda: P.tune_expected_loss(vs, costs, "score_logreg")[0],
        "lightgbm_cutoffs": lambda: P.tune_cutoffs(vs, costs, "score_lightgbm")[0],
        "lightgbm_expected_loss": lambda: P.tune_expected_loss(vs, costs, "score_lightgbm")[0],
    }[name]()


def run(spark, s: Settings, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or reports_dir()
    costs = CostModel.load()
    bundles = load_bundles(s)
    vs = score_frame(bundles, load_frame(spark, s, ["valid"]))

    policies = tune_all(vs, costs)
    valid = {k: P.cost(p, vs, costs) for k, p in policies.items()}
    chosen = min(MODEL_POLICIES, key=lambda k: valid[k]["total_cost"])
    rankings = P.compare_rankings(vs, costs, "score_lightgbm")
    frozen = {
        "chosen": chosen,
        "policy": {"type": type(policies[chosen]).__name__, **_plain(asdict(policies[chosen]))},
        "rules_baseline": {"type": "RulesPolicy", **_plain(asdict(policies["rules"]))},
        "costs": asdict(costs),
        "models": {n: b.meta.get("trees", b.kind) for n, b in bundles.items()},
        "chosen_on": "validation month (April 2018)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    log.info("frozen policy: %s", frozen["policy"])

    test_frame = load_frame(
        spark,
        s,
        ["test"],
        test_purpose="phase 5: frozen policy, rules baseline and sensitivity table on test",
    )
    ts = score_frame(bundles, test_frame)
    test = {k: P.cost(p, ts, costs) for k, p in policies.items()}

    sensitivity = []
    ranges = CostModel.ranges()
    for param, (low, high) in ranges.items():
        for value in (low, high):
            c = costs.with_(**{param: type(getattr(costs, param))(value)})
            rules_pol = retune("rules", vs, c)
            model_pol = retune(chosen, vs, c)
            sensitivity.append(
                {
                    "parameter": param,
                    "value": value,
                    "rules_test_cost": P.cost(rules_pol, ts, c)["total_cost"],
                    "policy_test_cost": P.cost(model_pol, ts, c)["total_cost"],
                    "policy_test_caught_share": P.cost(model_pol, ts, c)[
                        "fraud_value_caught_share"
                    ],
                    "approve_all_test_cost": P.cost(P.ApproveAll(), ts, c)["total_cost"],
                }
            )
    daily = daily_results(ts, policies, costs)
    result = {
        "valid": valid,
        "test": test,
        "chosen": chosen,
        "frozen": frozen,
        "rankings_valid": rankings,
        "sensitivity": sensitivity,
        "settings": {name: describe(name, pol) for name, pol in policies.items()},
    }
    (out_dir / "policy_results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    from fraud.policy.report import write_policy_report

    write_policy_report(result, costs, out_dir)
    exports = exports_dir()
    exports.mkdir(exist_ok=True)
    daily.to_csv(exports / "daily_policy_results.csv", index=False)
    return result


def daily_results(ts: pd.DataFrame, policies: dict, costs: CostModel) -> pd.DataFrame:
    """Per-day aggregates under each policy on the test month (for the dashboard)."""
    rows = []
    for name, pol in policies.items():
        actions = pol.decide(ts, costs)
        for day, idx in ts.groupby("event_date").indices.items():
            r = costs.realised(
                actions[idx], ts["isFraud"].to_numpy()[idx], ts["TransactionAmt"].to_numpy()[idx]
            )
            rows.append(
                {
                    "date": str(day),
                    "policy": name,
                    "transactions": len(idx),
                    "fraud_transactions": int(ts["isFraud"].to_numpy()[idx].sum()),
                    "fraud_value_usd": round(r["fraud_value"], 2),
                    "fraud_value_caught_usd": round(r["fraud_value_caught"], 2),
                    "declines": r["declines"],
                    "false_declines": r["declined_legit"],
                    "reviews": r["reviews"],
                    "reviews_fraud": r["reviewed_fraud"],
                    "false_alarms": r["declined_legit"] + r["reviews"] - r["reviewed_fraud"],
                    "missed_fraud_cost_usd": round(r["missed_fraud_cost"], 2),
                    "decline_friction_usd": round(r["declined_legit_cost"], 2),
                    "review_cost_usd": round(r["review_cost"] + r["review_delay_cost"], 2),
                    "total_cost_usd": round(r["total_cost"], 2),
                }
            )
    return pd.DataFrame(rows)


def rerank(spark, s: Settings, out_dir: Path | None = None) -> dict:
    """Recompute the validation-only parts and re-render the report, reading no test data.

    The policies are re-tuned on validation (deterministic) and must match the frozen
    ones; the ranking comparison is recomputed; test results come from
    reports/policy_results.json unchanged.
    """
    from fraud.policy.report import write_policy_report

    out_dir = out_dir or reports_dir()
    result = json.loads((out_dir / "policy_results.json").read_text())
    costs = CostModel.load()
    vs = score_frame(load_bundles(s), load_frame(spark, s, ["valid"]))
    policies = tune_all(vs, costs)
    chosen = result["chosen"]
    frozen = json.loads((out_dir / "policy_frozen.json").read_text())
    retuned = _plain(asdict(policies[chosen]))
    if retuned["review_threshold"] != frozen["policy"]["review_threshold"]:
        raise RuntimeError(f"re-tuned policy {retuned} differs from the frozen one")
    frozen["policy"] = {"type": type(policies[chosen]).__name__, **retuned}
    frozen["rules_baseline"] = {"type": "RulesPolicy", **_plain(asdict(policies["rules"]))}
    (out_dir / "policy_frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    result["frozen"] = frozen
    result["settings"] = {name: describe(name, pol) for name, pol in policies.items()}
    result["rankings_valid"] = P.compare_rankings(vs, costs, "score_lightgbm")
    (out_dir / "policy_results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    write_policy_report(result, costs, out_dir)
    return result
