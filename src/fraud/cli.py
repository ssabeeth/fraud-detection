"""Command-line entry point: ``fraud <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from fraud import __version__
from fraud.config import REPO_ROOT, reports_dir, settings


def _cmd_config(_: argparse.Namespace) -> int:
    s = settings()
    print(
        json.dumps(
            {
                "data_dir": str(s.data_dir),
                "anchor": s.anchor.isoformat(),
                "split_seconds": s.split_bounds_seconds(),
                "label_delay_days": s.label_delay_days,
            },
            indent=2,
        )
    )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from fraud.lakehouse.download import download
    from fraud.lakehouse.tables import raw_dir

    for p in download(args.raw_dir or raw_dir(settings())):
        print(p)
    return 0


def _cmd_bronze(args: argparse.Namespace) -> int:
    from fraud.lakehouse.bronze import build_bronze
    from fraud.lakehouse.tables import raw_dir
    from fraud.spark import get_spark

    counts = build_bronze(
        get_spark("bronze"),
        settings(),
        args.raw_dir or raw_dir(settings()),
        delete_source=not args.keep_source,
    )
    print(json.dumps(counts))
    return 0


def _cmd_silver(_: argparse.Namespace) -> int:
    from fraud.lakehouse.silver import build_silver
    from fraud.spark import get_spark

    print(build_silver(get_spark("silver"), settings()))
    return 0


def _cmd_gold(_: argparse.Namespace) -> int:
    from fraud.lakehouse.gold import build_gold
    from fraud.spark import get_spark

    print(json.dumps(build_gold(get_spark("gold"), settings())))
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    from fraud.lakehouse.profile import write_report
    from fraud.spark import get_spark

    write_report(get_spark("profile"), settings(), args.out or reports_dir())
    return 0


def _cmd_features(_: argparse.Namespace) -> int:
    from fraud.features.offline import build_features
    from fraud.spark import get_spark

    print(build_features(get_spark("features"), settings()))
    return 0


def _cmd_check_pit(args: argparse.Namespace) -> int:
    from fraud.features.pit import check
    from fraud.spark import get_spark

    report = args.report or reports_dir() / "pit_check.json"
    diffs = check(get_spark("pit"), settings(), sample=args.sample or None, report=report)
    for d in diffs[:20]:
        print("DIFF", d)
    return 1 if diffs else 0


def _cmd_check_parity(args: argparse.Namespace) -> int:
    from fraud.features.parity import check
    from fraud.spark import get_spark

    report = args.report or reports_dir() / "parity_check.json"
    result = check(get_spark("parity"), settings(), set(args.splits), report=report)
    return 1 if result["mismatches"] else 0


def _cmd_train(args: argparse.Namespace) -> int:
    from fraud.model.pipeline import MODELS, train_models
    from fraud.spark import get_spark

    results = train_models(get_spark("train"), settings(), tuple(args.models or MODELS))
    print(json.dumps({k: round(v["pr_auc"], 4) for k, v in results.items()}))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from fraud.model.pipeline import evaluate, rerender_report
    from fraud.spark import get_spark

    if args.report_only:
        rerender_report(settings())
        return 0
    results = evaluate(get_spark("evaluate"), settings())
    print(json.dumps({k: round(v["test"]["pr_auc"], 4) for k, v in results.items()}))
    return 0


def _cmd_policy(args: argparse.Namespace) -> int:
    from fraud.policy.evaluate import rerank, run
    from fraud.spark import get_spark

    if args.rerank:
        rerank(get_spark("policy"), settings())
        return 0

    result = run(get_spark("policy"), settings())
    print(json.dumps({k: round(v["total_cost"]) for k, v in result["test"].items()}))
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    from fraud.stream.run import run_all

    s = settings()
    start = args.start or s.split.test.start
    end = args.end or s.split.test.end
    result = run_all(s, start, end, args.speedup)
    print(json.dumps({"processor": result["processor"], "parity": result["parity"]}, default=str))
    return 1 if result["parity"]["feature_mismatches"] else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from fraud.model.compare import LIBRARIES, run
    from fraud.spark import get_spark

    if args.report_only:
        from fraud.model.compare_report import write_report

        out = reports_dir()
        print(write_report(json.loads((out / "model_comparison.json").read_text()), out))
        return 0

    result = run(get_spark("compare"), settings(), libraries=tuple(args.libraries or LIBRARIES))
    print(json.dumps({k: round(v["mean_cost"]) for k, v in result["summary"].items()}))
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    from fraud.model.experiments import run
    from fraud.spark import get_spark

    if args.report_only:
        from fraud.model.experiments_report import write_report

        out = reports_dir()
        print(write_report(json.loads((out / "experiments.json").read_text()), out))
        return 0
    result = run(get_spark("experiments"), settings(), names=args.only)
    print(json.dumps({e["name"]: e["adopted"] for e in result["experiments"]}))
    return 0


def _cmd_patterns(_: argparse.Namespace) -> int:
    from fraud.model.patterns import run
    from fraud.spark import get_spark

    result = run(get_spark("patterns"), settings())
    print(json.dumps({"adversarial_auc": round(result["adversarial"]["auc"], 3)}))
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from fraud.policy.dashboard import build

    print(build(out=args.out))
    return 0


def _cmd_explain(_: argparse.Namespace) -> int:
    from fraud.explain.experiment import run
    from fraud.spark import get_spark

    result = run(get_spark("explain"), settings())
    print(json.dumps({k: round(v["test"]["total_cost"]) for k, v in result["comparison"].items()}))
    return 0


def _cmd_monitor(_: argparse.Namespace) -> int:
    from fraud.monitor.run import run
    from fraud.spark import get_spark

    result = run(get_spark("monitor"), settings())
    print(json.dumps({k: result[k]["retrain_on"] for k in ("replay", "identity_outage")}))
    return 0


def _cmd_cards(_: argparse.Namespace) -> int:
    from fraud.explain.cards import write_cards

    for path in write_cards():
        print(path)
    return 0


def _cmd_catalogue(args: argparse.Namespace) -> int:
    from fraud.features.catalogue import render

    args.out.write_text(render())
    print(args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud", description=__doc__)
    parser.add_argument("--version", action="version", version=f"fraud {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--data-dir", help="data directory (overrides FRAUD_DATA_DIR), e.g. a Databricks volume"
    )
    parser.add_argument("--reports-dir", help="where reports go (overrides FRAUD_REPORTS_DIR)")
    parser.add_argument(
        "--test-note",
        help="note added to every test-month read this run logs (sets FRAUD_TEST_PURPOSE_NOTE)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="print the resolved configuration")
    p.set_defaults(func=_cmd_config)

    p = sub.add_parser("download", help="download the two training CSVs from Kaggle")
    p.add_argument("--raw-dir", type=Path, default=None)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("bronze", help="CSV -> Delta bronze (deletes the CSVs afterwards)")
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--keep-source", action="store_true", help="do not delete the CSVs")
    p.set_defaults(func=_cmd_bronze)

    p = sub.add_parser("silver", help="bronze -> typed, joined, checked silver")
    p.set_defaults(func=_cmd_silver)

    p = sub.add_parser("gold", help="silver -> modelling table with splits and entity keys")
    p.set_defaults(func=_cmd_gold)

    p = sub.add_parser("profile", help="write reports/data.md from the gold table")
    p.add_argument("--out", type=Path, default=None, help="default: the reports directory")
    p.set_defaults(func=_cmd_profile)

    p = sub.add_parser("features", help="gold -> point-in-time feature table")
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("check-pit", help="recompute sampled features from raw history")
    p.add_argument("--sample", type=int, default=3000, help="rows to check (0 = all)")
    p.add_argument("--report", type=Path, default=None, help="default: reports/pit_check.json")
    p.set_defaults(func=_cmd_check_pit)

    p = sub.add_parser("check-parity", help="replay through the online features and compare")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--report", type=Path, default=None, help="default: reports/parity_check.json")
    p.set_defaults(func=_cmd_check_parity)

    p = sub.add_parser("train", help="fit rules, logistic regression and LightGBM")
    p.add_argument("--models", nargs="+", choices=["rules", "logreg", "lightgbm"])
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("evaluate", help="score frozen models on valid and test; reports/model.md")
    p.add_argument(
        "--report-only", action="store_true", help="re-render the report from saved metrics"
    )
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("policy", help="choose the policy on valid, freeze, report on test")
    p.add_argument(
        "--rerank",
        action="store_true",
        help="recompute only the validation ranking comparison and re-render the report",
    )
    p.set_defaults(func=_cmd_policy)

    p = sub.add_parser(
        "compare-models", help="LightGBM vs XGBoost vs CatBoost over three months (no test)"
    )
    p.add_argument("--libraries", nargs="+", choices=["lightgbm", "xgboost", "catboost"])
    p.add_argument(
        "--report-only", action="store_true", help="re-render the report from the saved results"
    )
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("patterns", help="data patterns in December to April (no test month)")
    p.set_defaults(func=_cmd_patterns)

    p = sub.add_parser("experiments", help="the pre-registered experiments (no test month)")
    p.add_argument("--only", nargs="+", help="run only these experiments (plus the baseline)")
    p.add_argument(
        "--report-only", action="store_true", help="re-render the report from the saved results"
    )
    p.set_defaults(func=_cmd_experiments)

    p = sub.add_parser("dashboard", help="write site/index.html from the export and the report")
    p.add_argument("--out", type=Path, default=None, help="default: site/index.html")
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("explain", help="explainable-only model, segment checks, global SHAP")
    p.set_defaults(func=_cmd_explain)

    p = sub.add_parser("stream", help="replay a period through Redpanda and the processor")
    p.add_argument("--start", type=date.fromisoformat, default=None)
    p.add_argument("--end", type=date.fromisoformat, default=None)
    p.add_argument("--speedup", type=float, default=0.0, help="0 = as fast as possible")
    p.set_defaults(func=_cmd_stream)

    p = sub.add_parser("monitor", help="drift and delayed-label performance on the replay")
    p.set_defaults(func=_cmd_monitor)

    p = sub.add_parser("cards", help="write docs/model_card.md and docs/data_card.md")
    p.set_defaults(func=_cmd_cards)

    p = sub.add_parser("catalogue", help="write docs/features.md from the definitions")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "features.md")
    p.set_defaults(func=_cmd_catalogue)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_dir:
        os.environ["FRAUD_DATA_DIR"] = args.data_dir
        settings.cache_clear()
    if args.reports_dir:
        os.environ["FRAUD_REPORTS_DIR"] = args.reports_dir
        os.environ.setdefault("FRAUD_EXPORTS_DIR", str(Path(args.reports_dir) / "exports"))
    if args.test_note:
        os.environ["FRAUD_TEST_PURPOSE_NOTE"] = args.test_note
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
