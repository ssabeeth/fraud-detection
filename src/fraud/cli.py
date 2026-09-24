"""Command-line entry point: ``fraud <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fraud import __version__
from fraud.config import REPO_ROOT, settings


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

    write_report(get_spark("profile"), settings(), args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud", description=__doc__)
    parser.add_argument("--version", action="version", version=f"fraud {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
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
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    p.set_defaults(func=_cmd_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
