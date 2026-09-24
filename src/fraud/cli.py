"""Command-line entry point: ``fraud <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from fraud import __version__
from fraud.config import settings


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud", description=__doc__)
    parser.add_argument("--version", action="version", version=f"fraud {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="print the resolved configuration")
    p.set_defaults(func=_cmd_config)
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
