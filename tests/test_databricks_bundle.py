"""The Databricks Asset Bundle's tasks must be valid `fraud` commands, in pipeline order."""

from itertools import pairwise
from pathlib import Path

import yaml

from fraud.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def _tasks():
    resources = yaml.safe_load((ROOT / "databricks" / "resources.yml").read_text())
    return resources["resources"]["jobs"]["fraud_pipeline"]["tasks"]


def test_every_task_is_a_valid_cli_command():
    parser = build_parser()
    for task in _tasks():
        params = [
            p.replace("${var.data_dir}", "/Volumes/w/f/lake").replace("${var.raw_dir}", "/v/raw")
            for p in task["python_wheel_task"]["parameters"]
        ]
        args = parser.parse_args(params)
        assert args.data_dir == "/Volumes/w/f/lake", task["task_key"]
        assert task["python_wheel_task"]["entry_point"] == "fraud"


def test_tasks_run_in_pipeline_order():
    tasks = {t["task_key"]: t for t in _tasks()}
    order = ["bronze", "silver", "gold", "features", "check_pit", "train", "evaluate", "policy"]
    for before, after in pairwise(order):
        deps = {d["task_key"] for d in tasks[after].get("depends_on", [])}
        assert before in deps, f"{after} must depend on {before}"


def test_bundle_targets_free_edition_and_builds_the_wheel():
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())
    assert bundle["artifacts"]["fraud_wheel"]["type"] == "whl"
    assert "free" in bundle["targets"]
