"""The GitHub Pages dashboard: rebuilt from committed files, its numbers from the report."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from fraud.cli import main
from fraud.policy import dashboard

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "exports" / "daily_policy_results.csv"
RESULTS = ROOT / "reports" / "policy_results.json"


@pytest.fixture(scope="module")
def page():
    return dashboard.render(pd.read_csv(CSV), json.loads(RESULTS.read_text()))


def test_committed_page_is_up_to_date(page):
    assert dashboard.SITE.read_text() == page, "run `make dashboard` and commit site/"


def test_headline_numbers_are_the_policy_report_s(page):
    results = json.loads(RESULTS.read_text())
    test, chosen = results["test"], results["chosen"]
    for policy in (chosen, "rules", "approve_all"):
        assert f"${test[policy]['total_cost']:,.0f}" in page, policy
    assert f"{test[chosen]['fraud_value_caught_share']:.1%} of fraud value" in page
    saving = test["rules"]["total_cost"] - test[chosen]["total_cost"]
    assert f"${saving:,.0f}" in page


def test_page_is_self_contained(page):
    assert "<script" not in page and "<link" not in page and "<img" not in page
    assert not re.search(r"""(src|url)\(?=?['"]?https?:""", page)


def test_cli_writes_the_page(tmp_path, monkeypatch):
    # read the committed export and report (the suite points both elsewhere by default)
    monkeypatch.setenv("FRAUD_REPORTS_DIR", str(RESULTS.parent))
    monkeypatch.setenv("FRAUD_EXPORTS_DIR", str(CSV.parent))
    out = tmp_path / "index.html"
    assert main(["dashboard", "--out", str(out)]) == 0
    assert out.read_text() == dashboard.SITE.read_text()
