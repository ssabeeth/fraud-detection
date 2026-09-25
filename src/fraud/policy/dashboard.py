"""A static dashboard of the test month, rebuilt from the daily export and the policy report.

The brief planned a Tableau Public workbook; the owner chose not to use Tableau, so the
same views are one HTML page instead: inline SVG, no JavaScript and no external files,
served by GitHub Pages from ``site/``. Month totals and tables come from
``reports/policy_results.json`` (so they match ``reports/policy.md`` to the dollar) and
the daily charts from ``exports/daily_policy_results.csv``. A test rebuilds the page and
compares it with the committed copy.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from fraud.config import REPO_ROOT, exports_dir, reports_dir
from fraud.policy.report import PARAM_LABELS

REPO_URL = "https://github.com/ssabeeth/fraud-detection"
DESCRIPTION = "What each card fraud decision policy cost on an out-of-sample month, in US dollars."
SITE = REPO_ROOT / "site" / "index.html"

ORDER = (
    "lightgbm_expected_loss",
    "lightgbm_cutoffs",
    "logreg_expected_loss",
    "rules",
    "approve_all",
)
NAMES = {
    "lightgbm_expected_loss": "LightGBM, expected loss",
    "lightgbm_cutoffs": "LightGBM, probability cut-offs",
    "logreg_expected_loss": "Logistic regression, expected loss",
    "rules": "Rules baseline",
    "approve_all": "Approve everything",
}

# Chart geometry, in SVG units; the charts scale to their container.
W, H = 600, 240
LEFT, RIGHT, TOP, BOTTOM = 54, 12, 12, 30


def build(
    daily_csv: Path | None = None, results_json: Path | None = None, out: Path | None = None
) -> Path:
    daily = pd.read_csv(daily_csv or exports_dir() / "daily_policy_results.csv")
    results = json.loads((results_json or reports_dir() / "policy_results.json").read_text())
    out = out or SITE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(daily, results))
    return out


def render(daily: pd.DataFrame, results: dict) -> str:
    daily = daily.assign(date=pd.to_datetime(daily["date"])).sort_values(["policy", "date"])
    test, chosen = results["test"], results["chosen"]
    month = daily["date"].min().strftime("%B %Y")
    pick, rules, approve = test[chosen], test["rules"], test["approve_all"]
    saving = rules["total_cost"] - pick["total_cost"]
    lows = [r for r in results["sensitivity"] if r["policy_test_cost"] < r["rules_test_cost"]]
    shares = [1 - r["policy_test_cost"] / r["rules_test_cost"] for r in results["sensitivity"]]
    capacity = results["frozen"]["costs"]["review_capacity_per_day"]

    body = [
        "<header>",
        f"<p class='eyebrow'>Card fraud decision policies · IEEE-CIS · {month} · US dollars</p>",
        f"<h1>What each decision policy cost in {month}</h1>",
        "<p class='lede'>"
        f"The chosen review policy ({html.escape(NAMES[chosen])}) catches "
        f"<strong>{pick['fraud_value_caught_share']:.1%} of fraud value</strong> at a cost of "
        f"<strong>{_usd(pick['total_cost'])}</strong>, against {_usd(rules['total_cost'])} for "
        f"the rules baseline, which catches {rules['fraud_value_caught_share']:.1%}. "
        + _robustness(len(lows), len(results["sensitivity"]), min(shares), max(shares))
        + "</p>",
        "</header>",
        "<section class='kpis' aria-label='Month totals'>",
        _kpi(
            "Fraud value caught",
            f"{pick['fraud_value_caught_share']:.1%}",
            f"rules {rules['fraud_value_caught_share']:.1%}",
        ),
        _kpi("Total cost", _usd(pick["total_cost"]), f"rules {_usd(rules['total_cost'])}"),
        _kpi(
            "Saved against the rules",
            _usd(saving),
            f"{saving / rules['total_cost']:.0%} less",
        ),
        _kpi("Approving everything", _usd(approve["total_cost"]), "no screening at all"),
        "</section>",
        _card(
            "Month totals by policy",
            _policy_table(test, chosen),
            "Every policy was tuned on April 2018 and frozen before May was scored. "
            "Costs follow the assumptions in configs/costs.yaml.",
            wide=True,
        ),
        "<div class='charts'>",
        _card(
            "Cost per day",
            _daily_cost_chart(daily),
            f"Approving everything is left out: it cost {_usd(approve['total_cost'])} over "
            "the month and would flatten the other lines.",
        ),
        _card(
            "Saving against the rules, cumulative",
            _saving_chart(daily, chosen),
            f"{NAMES[chosen]} against the rules baseline, summed day by day.",
        ),
        _card(
            "Where the money goes",
            _breakdown_chart(test),
            "Missed fraud is the amount plus the chargeback fee; false declines cost lost "
            "margin plus a fixed amount; reviews cost analyst time and a delay cost.",
        ),
        _card(
            "Review queue",
            _queue_chart(daily, chosen, capacity),
            f"{NAMES[chosen]}: reviews per day and how many were fraud, against the "
            f"capacity of {capacity} a day.",
        ),
        "</div>",
        _card(
            "Sensitivity: each cost assumption at both ends of its range",
            _sensitivity_table(results["sensitivity"]),
            "The frozen policy re-costed on May with one assumption changed at a time.",
            wide=True,
        ),
        _footer(results),
    ]
    return _page(f"Card fraud policies, {month}", "\n".join(body))


# --- pieces -----------------------------------------------------------------------------


def _robustness(cheaper: int, checks: int, lo: float, hi: float) -> str:
    if cheaper == checks:
        return (
            "It is cheaper than the rules at both ends of every cost assumption's range "
            f"({lo:.0%} to {hi:.0%} less)."
        )
    return f"It is cheaper than the rules in {cheaper} of {checks} sensitivity checks."


def _kpi(label: str, value: str, note: str) -> str:
    return (
        f"<div class='kpi'><p class='kpi-label'>{html.escape(label)}</p>"
        f"<p class='kpi-value'>{html.escape(value)}</p>"
        f"<p class='kpi-note'>{html.escape(note)}</p></div>"
    )


def _card(title: str, content: str, caption: str, wide: bool = False) -> str:
    cls = "card wide" if wide else "card"
    return (
        f"<section class='{cls}'><h2>{html.escape(title)}</h2>{content}"
        f"<p class='caption'>{html.escape(caption)}</p></section>"
    )


def _policy_table(test: dict, chosen: str) -> str:
    head = (
        "<tr><th>Policy</th><th>Total cost</th><th>Fraud value caught</th>"
        "<th>Declines</th><th>False declines</th><th>Reviews</th>"
        "<th>Reviews that were fraud</th></tr>"
    )
    rows = []
    for p in ORDER:
        r = test[p]
        cls = " class='chosen'" if p == chosen else ""
        rows.append(
            f"<tr{cls}><td><span class='swatch s-{p}'></span>{html.escape(NAMES[p])}</td>"
            f"<td>{_usd(r['total_cost'])}</td><td>{r['fraud_value_caught_share']:.1%}</td>"
            f"<td>{r['declines']:,}</td><td>{r['declined_legit']:,}</td>"
            f"<td>{r['reviews']:,}</td><td>{r['reviewed_fraud']:,}</td></tr>"
        )
    return _table(head, rows)


def _sensitivity_table(rows: list[dict]) -> str:
    head = (
        "<tr><th>Assumption</th><th>Value</th><th>Rules</th><th>Chosen policy</th>"
        "<th>Saving</th><th>Fraud value caught</th></tr>"
    )
    body = []
    for r in rows:
        saving = r["rules_test_cost"] - r["policy_test_cost"]
        body.append(
            f"<tr><td>{html.escape(PARAM_LABELS[r['parameter']])}</td><td>{r['value']:g}</td>"
            f"<td>{_usd(r['rules_test_cost'])}</td><td>{_usd(r['policy_test_cost'])}</td>"
            f"<td>{_usd(saving)} ({saving / r['rules_test_cost']:.0%})</td>"
            f"<td>{r['policy_test_caught_share']:.1%}</td></tr>"
        )
    return _table(head, body)


def _table(head: str, rows: list[str]) -> str:
    return (
        f"<div class='table'><table><thead>{head}</thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _footer(results: dict) -> str:
    ranks = results.get("rankings_valid", {})
    by_saving = ranks.get("expected saving", {}).get("total_cost")
    by_p = next((v["total_cost"] for k, v in ranks.items() if k != "expected saving"), None)
    ranking = (
        f" On April, ranking the review queue by expected saving cost {_usd(by_saving)}, "
        f"against {_usd(by_p)} when ranking by probability."
        if by_saving and by_p
        else ""
    )
    return (
        "<footer><p>Data: IEEE-CIS Fraud Detection (Vesta, via Kaggle), aggregated to daily "
        "totals; no transaction-level data is published. Months 1 to 4 trained the models, "
        f"April chose and froze the policies, and May was scored once.{ranking}</p>"
        f"<p>Built by <code>fraud dashboard</code> from "
        f"<a href='{REPO_URL}/blob/main/exports/daily_policy_results.csv'>the daily export</a> "
        f"and <a href='{REPO_URL}/blob/main/reports/policy.md'>the policy report</a>. "
        f"Code, method and decisions: <a href='{REPO_URL}'>github.com/ssabeeth/fraud-detection</a>."
        "</p></footer>"
    )


# --- charts -----------------------------------------------------------------------------


def _daily_cost_chart(daily: pd.DataFrame) -> str:
    shown = [p for p in ORDER if p != "approve_all"]
    days = sorted(daily["date"].unique())
    ymax = daily.loc[daily["policy"].isin(shown), "total_cost_usd"].max()
    x, y, ticks = _scales(len(days), 0.0, ymax)
    parts = [_axes(ticks, y, days, x)]
    for p in reversed(shown):  # the chosen policy is drawn last, on top
        d = daily[daily["policy"] == p].set_index("date")["total_cost_usd"].reindex(days)
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(d))
        parts.append(f"<polyline class='line l-{p}' points='{pts}'/>")
        for i, v in enumerate(d):
            tip = f"{_day(days[i])} · {NAMES[p]}: {_usd(v)}"
            parts.append(
                f"<circle class='dot l-{p}' cx='{x(i):.1f}' cy='{y(v):.1f}' r='2.6'>"
                f"<title>{html.escape(tip)}</title></circle>"
            )
    legend = "".join(
        f"<span><i class='swatch s-{p}'></i>{html.escape(NAMES[p])}</span>" for p in shown
    )
    return _svg("Cost per day by policy", "".join(parts)) + f"<p class='legend'>{legend}</p>"


def _saving_chart(daily: pd.DataFrame, chosen: str) -> str:
    wide = daily.pivot_table(
        index="date", columns="policy", values="total_cost_usd", aggfunc="sum"
    ).sort_index()
    cum = (wide["rules"] - wide[chosen]).cumsum()
    days = list(cum.index)
    x, y, ticks = _scales(len(days), min(0.0, cum.min()), cum.max())
    top = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(cum))
    area = f"{x(0):.1f},{y(0):.1f} {top} {x(len(days) - 1):.1f},{y(0):.1f}"
    parts = [
        _axes(ticks, y, days, x),
        f"<polygon class='area' points='{area}'/>",
        f"<polyline class='line l-{chosen}' points='{top}'/>",
    ]
    for i, v in enumerate(cum):
        tip = f"{_day(days[i])}: {_usd(v)} saved so far"
        parts.append(
            f"<circle class='dot l-{chosen}' cx='{x(i):.1f}' cy='{y(v):.1f}' r='2.6'>"
            f"<title>{html.escape(tip)}</title></circle>"
        )
    return _svg("Cumulative saving against the rules", "".join(parts))


def _breakdown_chart(test: dict) -> str:
    parts_of = {
        "missed": ("Missed fraud", lambda r: r["missed_fraud_cost"]),
        "declined": ("False declines", lambda r: r["declined_legit_cost"]),
        "review": ("Reviews", lambda r: r["review_cost"] + r["review_delay_cost"]),
    }
    label_w, row_h, bar_h = 176, 36, 20
    height = TOP + row_h * len(ORDER) + BOTTOM
    xmax = max(test[p]["total_cost"] for p in ORDER)
    ticks = _ticks(xmax)
    span = W - label_w - RIGHT

    def sx(v: float) -> float:
        return label_w + span * v / ticks[-1]

    out = []
    for t in ticks:
        out.append(
            f"<line class='gl' x1='{sx(t):.1f}' x2='{sx(t):.1f}' y1='{TOP}' "
            f"y2='{height - BOTTOM}'/><text class='tick' x='{sx(t):.1f}' "
            f"y='{height - BOTTOM + 16}' text-anchor='middle'>{_usd_short(t)}</text>"
        )
    for row, p in enumerate(ORDER):
        r, x0 = test[p], 0.0
        yb = TOP + row * row_h + (row_h - bar_h) / 2
        out.append(
            f"<text class='rowlabel' x='{label_w - 10}' y='{yb + bar_h / 2 + 4:.1f}' "
            f"text-anchor='end'>{html.escape(NAMES[p])}</text>"
        )
        for key, (name, get) in parts_of.items():
            v = get(r)
            if v <= 0:
                continue
            tip = f"{NAMES[p]} · {name}: {_usd(v)}"
            out.append(
                f"<rect class='seg g-{key}' x='{sx(x0):.1f}' y='{yb:.1f}' "
                f"width='{sx(x0 + v) - sx(x0):.1f}' height='{bar_h}'>"
                f"<title>{html.escape(tip)}</title></rect>"
            )
            x0 += v
        out.append(
            f"<text class='value' x='{sx(x0) + 6:.1f}' y='{yb + bar_h / 2 + 4:.1f}'>"
            f"{_usd(r['total_cost'])}</text>"
        )
    legend = "".join(
        f"<span><i class='swatch g-{k}'></i>{html.escape(n)}</span>"
        for k, (n, _) in parts_of.items()
    )
    return _svg("Cost breakdown by policy", "".join(out), height) + (
        f"<p class='legend'>{legend}</p>"
    )


def _queue_chart(daily: pd.DataFrame, chosen: str, capacity: int) -> str:
    d = daily[daily["policy"] == chosen].sort_values("date")
    days = list(d["date"])
    x, y, ticks = _scales(len(days), 0.0, max(capacity, d["reviews"].max()) * 1.1, pad=True)
    step = (W - LEFT - RIGHT) / len(days)
    bw = step * 0.7
    parts = [_axes(ticks, y, days, x, money=False)]
    for i, (reviews, fraud) in enumerate(zip(d["reviews"], d["reviews_fraud"], strict=True)):
        left = x(i) - bw / 2
        tip = f"{_day(days[i])}: {reviews} reviews, {fraud} were fraud"
        parts.append(
            f"<g><title>{html.escape(tip)}</title>"
            f"<rect class='bar' x='{left:.1f}' y='{y(reviews):.1f}' width='{bw:.1f}' "
            f"height='{y(0) - y(reviews):.1f}'/>"
            f"<rect class='bar-fraud' x='{left:.1f}' y='{y(fraud):.1f}' width='{bw:.1f}' "
            f"height='{y(0) - y(fraud):.1f}'/></g>"
        )
    parts.append(
        f"<line class='capacity' x1='{LEFT}' x2='{W - RIGHT}' y1='{y(capacity):.1f}' "
        f"y2='{y(capacity):.1f}'/><text class='tick cap' x='{W - RIGHT}' "
        f"y='{y(capacity) - 6:.1f}' text-anchor='end'>capacity {capacity} a day</text>"
    )
    legend = (
        "<span><i class='swatch bar'></i>Reviews</span>"
        "<span><i class='swatch bar-fraud'></i>Reviews that were fraud</span>"
    )
    return _svg("Review queue per day", "".join(parts)) + f"<p class='legend'>{legend}</p>"


def _scales(n: int, lo: float, hi: float, pad: bool = False):
    ticks = _ticks(hi, lo)
    y0, y1 = ticks[0], ticks[-1]
    inner = W - LEFT - RIGHT
    step = inner / n if pad else inner / max(n - 1, 1)

    def x(i: int) -> float:
        return LEFT + step * (i + 0.5 if pad else i)

    def y(v: float) -> float:
        return TOP + (H - TOP - BOTTOM) * (1 - (v - y0) / (y1 - y0))

    return x, y, ticks


def _axes(ticks, y, days, x, money: bool = True) -> str:
    out = []
    for t in ticks:
        label = _usd_short(t) if money else f"{t:g}"
        out.append(
            f"<line class='gl{' zero' if t == 0 else ''}' x1='{LEFT}' x2='{W - RIGHT}' "
            f"y1='{y(t):.1f}' y2='{y(t):.1f}'/>"
            f"<text class='tick' x='{LEFT - 8}' y='{y(t) + 4:.1f}' text-anchor='end'>{label}</text>"
        )
    for i in range(0, len(days), 7):
        out.append(
            f"<text class='tick' x='{x(i):.1f}' y='{H - BOTTOM + 18}' text-anchor='middle'>"
            f"{_day(days[i])}</text>"
        )
    return "".join(out)


def _ticks(hi: float, lo: float = 0.0, n: int = 4) -> list[float]:
    span = hi - lo if hi > lo else 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    count = math.ceil((hi - start) / step - 1e-9)
    return [round(start + i * step, 6) for i in range(count + 1)]


def _svg(label: str, inner: str, height: int = H) -> str:
    return (
        f"<svg viewBox='0 0 {W} {height}' role='img' aria-label='{html.escape(label)}'>"
        f"{inner}</svg>"
    )


def _usd(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _usd_short(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v / 1e6:g}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:g}k"
    return f"${v:g}"


def _day(ts) -> str:
    return f"{pd.Timestamp(ts).day} {pd.Timestamp(ts):%b}"


def _page(title: str, body: str) -> str:
    css = (Path(__file__).parent / "dashboard.css").read_text()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{DESCRIPTION}">
<style>
{css}</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""
