"""Render ``docs/features.md`` from the feature definitions, so the two cannot drift."""

from __future__ import annotations

from fraud.features.definitions import AGGREGATES, LOCAL_FEATURES, _span

HEADER = """# Feature catalogue

Generated from `src/fraud/features/definitions.py` by `fraud catalogue`; a test fails if
this file is out of date. Amounts are USD.

**Point-in-time rule.** Every aggregate for a transaction at time `t` uses only
transactions of the same entity with event time strictly before `t`; a window of `W`
keeps those with `t - W <= t_e < t`. Transactions in the same second are not visible to
each other. The three `card_known_*` features use fraud labels, and only labels that had
arrived: a label arrives 30 days after its transaction (the chargeback delay), so they
count transactions with `t_e < t - 30 days`, and never see the current transaction's own
label. The offline (Spark) and online (stream) code both implement this list, and three
tests hold them to it: the point-in-time test against a naive recomputation from raw
history, a canary that must fail when a window includes the current row, and the
online/offline parity test, which releases each label 30 days after its transaction.

**Entities.** `card_key` is the pseudo-card (`card1` + `addr1` + first-seen day, see
`reports/data.md` for how stable it is); `device_key` is a coarse device fingerprint
(`DeviceType`, `DeviceInfo`, OS `id_30`, browser `id_31`, screen `id_33`), missing
without an identity record; `email_key` is the purchaser's email domain.
"""


def render() -> str:
    lines = [
        HEADER,
        "## History aggregates",
        "",
        "| Feature | Reviewer would call it | Entity | Window | Definition |",
        "|---|---|---|---|---|",
    ]
    for a in AGGREGATES:
        window = "all prior history" if a.window is None else _span(a.window)
        lines.append(
            f"| `{a.name}` | {a.reviewer_name} | `{a.entity}` | {window} | {a.definition()} |"
        )
    lines += [
        "",
        "## Transaction fields",
        "",
        "Computed from the transaction itself; no history involved.",
        "",
        "| Feature | Reviewer would call it | Type | Definition |",
        "|---|---|---|---|",
    ]
    for f in LOCAL_FEATURES:
        kind = "categorical" if f.categorical else "numeric"
        lines.append(f"| `{f.name}` | {f.reviewer_name} | {kind} | {f.definition} |")
    lines += [
        "",
        "## Anonymised fields",
        "",
        "`C1`-`C14`, `D1`-`D15`, `M1`-`M9`, `V1`-`V339` and the identity fields `id_01`-`id_38`",
        "and `DeviceInfo` are masked by Vesta. They are used as supplied in the *all features*",
        "model; phase 6 measures what it costs in money to leave them out. Vesta computed them",
        "and we cannot verify that each was available at transaction time; that is recorded as",
        "a limitation.",
        "",
    ]
    return "\n".join(lines)
