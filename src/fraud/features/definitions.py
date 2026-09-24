"""The single definition of every feature.

Two implementations read this list: ``offline.py`` turns each aggregate into a Spark
window expression, and ``online.py`` computes it from per-key state in the stream
processor. A third, deliberately naive one in ``reference.py`` recomputes features from
raw history for the point-in-time test. ``docs/features.md`` is generated from here.

**The point-in-time rule.** An aggregate for a transaction at time ``t`` sees only
transactions of the same entity with event time ``t_e < t`` (strictly before), and a
window of ``W`` seconds keeps those with ``t - W <= t_e``. Transactions in the same
second as the current one are not visible to it, whatever order they arrive in: the
stream cannot know which of two same-second events "came first", and the offline job
must not either. No feature uses a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HOUR = 3_600
DAY = 86_400

Entity = Literal["card_key", "device_key", "email_key"]
Kind = Literal[
    "count",  # transactions in the window
    "sum_amount",  # sum of TransactionAmt in the window
    "distinct",  # distinct non-null values of `column` in the window
    "seconds_since_prev",  # seconds since the entity's previous transaction
    "amount_zscore",  # (amount - mean of prior amounts) / their sample std
    "amount_ratio",  # amount / mean of prior amounts
    "is_new",  # 1 if `column`'s value has never been seen for this entity, else 0
]

# Below this the prior standard deviation is treated as zero, so both implementations
# agree on when the z-score is undefined despite rounding noise.
STD_EPSILON = 1e-9


@dataclass(frozen=True)
class Aggregate:
    name: str
    entity: Entity
    kind: Kind
    window: int | None  # seconds; None means all prior history
    reviewer_name: str
    column: str | None = None

    def definition(self) -> str:
        who = {
            "card_key": "this card",
            "device_key": "this device",
            "email_key": "this email domain",
        }[self.entity]
        span = "ever before" if self.window is None else f"in the previous {_span(self.window)}"
        if self.kind == "count":
            return f"Number of transactions by {who} {span}."
        if self.kind == "sum_amount":
            return f"Total USD spent by {who} {span}."
        if self.kind == "distinct":
            return f"Distinct {_col_name(self.column)} used by {who} {span}."
        if self.kind == "seconds_since_prev":
            return f"Seconds since {who}'s previous transaction (null if none)."
        if self.kind == "amount_zscore":
            return (
                f"Standard deviations between this amount and {who}'s previous amounts "
                "(null with fewer than two previous transactions)."
            )
        if self.kind == "amount_ratio":
            return f"This amount divided by the mean of {who}'s previous amounts (null if none)."
        return (
            f"1 if this {_col_name(self.column, plural=False)} has never been seen for "
            f"{who} before, else 0 (null when unknown)."
        )


def _span(seconds: int) -> str:
    if seconds % DAY == 0:
        d = seconds // DAY
        return "24 hours" if d == 1 else f"{d} days"
    return f"{seconds // HOUR} hour" + ("s" if seconds // HOUR != 1 else "")


def _col_name(column: str | None, plural: bool = True) -> str:
    base = {"device_key": "device", "email_key": "email domain", "card_key": "card"}[column or ""]
    return base + ("s" if plural else "")


AGGREGATES: list[Aggregate] = [
    # card velocity
    Aggregate("card_txn_1h", "card_key", "count", HOUR, "Card transactions, last hour"),
    Aggregate("card_txn_24h", "card_key", "count", DAY, "Card transactions, last 24 hours"),
    Aggregate("card_txn_7d", "card_key", "count", 7 * DAY, "Card transactions, last 7 days"),
    Aggregate("card_txn_30d", "card_key", "count", 30 * DAY, "Card transactions, last 30 days"),
    Aggregate("card_txn_prior", "card_key", "count", None, "Card's transaction history length"),
    Aggregate("card_amt_24h", "card_key", "sum_amount", DAY, "Card spend, last 24 hours"),
    Aggregate("card_amt_7d", "card_key", "sum_amount", 7 * DAY, "Card spend, last 7 days"),
    Aggregate(
        "card_secs_since_prev",
        "card_key",
        "seconds_since_prev",
        None,
        "Time since the card's last transaction",
    ),
    # card amount profile
    Aggregate("card_amt_zscore", "card_key", "amount_zscore", None, "Amount unusual for this card"),
    Aggregate(
        "card_amt_ratio", "card_key", "amount_ratio", None, "Amount vs the card's usual spend"
    ),
    # card identity changes
    Aggregate(
        "card_devices_30d",
        "card_key",
        "distinct",
        30 * DAY,
        "Devices used by the card, last 30 days",
        column="device_key",
    ),
    Aggregate(
        "card_emails_30d",
        "card_key",
        "distinct",
        30 * DAY,
        "Email domains used by the card, last 30 days",
        column="email_key",
    ),
    Aggregate(
        "card_new_device",
        "card_key",
        "is_new",
        None,
        "New device for this card",
        column="device_key",
    ),
    Aggregate(
        "card_new_email",
        "card_key",
        "is_new",
        None,
        "New email domain for this card",
        column="email_key",
    ),
    # device and email activity
    Aggregate("device_txn_24h", "device_key", "count", DAY, "Device transactions, last 24 hours"),
    Aggregate(
        "device_cards_7d",
        "device_key",
        "distinct",
        7 * DAY,
        "Cards seen on this device, last 7 days",
        column="card_key",
    ),
    Aggregate(
        "email_txn_7d", "email_key", "count", 7 * DAY, "How common this email domain is lately"
    ),
]

AGGREGATE_NAMES = [a.name for a in AGGREGATES]
ENTITIES: tuple[Entity, ...] = ("card_key", "device_key", "email_key")


@dataclass(frozen=True)
class LocalFeature:
    """A feature computed from the transaction's own fields (no history needed)."""

    name: str
    reviewer_name: str
    definition: str
    categorical: bool = False


# Features from the transaction itself, readable by a reviewer. Together with the
# aggregates they form the "explainable" feature set of phase 6.
LOCAL_FEATURES: list[LocalFeature] = [
    LocalFeature("TransactionAmt", "Amount (USD)", "Transaction amount in US dollars."),
    LocalFeature(
        "amt_cents", "Cents part of the amount", "TransactionAmt minus its whole-dollar part."
    ),
    LocalFeature(
        "hour", "Hour of day", "Hour of day relative to the time anchor (0-23, UTC-relative)."
    ),
    LocalFeature(
        "weekday", "Day of week", "Day of week relative to the anchor (0 = Thursday 30 Nov)."
    ),
    LocalFeature("ProductCD", "Product code", "Vesta product code (W, H, C, S, R).", True),
    LocalFeature("card4", "Card network", "visa, mastercard, american express, discover.", True),
    LocalFeature("card6", "Card type", "debit, credit, charge card.", True),
    LocalFeature("card1", "Card issuer code", "Masked card attribute (issuer/BIN-like)."),
    LocalFeature("card2", "Card code 2", "Masked card attribute."),
    LocalFeature("card3", "Card country code", "Masked card attribute (country-like)."),
    LocalFeature("card5", "Card code 5", "Masked card attribute."),
    LocalFeature("addr1", "Billing region", "Masked billing region code."),
    LocalFeature("addr2", "Billing country", "Masked billing country code."),
    LocalFeature("dist1", "Distance 1", "Masked distance between addresses."),
    LocalFeature("P_emaildomain", "Purchaser email domain", "Email domain of the purchaser.", True),
    LocalFeature("R_emaildomain", "Recipient email domain", "Email domain of the recipient.", True),
    LocalFeature(
        "email_match",
        "Purchaser and recipient domains match",
        "1 if both email domains are present and equal, 0 if both present and different.",
    ),
    LocalFeature("DeviceType", "Device type", "mobile or desktop (identity records only).", True),
    LocalFeature(
        "has_identity", "Has identity record", "1 if the transaction has an identity row."
    ),
]

LOCAL_NAMES = [f.name for f in LOCAL_FEATURES]
CATEGORICAL = [f.name for f in LOCAL_FEATURES if f.categorical]


def reviewer_names() -> dict[str, str]:
    """Plain names for every engineered and local feature (used in reason codes)."""
    return {
        **{a.name: a.reviewer_name for a in AGGREGATES},
        **{f.name: f.reviewer_name for f in LOCAL_FEATURES},
    }
