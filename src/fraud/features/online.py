"""Online features: the same definitions computed from per-key state, one event at a time.

The stream processor calls ``OnlineFeatures.process(event)`` for each transaction in
event-time order. For each entity key the state holds:

- *committed* history: every earlier transaction with a strictly smaller timestamp,
  trimmed to the longest window that entity needs, plus all-history summaries (count,
  running sum, Welford moments, last time, sets of values seen);
- *pending* transactions that share the latest timestamp seen for the key. They are
  folded into the committed history only when a later timestamp arrives, so an event
  never sees another event from the same second (the rule in ``definitions.py``).

The running statistics use the same update formulas as Spark's ``avg`` and
``stddev_samp``, in the same order, so the two implementations agree to rounding.

Labels reach the state through ``observe_label``, called when a label arrives (in the
stream, from the labels topic, before any later transaction). The card's count of known
frauds is the number of fraud labels observed; its count of labelled transactions needs
no label at all, since every transaction more than ``LABEL_DELAY`` old has one.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field

from fraud.features.definitions import (
    AGGREGATES,
    ENTITIES,
    LABEL_DELAY,
    LABEL_KINDS,
    STD_EPSILON,
    Aggregate,
)
from fraud.features.keys import SECONDS_PER_DAY, card_key, device_key, email_key

_AMOUNT = "TransactionAmt"
_TIME = "TransactionDT"


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def entity_keys(event: dict) -> dict[str, str | None]:
    return {
        "card_key": card_key(event.get("card1"), event.get("addr1"), event[_TIME], event.get("D1")),
        "device_key": device_key(event),
        "email_key": email_key(event),
    }


def local_features(event: dict) -> dict[str, float | int | None]:
    amt = float(event[_AMOUNT])
    t = int(event[_TIME])
    p, r = event.get("P_emaildomain"), event.get("R_emaildomain")
    return {
        "amt_cents": amt - math.floor(amt),
        "hour": (t % SECONDS_PER_DAY) // 3_600,
        "weekday": (t // SECONDS_PER_DAY) % 7,
        "email_match": None if _missing(p) or _missing(r) else int(p == r),
        "has_identity": int(bool(event.get("has_identity"))),
    }


@dataclass
class _KeyState:
    # committed, windowed history (sorted by time)
    times: list[int] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)
    values: dict[str, list] = field(default_factory=dict)
    # committed, all-history summaries
    n: int = 0
    amount_sum: float = 0.0
    w_mean: float = 0.0  # Welford mean (as Spark's CentralMomentAgg)
    w_m2: float = 0.0
    last_time: int | None = None
    seen: dict[str, set] = field(default_factory=dict)
    known_frauds: int = 0  # fraud labels observed for this key
    # same-second events not yet visible
    pending: list[tuple[int, float, dict]] = field(default_factory=list)


class OnlineFeatures:
    """Per-key state and feature computation for the stream processor."""

    def __init__(self, aggregates: Iterable[Aggregate] = AGGREGATES) -> None:
        self.aggregates = list(aggregates)
        self._by_entity = {e: [a for a in self.aggregates if a.entity == e] for e in ENTITIES}
        label_entities = {a.entity for a in self.aggregates if a.kind in LABEL_KINDS}
        if not label_entities <= {"card_key"}:
            raise ValueError("label aggregates are defined for the card key only")
        self._max_window = {
            e: max(
                [
                    *(a.window for a in aggs if a.window is not None),
                    *([LABEL_DELAY] if e in label_entities else []),
                ],
                default=0,
            )
            for e, aggs in self._by_entity.items()
        }
        self._columns = {
            e: sorted({a.column for a in aggs if a.column}) for e, aggs in self._by_entity.items()
        }
        self._state: dict[str, dict[str, _KeyState]] = {e: {} for e in ENTITIES}

    # -- state ---------------------------------------------------------------------------

    def _commit(self, entity: str, st: _KeyState, now: int) -> None:
        """Fold pending events into history once a strictly later timestamp arrives."""
        if st.pending and st.pending[0][0] < now:
            for t, amt, vals in st.pending:
                st.times.append(t)
                st.amounts.append(amt)
                for c in self._columns[entity]:
                    st.values.setdefault(c, []).append(vals[c])
                    if vals[c] is not None:
                        st.seen.setdefault(c, set()).add(vals[c])
                st.n += 1
                st.amount_sum += amt
                delta = amt - st.w_mean
                delta_n = delta / st.n
                st.w_mean += delta_n
                st.w_m2 += delta * (delta - delta_n)
                st.last_time = t
            st.pending.clear()
        # Drop history older than the longest window this entity needs.
        cutoff = bisect_left(st.times, now - self._max_window[entity])
        if cutoff:
            del st.times[:cutoff]
            del st.amounts[:cutoff]
            for c in st.values:
                del st.values[c][:cutoff]

    def _compute(self, a: Aggregate, st: _KeyState, t: int, amt: float, vals: dict):
        if a.window is not None:
            lo = bisect_left(st.times, t - a.window)
        if a.kind == "count":
            return st.n if a.window is None else len(st.times) - lo
        if a.kind == "sum_amount":
            return st.amount_sum if a.window is None else float(sum(st.amounts[lo:]))
        if a.kind == "distinct":
            if a.window is None:
                return len(st.seen.get(a.column, ()))
            return len({v for v in st.values.get(a.column, [])[lo:] if v is not None})
        if a.kind == "seconds_since_prev":
            return None if st.last_time is None else t - st.last_time
        if a.kind == "amount_zscore":
            if st.n < 2:
                return None
            std = math.sqrt(st.w_m2 / (st.n - 1))
            return (amt - st.amount_sum / st.n) / std if std > STD_EPSILON else None
        if a.kind == "amount_ratio":
            if st.n == 0:
                return None
            mean = st.amount_sum / st.n
            return amt / mean if mean > 0 else None
        if a.kind == "is_new":
            v = vals[a.column]
            if v is None:
                return None
            return 0 if v in st.seen.get(a.column, ()) else 1
        if a.kind in LABEL_KINDS:
            # committed transactions older than the delay: all of them minus the recent ones
            labelled = st.n - (len(st.times) - bisect_left(st.times, t - LABEL_DELAY))
            if a.kind == "known_labelled":
                return labelled
            if a.kind == "known_frauds":
                return st.known_frauds
            return st.known_frauds / labelled if labelled else None
        raise ValueError(a.kind)  # pragma: no cover

    # -- public API ----------------------------------------------------------------------

    def process(self, event: dict) -> dict[str, float | int | str | None]:
        """Features for ``event`` from strictly earlier history, then remember the event.

        Events must arrive in non-decreasing ``TransactionDT`` order for each key.
        """
        t = int(event[_TIME])
        amt = float(event[_AMOUNT])
        keys = entity_keys(event)
        out: dict[str, float | int | str | None] = dict(keys)
        for entity in ENTITIES:
            key = keys[entity]
            aggs = self._by_entity[entity]
            if key is None:
                out.update({a.name: None for a in aggs})
                continue
            st = self._state[entity].get(key)
            if st is None:
                st = self._state[entity][key] = _KeyState()
            latest = st.pending[0][0] if st.pending else st.last_time
            if latest is not None and latest > t:
                raise ValueError(
                    f"event at {t} arrived after {latest} for {entity}={key}; "
                    "the online path needs time order per key"
                )
            self._commit(entity, st, t)
            vals = {c: keys[c] for c in self._columns[entity]}
            for a in aggs:
                out[a.name] = self._compute(a, st, t, amt, vals)
            st.pending.append((t, amt, vals))
        out.update(local_features(event))
        return out

    def observe_label(self, label: dict) -> None:
        """A label has arrived for an earlier transaction (its card fields, time and
        ``isFraud``). Only frauds change the state."""
        if not int(label.get("isFraud") or 0):
            return
        key = card_key(label.get("card1"), label.get("addr1"), label[_TIME], label.get("D1"))
        if key is not None:
            self._state["card_key"].setdefault(key, _KeyState()).known_frauds += 1

    def state_size(self) -> dict[str, int]:
        return {e: len(s) for e, s in self._state.items()}
