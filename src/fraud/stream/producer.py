"""Replay a period of transactions to Redpanda in event-time order, with delayed labels.

Two timelines are merged on one simulated clock:

- each transaction is published to ``transactions`` at its own ``TransactionDT``;
- its label is published to ``labels`` at ``TransactionDT + label delay``. Labels of
  transactions before the replay window whose delay ends inside it are published too, so
  a monitor watching the stream sees exactly the labels that would have arrived by then.

``speedup`` compresses time: at 3,600 an hour of traffic takes a second; 0 means as fast
as possible. One partition per topic keeps a single global event-time order; see
DECISIONS.md for what scaling out would change.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections.abc import Iterator

import pandas as pd

from fraud.stream.messages import encode, records

log = logging.getLogger(__name__)

TX, LABEL = 0, 1
LABEL_FIELDS = ["TransactionID", "TransactionDT", "isFraud", "card1", "addr1", "D1"]


def timeline(
    window: pd.DataFrame,
    earlier: pd.DataFrame,
    start: int,
    end: int,
    delay: int,
    labels_until: int | None = None,
) -> Iterator[tuple[int, int, dict]]:
    """(release time, kind, payload) in release order; transactions before labels on ties.

    Transactions are those in ``[start, end)``. Labels are released in
    ``[start, labels_until)`` (default ``end``); a later ``labels_until`` lets the clock run
    on after the last transaction so the window's own labels arrive, as they would in June.
    """
    labels_until = end if labels_until is None else labels_until
    txs = ((int(r["TransactionDT"]), TX, r) for r in records(window))
    labelled = pd.concat([earlier, window])[LABEL_FIELDS]
    labelled = labelled.assign(released_at=labelled["TransactionDT"] + delay)
    labelled = labelled[
        (labelled["released_at"] >= start) & (labelled["released_at"] < labels_until)
    ]
    labelled = labelled.sort_values(["released_at", "TransactionID"], kind="stable")
    labels = ((int(r["released_at"]), LABEL, _label(r)) for r in labelled.to_dict("records"))
    yield from heapq.merge(txs, labels, key=lambda x: (x[0], x[1]))


def _label(r: dict) -> dict:
    """A chargeback notice: the transaction it is for, its card fields and the label."""
    out = {k: int(r[k]) for k in ("TransactionID", "TransactionDT", "isFraud", "released_at")}
    for k in ("card1", "addr1", "D1"):
        v = r.get(k)
        if v is not None and not (isinstance(v, float) and v != v):
            out[k] = v
    return out


def replay(
    producer,
    topics: dict[str, str],
    events: Iterator[tuple[int, int, dict]],
    speedup: float = 0.0,
    flush_every: int = 5_000,
) -> dict:
    wall0 = time.perf_counter()
    sim0 = None
    counts = {"transactions": 0, "labels": 0}
    for i, (t, kind, payload) in enumerate(events):
        if sim0 is None:
            sim0 = t
        if speedup > 0:
            due = wall0 + (t - sim0) / speedup
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        if kind == TX:
            # How many labels went out before this transaction: the processor applies
            # exactly those first, so it sees what the offline features assume.
            payload = {**payload, "_sent_at": time.time(), "_labels_before": counts["labels"]}
            producer.produce(
                topics["transactions"], key=str(payload["TransactionID"]), value=encode(payload)
            )
            counts["transactions"] += 1
        else:
            producer.produce(
                topics["labels"], key=str(payload["TransactionID"]), value=encode(payload)
            )
            counts["labels"] += 1
        producer.poll(0)
        if i % flush_every == 0:
            producer.flush()
    producer.flush()
    counts["seconds"] = round(time.perf_counter() - wall0, 2)
    log.info("replayed %s", counts)
    return counts
