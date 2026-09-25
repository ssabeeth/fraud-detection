"""The stream processor: consume transactions, decide, publish decisions with reasons.

It keeps the per-key feature state in memory (``OnlineScorer``), warmed from the lake
with every transaction before the replay window, so the first streamed event already
has its full history. It stops after ``idle_seconds`` without a message.

It also reads the labels topic, because the card's chargeback history is a feature. Each
transaction says how many labels were published before it (``_labels_before``); it is
scored once that many have been applied, so it sees exactly the labels released before
it, whatever order the two topics are read in. A transaction waits only while one of
those labels is still in flight.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

from fraud.stream.messages import decode, encode
from fraud.stream.scorer import OnlineScorer

log = logging.getLogger(__name__)


def run(
    consumer,
    producer,
    scorer: OnlineScorer,
    topics: dict[str, str],
    *,
    max_events: int | None = None,
    idle_seconds: float = 10.0,
) -> dict:
    consumer.subscribe([topics["transactions"], topics["labels"]])
    processing_ms, end_to_end_ms = [], []
    n = labels_applied = 0
    waiting: deque = deque()
    first = last = None
    idle_since = time.perf_counter()

    def score(key, event) -> None:
        nonlocal n, first, last
        sent_at = event.pop("_sent_at", None)
        event.pop("_labels_before", None)
        decision = scorer.decide(event)
        decision["_produced_at"] = time.time()
        producer.produce(topics["decisions"], key=key, value=encode_decision(decision))
        producer.poll(0)
        processing_ms.append(decision["timing_ms"]["total"])
        if sent_at is not None:
            end_to_end_ms.append((decision["_produced_at"] - sent_at) * 1e3)
        first = first or time.perf_counter()
        last = time.perf_counter()
        n += 1

    while max_events is None or n < max_events:
        msg = consumer.poll(0.5)
        if msg is None:
            if time.perf_counter() - idle_since > idle_seconds:
                break
            continue
        if msg.error():
            raise RuntimeError(msg.error())
        idle_since = time.perf_counter()
        if msg.topic() == topics["labels"]:
            scorer.observe_label(decode(msg.value()))
            labels_applied += 1
        else:
            waiting.append((msg.key(), decode(msg.value())))
        while waiting and waiting[0][1].get("_labels_before", 0) <= labels_applied:
            score(*waiting.popleft())
            if max_events is not None and n >= max_events:
                break
    if waiting:
        log.warning("%d transactions still waiting for labels at the end", len(waiting))
    producer.flush()
    consumer.close()
    elapsed = (last - first) if n > 1 else 0.0
    stats = {
        "events": n,
        "seconds": round(elapsed, 2),
        "events_per_second": round(n / elapsed, 1) if elapsed else None,
        "processing_ms": _percentiles(processing_ms),
        "end_to_end_ms": _percentiles(end_to_end_ms),
        "labels_applied": labels_applied,
        "keys_in_state": scorer.features.state_size(),
    }
    log.info("processor: %s", stats)
    return stats


def encode_decision(decision: dict) -> bytes:
    return encode(decision)


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values)
    return {q: round(float(np.percentile(a, int(q[1:]))), 3) for q in ("p50", "p95", "p99")} | {
        "max": round(float(a.max()), 3),
        "mean": round(float(a.mean()), 3),
    }
