"""The stream processor: consume transactions, decide, publish decisions with reasons.

It keeps the per-key feature state in memory (``OnlineScorer``), warmed from the lake
with every transaction before the replay window, so the first streamed event already
has its full history. It stops after ``idle_seconds`` without a message.
"""

from __future__ import annotations

import logging
import time

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
    consumer.subscribe([topics["transactions"]])
    processing_ms, end_to_end_ms = [], []
    n = 0
    first = last = None
    idle_since = time.perf_counter()
    while max_events is None or n < max_events:
        msg = consumer.poll(0.5)
        if msg is None:
            if time.perf_counter() - idle_since > idle_seconds:
                break
            continue
        if msg.error():
            raise RuntimeError(msg.error())
        idle_since = time.perf_counter()
        event = decode(msg.value())
        sent_at = event.pop("_sent_at", None)
        decision = scorer.decide(event)
        decision["_produced_at"] = time.time()
        producer.produce(topics["decisions"], key=msg.key(), value=encode_decision(decision))
        producer.poll(0)
        processing_ms.append(decision["timing_ms"]["total"])
        if sent_at is not None:
            end_to_end_ms.append((decision["_produced_at"] - sent_at) * 1e3)
        first = first or time.perf_counter()
        last = time.perf_counter()
        n += 1
    producer.flush()
    consumer.close()
    elapsed = (last - first) if n > 1 else 0.0
    stats = {
        "events": n,
        "seconds": round(elapsed, 2),
        "events_per_second": round(n / elapsed, 1) if elapsed else None,
        "processing_ms": _percentiles(processing_ms),
        "end_to_end_ms": _percentiles(end_to_end_ms),
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
