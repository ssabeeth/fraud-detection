"""The scoring API: ``/score``, ``/health`` and ``/metrics``.

``/score`` is stateless: the caller sends the transaction and the earlier transactions
of the same card, device and email domain (``history``), and the service computes the
point-in-time features from that history with the same online code as the stream
processor, scores, explains and recommends an action with the frozen phase 5 policy.
History at or after the transaction's own time is rejected, because it would leak the
future into the features.

The daily review capacity is a property of the queue, not of one request, so the API
reports whether a review is worth its cost (``review_recommended``) and leaves the
capacity to the queue; the stream processor enforces it.

Configuration: ``FRAUD_MODEL_DIR`` (a LightGBM bundle, default ``data/models/lightgbm``)
and ``FRAUD_POLICY`` (default ``reports/policy_frozen.json``).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from fraud import __version__
from fraud.config import REPO_ROOT
from fraud.features.definitions import LABEL_DELAY
from fraud.features.online import OnlineFeatures
from fraud.model.bundle import ModelBundle
from fraud.policy.costs import ACTION_NAMES, APPROVE, DECLINE, REVIEW
from fraud.policy.frozen import FrozenPolicy

REQUESTS = Counter("fraud_score_requests_total", "Scoring requests", ["outcome"])
ACTIONS = Counter("fraud_score_actions_total", "Recommended actions", ["action"])
LATENCY = Histogram(
    "fraud_score_latency_seconds",
    "Time to compute features, score, explain and decide",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
)
SCORES = Histogram(
    "fraud_score_probability",
    "Calibrated P(fraud) of scored transactions",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0),
)


class Transaction(BaseModel):
    """A raw transaction. The named fields are required; any other IEEE-CIS field
    (C, D, M, V, identity columns) may be included and is used by the model."""

    model_config = ConfigDict(extra="allow")

    TransactionID: int
    TransactionDT: int = Field(ge=0, description="seconds since the time anchor")
    TransactionAmt: float = Field(gt=0, description="USD")
    ProductCD: str
    card1: int
    addr1: float | None = None
    D1: float | None = None
    P_emaildomain: str | None = None
    isFraud: int | None = Field(
        None,
        ge=0,
        le=1,
        description="history only: the label, if known. It counts only when the transaction "
        "is more than 30 days before the one being scored (the chargeback delay); a label on "
        "the scored transaction itself is ignored.",
    )


class ScoreRequest(BaseModel):
    transaction: Transaction
    history: list[Transaction] = Field(
        default_factory=list,
        description="earlier transactions of the same card, device or email domain",
    )


class Reason(BaseModel):
    feature: str
    reason: str
    value: float | int | str | None
    contribution: float


class ScoreResponse(BaseModel):
    TransactionID: int
    p_fraud: float
    action: str
    review_recommended: bool
    expected_cost_usd: dict[str, float]
    review_saving_usd: float
    reasons: list[Reason]
    features: dict[str, float | int | str | None]
    model: dict
    latency_ms: float


class State:
    bundle: ModelBundle | None = None
    policy: FrozenPolicy | None = None


state = State()


def load(model_dir: Path | None = None, policy_path: Path | None = None) -> None:
    model_dir = model_dir or Path(
        os.environ.get("FRAUD_MODEL_DIR", REPO_ROOT / "data" / "models" / "lightgbm")
    )
    state.bundle = ModelBundle.load(model_dir)
    state.policy = FrozenPolicy.load(policy_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if state.bundle is None:
        load()
    yield


app = FastAPI(title="Fraud scoring", version=__version__, lifespan=lifespan)


def _event(t: Transaction) -> dict:
    d = t.model_dump()
    d.setdefault("has_identity", d.get("DeviceInfo") is not None or d.get("DeviceType") is not None)
    return d


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    start = time.perf_counter()
    tx = req.transaction
    if any(h.TransactionDT > tx.TransactionDT for h in req.history):
        REQUESTS.labels("rejected").inc()
        raise HTTPException(422, "history contains transactions after the one being scored")
    online = OnlineFeatures()
    history = sorted(req.history, key=lambda h: (h.TransactionDT, h.TransactionID))
    for h in history:
        online.process(_event(h))
    for h in history:  # labels that had arrived by the time of the scored transaction
        if h.isFraud is not None and h.TransactionDT + LABEL_DELAY < tx.TransactionDT:
            online.observe_label(_event(h))
    event = _event(tx)
    event.pop("isFraud", None)
    feats = online.process(event)
    p, reasons = state.bundle.score_one({**event, **feats})
    costs = state.policy.costs
    e = costs.expected(np.array([p]), np.array([tx.TransactionAmt]))[0]
    saving = float(min(e[APPROVE], e[DECLINE]) - e[REVIEW])
    review = saving > state.policy.review_threshold and saving > 0
    action = REVIEW if review else (DECLINE if e[DECLINE] < e[APPROVE] else APPROVE)
    elapsed = time.perf_counter() - start
    REQUESTS.labels("scored").inc()
    ACTIONS.labels(ACTION_NAMES[action]).inc()
    LATENCY.observe(elapsed)
    SCORES.observe(p)
    return ScoreResponse(
        TransactionID=tx.TransactionID,
        p_fraud=p,
        action=ACTION_NAMES[action],
        review_recommended=review,
        expected_cost_usd={"approve": e[APPROVE], "decline": e[DECLINE], "review": e[REVIEW]},
        review_saving_usd=saving,
        reasons=reasons,
        features=feats,
        model={
            "name": state.bundle.name,
            "trees": state.bundle.meta.get("trees"),
            "feature_set": state.bundle.feature_set,
            "calibration": state.bundle.calibrator.method,
        },
        latency_ms=elapsed * 1e3,
    )


@app.get("/health")
def health() -> dict:
    ok = state.bundle is not None and state.policy is not None
    return {
        "status": "ok" if ok else "not ready",
        "version": __version__,
        "model": state.bundle.name if ok else None,
        "features": len(state.bundle.preprocessor.features) if ok else None,
        "review_threshold_usd": state.policy.review_threshold if ok else None,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
