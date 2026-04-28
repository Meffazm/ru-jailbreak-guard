"""Predictor ABC + FastAPI app builder + locked response schema.

Concrete predictors subclass `Predictor` and implement `load()` and
`_predict_one()`. The HTTP layer is shared.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Literal

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from ru_jailbreak_guard.serve.metrics import (
    observe_prediction,
    set_model_version_info,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    label: Literal["jailbreak", "benign"]
    confidence: float
    model_family: Literal["tfidf_logreg", "lgbm_emb", "rubert_ft"]
    model_version: str
    data_version: str
    latency_ms: float


class Predictor(ABC):
    """Subclasses must override `family`, `load`, and `_predict_one`."""

    family: Literal["tfidf_logreg", "lgbm_emb", "rubert_ft"]

    def __init__(self, *, version: str, data_version: str) -> None:
        self.version = version
        self.data_version = data_version

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def _predict_one(self, text: str) -> tuple[str, float]:
        """Return (label, confidence). Label is 'jailbreak' or 'benign'."""

    def predict(self, text: str) -> PredictionResponse:
        t0 = time.perf_counter()
        label, confidence = self._predict_one(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        observe_prediction(
            family=self.family,
            label=label,  # ty: ignore[invalid-argument-type]
            confidence=float(confidence),
            latency_seconds=latency_ms / 1000.0,
            text=text,
        )
        return PredictionResponse(
            label=label,  # ty: ignore[invalid-argument-type]
            confidence=float(confidence),
            model_family=self.family,
            model_version=self.version,
            data_version=self.data_version,
            latency_ms=latency_ms,
        )


def build_app(*, predictor: Predictor) -> FastAPI:
    """Build a FastAPI app exposing /health, /predict, and /metrics."""
    app = FastAPI(title=f"ru-jailbreak-{predictor.family}")
    predictor.load()
    set_model_version_info(
        family=predictor.family,
        version=predictor.version,
        data_version=predictor.data_version,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictRequest) -> PredictionResponse:
        return predictor.predict(req.text)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
