"""Tests for predictor base + a fake concrete predictor."""

import time

from fastapi.testclient import TestClient

from ru_jailbreak_guard.serve.predictor import PredictionResponse, Predictor, build_app


class _FakePredictor(Predictor):
    family = "tfidf_logreg"

    def __init__(self, *, version: str = "1", data_version: str = "abc") -> None:
        super().__init__(version=version, data_version=data_version)
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def _predict_one(self, text: str) -> tuple[str, float]:
        time.sleep(0.001)
        # Trivial rule: contains "взлом" → jailbreak
        if "взлом" in text.lower():
            return "jailbreak", 0.95
        return "benign", 0.05


def test_health_endpoint() -> None:
    app = build_app(predictor=_FakePredictor())
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_locked_schema() -> None:
    app = build_app(predictor=_FakePredictor(version="7", data_version="dv-xyz"))
    client = TestClient(app)
    resp = client.post("/predict", json={"text": "взлом сейчас"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "jailbreak"
    assert body["confidence"] >= 0.5
    assert body["model_family"] == "tfidf_logreg"
    assert body["model_version"] == "7"
    assert body["data_version"] == "dv-xyz"
    assert body["latency_ms"] >= 0


def test_predict_benign() -> None:
    app = build_app(predictor=_FakePredictor())
    client = TestClient(app)
    resp = client.post("/predict", json={"text": "опиши погоду в москве"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "benign"


def test_response_model_is_PredictionResponse_compatible() -> None:  # noqa: N802
    pr = PredictionResponse(
        label="jailbreak",
        confidence=0.9,
        model_family="tfidf_logreg",
        model_version="1",
        data_version="dv",
        latency_ms=2.5,
    )
    assert pr.label == "jailbreak"
    assert pr.confidence == 0.9
