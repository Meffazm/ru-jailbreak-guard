"""Tests for the predictor input sampling buffer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ru_jailbreak_guard.serve.sampling import (
    SamplingBuffer,
    cyrillic_aware_record,
)


def test_sampling_buffer_flushes_when_full() -> None:
    s3 = MagicMock()
    buf = SamplingBuffer(family="tfidf_logreg", flush_size=3, s3_client=s3, sample_rate=1.0)

    for i in range(5):
        buf.maybe_append(text=f"text{i}", label="benign", confidence=0.5)

    # First flush at 3 → 1 put_object, 2 still buffered
    assert s3.put_object.call_count == 1


def test_sampling_buffer_flush_groups_by_date() -> None:
    s3 = MagicMock()
    buf = SamplingBuffer(family="lgbm_emb", flush_size=2, s3_client=s3, sample_rate=1.0)
    buf.maybe_append(text="a", label="benign", confidence=0.5)
    buf.maybe_append(text="b", label="jailbreak", confidence=0.9)

    call = s3.put_object.call_args
    key = call.kwargs["Key"]
    assert key.startswith(buf.date_prefix)  # YYYY-MM-DD
    assert "/lgbm_emb/" in key
    assert key.endswith(".parquet")


def test_cyrillic_aware_record_shape() -> None:
    rec = cyrillic_aware_record(text="привет", label="benign", confidence=0.7)
    assert rec["text"] == "привет"
    assert rec["label"] == "benign"
    assert rec["confidence"] == 0.7
    assert "ts" in rec


def test_sample_rate_zero_never_appends() -> None:
    s3 = MagicMock()
    buf = SamplingBuffer(family="rubert_ft", flush_size=10, s3_client=s3, sample_rate=0.0)
    for _ in range(50):
        buf.maybe_append(text="x", label="benign", confidence=0.5)
    assert s3.put_object.call_count == 0


def test_predictor_predict_appends_to_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from ru_jailbreak_guard.serve import predictor as predictor_module
    from ru_jailbreak_guard.serve.predictor import Predictor, build_app

    appended: list[dict] = []

    class _FakeBuffer:
        def maybe_append(self, *, text: str, label: str, confidence: float) -> None:
            appended.append({"text": text, "label": label, "confidence": confidence})

    monkeypatch.setattr(predictor_module, "_get_buffer", lambda family: _FakeBuffer())

    class _FakePredictor(Predictor):
        family = "rubert_ft"

        def load(self) -> None:
            pass

        def _predict_one(self, text: str) -> tuple[str, float]:
            return ("benign", 0.7)

    p = _FakePredictor(version="1", data_version="x" * 12)
    client = TestClient(build_app(predictor=p))
    response = client.post("/predict", json={"text": "тест"})

    assert response.status_code == 200
    assert len(appended) == 1
    assert appended[0]["label"] == "benign"
