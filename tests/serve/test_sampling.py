"""Tests for the predictor input sampling buffer."""

from __future__ import annotations

from unittest.mock import MagicMock

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
