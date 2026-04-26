"""Tests for ruBERT embedding extractor."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ru_jailbreak_guard.features.embeddings import (
    EmbeddingCache,
    Encoder,
    embed_texts_cached,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.mark.slow
def test_encoder_returns_2d_array_shape() -> None:
    """Hits real model — slow, but only once. Marked slow."""
    enc = Encoder(model_name="cointegrated/rubert-tiny2", device="cpu")
    out = enc.encode(["привет", "тест"])
    assert out.shape == (2, 312)  # rubert-tiny2 hidden size = 312
    assert out.dtype == np.float32


def test_embedding_cache_writes_and_reads(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    key = "data-v1__model-x__rev-abc"
    arr = np.random.rand(5, 8).astype(np.float32)
    cache.put(key=key, arr=arr)
    loaded = cache.get(key=key)
    assert loaded is not None
    assert np.array_equal(loaded, arr)


def test_embedding_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    assert cache.get(key="missing") is None


def test_embed_texts_cached_uses_cache_on_second_call(tmp_path: Path) -> None:
    fake_encoder = MagicMock()
    fake_encoder.encode.return_value = np.ones((2, 4), dtype=np.float32)
    cache = EmbeddingCache(root=tmp_path)
    out1 = embed_texts_cached(
        texts=["a", "b"],
        encoder=fake_encoder,
        cache=cache,
        cache_key="k1",
    )
    out2 = embed_texts_cached(
        texts=["a", "b"],
        encoder=fake_encoder,
        cache=cache,
        cache_key="k1",
    )
    assert np.array_equal(out1, out2)
    fake_encoder.encode.assert_called_once()


def test_embed_texts_cached_recomputes_on_different_key(tmp_path: Path) -> None:
    fake_encoder = MagicMock()
    fake_encoder.encode.return_value = np.ones((2, 4), dtype=np.float32)
    cache = EmbeddingCache(root=tmp_path)
    embed_texts_cached(texts=["a", "b"], encoder=fake_encoder, cache=cache, cache_key="k1")
    embed_texts_cached(texts=["a", "b"], encoder=fake_encoder, cache=cache, cache_key="k2")
    assert fake_encoder.encode.call_count == 2
