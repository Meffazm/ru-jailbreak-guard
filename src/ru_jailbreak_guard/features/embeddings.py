"""ruBERT embedding extractor with on-disk numpy cache.

Cache key convention: `{data_version}__{backbone_id}__{revision}`. Stored as `.npy`
under the cache root. Used by Task 6 (LightGBM trainer) to avoid recomputing
embeddings across runs with the same data version.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class _SupportsEncode(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class Encoder:
    """ruBERT-tiny2 mean-pooled encoder.

    Mean pooling over non-pad tokens — standard for sentence-level embeddings of
    BERT-family models without a [CLS]-style readout.
    """

    def __init__(
        self,
        *,
        model_name: str = "cointegrated/rubert-tiny2",
        revision: str | None = None,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModel.from_pretrained(model_name, revision=revision).to(device)
        self._model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode `texts` to mean-pooled embeddings.

        Returns:
            float32 array of shape (N, hidden_size).
        """
        with torch.no_grad():
            tokens = self._tokenizer(  # ty: ignore[call-non-callable]
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self._model(**tokens)
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            mean_pooled = summed / counts
        return mean_pooled.cpu().numpy().astype(np.float32)


class EmbeddingCache:
    """Tiny content-addressed numpy cache. Keys are filesystem-safe strings."""

    def __init__(self, *, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Sanitize: replace path separators so model names like
        # `cointegrated/rubert-tiny2` don't create directory components.
        safe = key.replace("/", "_").replace(os.sep, "_")
        return self.root / f"{safe}.npy"

    def get(self, *, key: str) -> np.ndarray | None:
        path = self._path(key)
        if not path.exists():
            return None
        return np.load(path)

    def put(self, *, key: str, arr: np.ndarray) -> None:
        np.save(self._path(key), arr)


def embed_texts_cached(
    *,
    texts: list[str],
    encoder: _SupportsEncode,
    cache: EmbeddingCache,
    cache_key: str,
) -> np.ndarray:
    """Return embeddings for `texts`. Hits cache if `cache_key` was seen before.

    Caching is keyed only on `cache_key`; callers must pick a key that captures
    everything that affects the embeddings (data_version, model_name, revision).
    """
    cached = cache.get(key=cache_key)
    if cached is not None:
        return cached
    arr = encoder.encode(texts)
    cache.put(key=cache_key, arr=arr)
    return arr
