"""Prometheus metric definitions + emission helpers for predictors."""

from __future__ import annotations

from typing import Literal

from prometheus_client import Counter, Gauge, Histogram

# Bucket boundaries for confidence — centered around the [0,1] range
_CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
# Inference latency in seconds — log-spaced
_LATENCY_BUCKETS = (0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5)
# Token count proxy buckets
_TOKEN_BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
# Cyrillic ratio buckets
_CYRILLIC_BUCKETS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0)


prediction_label_total = Counter(
    "ru_jailbreak_prediction_label_total",
    "Count of predictions by family and label.",
    ("family", "label"),
)

prediction_confidence = Histogram(
    "ru_jailbreak_prediction_confidence",
    "Distribution of prediction confidences.",
    ("family",),
    buckets=_CONFIDENCE_BUCKETS,
)

inference_latency_seconds = Histogram(
    "ru_jailbreak_inference_latency_seconds",
    "Predictor end-to-end inference latency.",
    ("family",),
    buckets=_LATENCY_BUCKETS,
)

tokens_per_request = Histogram(
    "ru_jailbreak_tokens_per_request",
    "Approximate token count of the input text (chars/4).",
    ("family",),
    buckets=_TOKEN_BUCKETS,
)

cyrillic_ratio_hist = Histogram(
    "ru_jailbreak_cyrillic_ratio",
    "Fraction of input characters that are Cyrillic.",
    ("family",),
    buckets=_CYRILLIC_BUCKETS,
)

model_version_info = Gauge(
    "ru_jailbreak_model_version_info",
    "Currently deployed model version (value=1; labels carry version metadata).",
    ("family", "version", "data_version"),
)


def cyrillic_ratio(text: str) -> float:
    """Return fraction of chars in `text` that are Cyrillic. 0.0 if empty."""
    if not text:
        return 0.0
    cyrillic = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return cyrillic / len(text)


def observe_prediction(
    *,
    family: Literal["tfidf_logreg", "lgbm_emb", "rubert_ft"],
    label: Literal["jailbreak", "benign"],
    confidence: float,
    latency_seconds: float,
    text: str,
) -> None:
    """Emit all per-prediction metrics in a single call."""
    prediction_label_total.labels(family=family, label=label).inc()
    prediction_confidence.labels(family=family).observe(confidence)
    inference_latency_seconds.labels(family=family).observe(latency_seconds)
    tokens_per_request.labels(family=family).observe(len(text) / 4)
    cyrillic_ratio_hist.labels(family=family).observe(cyrillic_ratio(text))


def set_model_version_info(
    *,
    family: Literal["tfidf_logreg", "lgbm_emb", "rubert_ft"],
    version: str,
    data_version: str,
) -> None:
    """Set the model_version_info gauge for a freshly loaded predictor."""
    model_version_info.labels(family=family, version=version, data_version=data_version).set(1.0)
