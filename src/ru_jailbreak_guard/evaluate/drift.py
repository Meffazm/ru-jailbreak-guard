"""Drift detection: KS-test on length, chi-square on charset, cosine on embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

_LENGTH_THRESHOLD = 0.01
_CHARSET_THRESHOLD = 0.01
_COSINE_THRESHOLD = 0.15


@dataclass
class DriftReport:
    p_value_length: float
    p_value_charset: float
    cosine_distance: float
    drift_detected: bool
    n_production: int
    n_training: int


def _cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    cy = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return cy / len(text)


def ks_test_length(prod: list[str], train: list[str]) -> float:
    """Two-sample KS test on text-length distributions. Returns p-value."""
    if len(prod) < 5 or len(train) < 5:
        return 1.0
    p_lens = [len(t) for t in prod]
    t_lens = [len(t) for t in train]
    res = stats.ks_2samp(p_lens, t_lens)
    return float(res.pvalue)


def chi_square_charset(prod: list[str], train: list[str]) -> float:
    """Chi-square test on cyrillic-ratio quartile distribution. Returns p-value."""
    if len(prod) < 5 or len(train) < 5:
        return 1.0
    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    p_hist, _ = np.histogram([_cyrillic_ratio(t) for t in prod], bins=bins)
    t_hist, _ = np.histogram([_cyrillic_ratio(t) for t in train], bins=bins)
    if t_hist.sum() == 0:
        return 1.0
    expected = t_hist * (p_hist.sum() / t_hist.sum())
    expected = np.where(expected < 1, 1, expected)
    chi2 = float(((p_hist - expected) ** 2 / expected).sum())
    df = max(len(p_hist) - 1, 1)
    return float(1.0 - stats.chi2.cdf(chi2, df))


def cosine_distance_centroids(prod_embeddings: np.ndarray, train_embeddings: np.ndarray) -> float:
    """Cosine distance between centroids of two embedding sets. 0=identical, 1=orthogonal."""
    if len(prod_embeddings) == 0 or len(train_embeddings) == 0:
        return 0.0
    c_p = prod_embeddings.mean(axis=0)
    c_t = train_embeddings.mean(axis=0)
    n_p = np.linalg.norm(c_p)
    n_t = np.linalg.norm(c_t)
    if n_p == 0 or n_t == 0:
        return 0.0
    cos_sim = float(np.dot(c_p, c_t) / (n_p * n_t))
    return 1.0 - cos_sim


def compute_drift(
    *,
    production_texts: list[str],
    training_texts: list[str],
    production_embeddings: np.ndarray | None = None,
    training_embeddings: np.ndarray | None = None,
) -> DriftReport:
    p_len = ks_test_length(production_texts, training_texts)
    p_chr = chi_square_charset(production_texts, training_texts)
    cos = (
        cosine_distance_centroids(production_embeddings, training_embeddings)
        if production_embeddings is not None and training_embeddings is not None
        else 0.0
    )
    detected = p_len < _LENGTH_THRESHOLD or p_chr < _CHARSET_THRESHOLD or cos > _COSINE_THRESHOLD
    return DriftReport(
        p_value_length=p_len,
        p_value_charset=p_chr,
        cosine_distance=cos,
        drift_detected=detected,
        n_production=len(production_texts),
        n_training=len(training_texts),
    )
