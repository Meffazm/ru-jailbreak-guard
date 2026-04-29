"""Tests for drift detection statistical tests."""

from __future__ import annotations

import numpy as np

from ru_jailbreak_guard.evaluate.drift import (
    DriftReport,
    chi_square_charset,
    compute_drift,
    cosine_distance_centroids,
    ks_test_length,
)


def test_ks_test_identical_distributions_no_drift() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(100, 10, size=500).tolist()
    b = rng.normal(100, 10, size=500).tolist()
    p = ks_test_length(
        ["x" * int(x) for x in a],
        ["x" * int(x) for x in b],
    )
    assert p > 0.01


def test_ks_test_shifted_distribution_detects_drift() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(100, 10, size=500).tolist()
    b = rng.normal(200, 10, size=500).tolist()
    p = ks_test_length(
        ["x" * int(x) for x in a],
        ["x" * int(x) for x in b],
    )
    assert p < 0.001


def test_chi_square_charset_balanced_no_drift() -> None:
    a = ["привет"] * 100
    b = ["привет"] * 100
    p = chi_square_charset(a, b)
    assert p > 0.01


def test_chi_square_charset_skew_detects_drift() -> None:
    a = ["привет"] * 100
    b = ["hello"] * 100
    p = chi_square_charset(a, b)
    assert p < 0.01


def test_cosine_distance_identical_zero() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert cosine_distance_centroids(a, b) < 0.01


def test_cosine_distance_orthogonal_one() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert abs(cosine_distance_centroids(a, b) - 1.0) < 0.01


def test_compute_drift_aggregates_signals() -> None:
    a = ["привет"] * 100
    b = ["hello"] * 100
    embs_a = np.random.default_rng(1).normal(size=(100, 4))
    embs_b = np.random.default_rng(2).normal(size=(100, 4)) + 5.0

    report = compute_drift(
        production_texts=b,
        training_texts=a,
        production_embeddings=embs_b,
        training_embeddings=embs_a,
    )
    assert isinstance(report, DriftReport)
    assert report.drift_detected
