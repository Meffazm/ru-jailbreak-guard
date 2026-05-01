"""Tests for metrics module."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ru_jailbreak_guard.evaluate.metrics import (
    binary_metrics,
    confusion_matrix_png,
    latency_benchmark,
    per_source_f1,
)


def test_binary_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.95])
    metrics = binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_binary_metrics_all_wrong() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.05])
    metrics = binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["auroc"] == pytest.approx(0.0)


def test_per_source_f1_returns_dict_per_source() -> None:
    df = pl.DataFrame(
        {
            "label": [1, 1, 0, 0, 1, 0],
            "source": ["a", "a", "a", "b", "b", "b"],
        }
    )
    y_pred = np.array([1, 0, 0, 0, 1, 1])  # source a F1=0.667, source b F1=0.667
    result = per_source_f1(df=df, y_pred=y_pred)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"] == pytest.approx(2 / 3, rel=1e-3)
    assert result["b"] == pytest.approx(2 / 3, rel=1e-3)


def test_confusion_matrix_png_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "cm.png"
    confusion_matrix_png(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        out_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_latency_benchmark_returns_p50_p95_p99() -> None:
    def mock_predict(texts: list[str]) -> list[int]:
        return [0] * len(texts)

    result = latency_benchmark(predict_fn=mock_predict, sample_texts=["x"] * 50, batch_size=1)
    assert "p50_ms" in result
    assert "p95_ms" in result
    assert "p99_ms" in result
    assert result["p50_ms"] >= 0
    assert result["p95_ms"] >= result["p50_ms"]
    assert result["p99_ms"] >= result["p95_ms"]
