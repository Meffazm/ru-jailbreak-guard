"""Common evaluation metrics for binary jailbreak classifiers."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute binary classification metrics.

    Args:
        y_true: Ground-truth 0/1 labels, shape (N,).
        y_pred: Predicted 0/1 labels, shape (N,).
        y_score: Predicted probabilities for class 1, shape (N,).

    Returns:
        Dict with keys: f1, precision, recall, auroc, accuracy.
    """
    return {
        "f1": float(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0.0)),
        "precision": float(precision_score(y_true=y_true, y_pred=y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true=y_true, y_pred=y_pred, zero_division=0.0)),
        "auroc": float(roc_auc_score(y_true=y_true, y_score=y_score)),
        "accuracy": float(accuracy_score(y_true=y_true, y_pred=y_pred)),
    }


def per_source_f1(*, df: pl.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    """Compute F1 per `source` column value.

    Args:
        df: DataFrame with `label` and `source` columns. Order matches `y_pred`.
        y_pred: Predicted 0/1 labels in the same row order as df.

    Returns:
        Dict mapping source string to F1 score.
    """
    out: dict[str, float] = {}
    df_with_pred = df.with_columns(pl.Series("y_pred", y_pred))
    for source in df_with_pred["source"].unique().to_list():
        sub = df_with_pred.filter(pl.col("source") == source)
        y_true = sub["label"].to_numpy()
        y_p = sub["y_pred"].to_numpy()
        out[source] = float(f1_score(y_true=y_true, y_pred=y_p, zero_division=0.0))
    return out


def confusion_matrix_png(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str = "Confusion matrix",
) -> None:
    """Write a confusion matrix as a PNG to `out_path`."""
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["benign", "jailbreak"])
    ax.set_yticks([0, 1], ["benign", "jailbreak"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(int(val)), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def latency_benchmark(
    *,
    predict_fn: Callable[[list[str]], list[int]],
    sample_texts: list[str],
    batch_size: int = 1,
) -> dict[str, float]:
    """Measure per-request latency in ms over `sample_texts`.

    Args:
        predict_fn: Callable that takes a list of texts and returns a list of predictions.
        sample_texts: Texts to time. Each batch of `batch_size` is one timed call.
        batch_size: How many texts per call. Per-call latency is divided by batch_size
            to give per-text ms (so callers can compare batch and single-call modes).

    Returns:
        Dict with p50_ms, p95_ms, p99_ms, mean_ms (per request).
    """
    timings_ms: list[float] = []
    for start in range(0, len(sample_texts), batch_size):
        batch = sample_texts[start : start + batch_size]
        if not batch:
            continue
        t0 = time.perf_counter()
        predict_fn(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(elapsed_ms / len(batch))
    arr = np.array(timings_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }
