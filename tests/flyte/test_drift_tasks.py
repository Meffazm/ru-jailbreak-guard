"""Tests for drift detection Flyte tasks."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

flytekit = pytest.importorskip("flytekit")

from flyte.workflows import drift as drift_module  # noqa: E402


def _task_fn(t):
    return getattr(t, "task_function", None) or t._task_function


def test_compute_drift_task_skips_when_insufficient() -> None:
    with (
        patch.object(drift_module, "_read_production_texts", return_value=["a"] * 50),
        patch.object(drift_module, "_read_training_texts", return_value=["a"] * 100),
    ):
        result = _task_fn(drift_module.compute_drift_task)(
            data_version="vTEST",
            family="tfidf_logreg",
        )
    assert result["drift_detected"] is False
    assert result.get("skipped_reason") == "insufficient_samples"


def test_compute_drift_task_returns_full_dict() -> None:
    with (
        patch.object(drift_module, "_read_production_texts", return_value=["a"] * 100),
        patch.object(drift_module, "_read_training_texts", return_value=["a"] * 100),
        patch.object(drift_module, "_embed_texts", return_value=np.zeros((100, 4))),
    ):
        result = _task_fn(drift_module.compute_drift_task)(
            data_version="vTEST",
            family="tfidf_logreg",
        )
    assert "drift_detected" in result
    assert "p_value_length" in result
    assert "p_value_charset" in result
    assert "cosine_distance" in result
    assert isinstance(result["drift_detected"], bool)


def test_push_drift_metric_handles_unreachable_gateway() -> None:
    """The push should tolerate Pushgateway being unreachable (logs and continues)."""
    with patch("flyte.workflows.drift.push_to_gateway", side_effect=Exception("conn refused")):
        # Should not raise
        _task_fn(drift_module.push_drift_metric)(
            report={"drift_detected": True}, family="tfidf_logreg"
        )
