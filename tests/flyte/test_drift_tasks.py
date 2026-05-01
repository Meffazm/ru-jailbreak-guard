"""Tests for drift detection Flyte tasks."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

flytekit = pytest.importorskip("flytekit")

from flyte.workflows import drift as drift_module  # noqa: E402


def _task_fn(t):
    return getattr(t, "task_function", None) or t._task_function


def test_fetch_production_samples_empty_returns_empty(tmp_path: Path) -> None:
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {}
    with patch.object(drift_module, "_make_s3_client", return_value=s3):
        texts = _task_fn(drift_module.fetch_production_samples)(
            family="tfidf_logreg",
            days=7,
        )
    assert texts == []


def test_compute_drift_task_skips_when_insufficient() -> None:
    result = _task_fn(drift_module.compute_drift_task)(
        production_texts=["a"] * 50,  # < 100 threshold
        training_texts=["a"] * 100,
        family="tfidf_logreg",
    )
    assert result["drift_detected"] is False
    assert result.get("skipped_reason") == "insufficient_samples"


def test_compute_drift_task_returns_full_dict() -> None:
    with patch("flyte.workflows.drift._embed_texts") as embed:
        embed.return_value = np.zeros((100, 4))
        result = _task_fn(drift_module.compute_drift_task)(
            production_texts=["a"] * 100,
            training_texts=["a"] * 100,
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
