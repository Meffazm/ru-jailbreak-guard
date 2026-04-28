"""Tests for evaluate + promote tasks."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

flytekit = pytest.importorskip("flytekit")

from flyte.workflows import tasks  # noqa: E402


def _task_fn(t):
    return getattr(t, "task_function", None) or t._task_function


def _fake_run(run_id: str, family: str, val_f1: float, model_version: str = "1") -> MagicMock:
    r = MagicMock()
    r.info.run_id = run_id
    r.data.tags = {"model_family": family, "data_version": "vTEST"}
    r.data.metrics = {"val_f1": val_f1}
    r.data.params = {"model_version": model_version}
    return r


def test_evaluate_picks_best_per_family() -> None:
    client = MagicMock()
    client.search_runs.return_value = [
        _fake_run("a", "tfidf_logreg", 0.90),
        _fake_run("b", "tfidf_logreg", 0.93),  # best tfidf
        _fake_run("c", "lgbm_emb", 0.97),       # best lgbm
        _fake_run("d", "rubert_ft", 0.99),      # best rubert + champion
    ]

    with patch("flyte.workflows.tasks._mlflow_client", return_value=client):
        result = _task_fn(tasks.evaluate)(data_version="vTEST")

    assert result["per_family"]["tfidf_logreg"]["run_id"] == "b"
    assert result["per_family"]["lgbm_emb"]["run_id"] == "c"
    assert result["per_family"]["rubert_ft"]["run_id"] == "d"
    assert result["champion"]["run_id"] == "d"
    assert result["champion"]["family"] == "rubert_ft"


def test_evaluate_raises_when_no_runs() -> None:
    client = MagicMock()
    client.search_runs.return_value = []

    with patch("flyte.workflows.tasks._mlflow_client", return_value=client):
        with pytest.raises(RuntimeError, match="No successful runs"):
            _task_fn(tasks.evaluate)(data_version="vTEST")
