"""Slow integration test: cheap_train_pipeline composition wired to real @task fns.

Marked @pytest.mark.slow — excluded from `make test` (default CI).
Run with: `uv run --group flyte pytest -m slow tests/flyte/test_workflows_local.py`.

Why this is needed: `tests/flyte/test_tasks.py` only verifies each @task body in
isolation. This test exercises the @workflow itself (`cheap_train_pipeline`) so a
broken composition (wrong arg name, missing edge, return-shape mismatch) is
caught locally without spinning up a Flyte cluster.

Implementation note: calling `cheap_train_pipeline(...)` directly triggers
Flytekit's compile-time path, which yields `Promise` objects that can't be
serialized into a Python dict (see "can not serialize 'Promise' object"). We
therefore call the underlying `__wrapped__` function with the @task primitives
patched at the `flyte.workflows.pipelines` module-level so each task call is a
plain Python function returning a real dict.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

flytekit = pytest.importorskip("flytekit")


@pytest.mark.slow
def test_cheap_pipeline_local(tmp_path: Path) -> None:
    """Run cheap_train_pipeline body locally with both train @task primitives mocked."""
    from flyte.workflows import pipelines

    fake_tfidf = {
        "run_id": "tfidf-1",
        "val_metrics": {"f1": 0.9},
        "test_metrics": {"f1": 0.91},
        "model_family": "tfidf_logreg",
    }
    fake_lgbm = {
        "run_id": "lgbm-1",
        "val_metrics": {"f1": 0.95},
        "test_metrics": {"f1": 0.96},
        "model_family": "lgbm_emb",
    }

    tfidf_calls: list[dict[str, Any]] = []
    lgbm_calls: list[dict[str, Any]] = []

    def fake_train_tfidf(data_version: str) -> dict:
        tfidf_calls.append({"data_version": data_version})
        return fake_tfidf

    def fake_train_lgbm(data_version: str) -> dict:
        lgbm_calls.append({"data_version": data_version})
        return fake_lgbm

    with (
        patch.object(pipelines, "train_tfidf", fake_train_tfidf),
        patch.object(pipelines, "train_lgbm", fake_train_lgbm),
    ):
        # __wrapped__ is the raw Python function under @workflow — bypasses
        # Flytekit's Promise/compile machinery so the body runs as plain Python.
        wrapped = pipelines.cheap_train_pipeline.__wrapped__
        result = wrapped(data_version="vTEST")

    assert result == (fake_tfidf, fake_lgbm)
    assert tfidf_calls == [{"data_version": "vTEST"}]
    assert lgbm_calls == [{"data_version": "vTEST"}]
