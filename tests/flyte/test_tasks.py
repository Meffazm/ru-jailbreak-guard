"""Unit tests for flyte/workflows/tasks.py — call task bodies directly."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Importing flyte.workflows.tasks requires flytekit (the `flyte` dep group).
flytekit = pytest.importorskip("flytekit")

from flyte.workflows import tasks  # noqa: E402


def _task_fn(task_obj):
    """Extract the underlying Python function from a Flyte @task object."""
    fn = getattr(task_obj, "task_function", None) or getattr(task_obj, "_task_function", None)
    if fn is None:
        raise AttributeError(f"Cannot find underlying function on {task_obj}")
    return fn


def test_download_splits_pulls_three_parquets(tmp_path: Path) -> None:
    s3_mock = MagicMock()

    def _download_file(Bucket, Key, Filename):  # noqa: N803  # mirror boto3 PascalCase kwargs
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_bytes(b"fake-parquet")

    s3_mock.download_file.side_effect = _download_file

    with patch.object(tasks, "_make_s3_client", return_value=s3_mock):
        result_dir = _task_fn(tasks.download_splits)(
            data_version="v123abc456",
            target_dir=str(tmp_path),
            bucket="splits",
        )

    assert Path(result_dir).is_dir()
    assert (Path(result_dir) / "train.parquet").exists()
    assert (Path(result_dir) / "val.parquet").exists()
    assert (Path(result_dir) / "test.parquet").exists()
    assert s3_mock.download_file.call_count == 3
