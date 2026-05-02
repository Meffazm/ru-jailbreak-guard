"""Unit tests for flyte/workflows/tasks.py - call task bodies directly."""

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


def _stub_split_download(splits_dir: Path):
    """Patch _download_splits to return a pre-populated splits dir without S3."""
    splits_dir.mkdir(exist_ok=True)
    for n in ("train", "val", "test"):
        (splits_dir / f"{n}.parquet").write_bytes(b"")
    return patch.object(tasks, "_download_splits", return_value=splits_dir)


def test_download_splits_pulls_three_parquets(tmp_path: Path) -> None:
    s3_mock = MagicMock()

    def _download_file(Bucket, Key, Filename):  # noqa: N803  # mirror boto3 PascalCase kwargs
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_bytes(b"fake-parquet")

    s3_mock.download_file.side_effect = _download_file

    with patch.object(tasks, "_make_s3_client", return_value=s3_mock):
        result_dir = tasks._download_splits(
            data_version="v123abc456",
            target_dir=str(tmp_path),
        )

    assert Path(result_dir).is_dir()
    assert (Path(result_dir) / "train.parquet").exists()
    assert (Path(result_dir) / "val.parquet").exists()
    assert (Path(result_dir) / "test.parquet").exists()
    assert s3_mock.download_file.call_count == 3


def test_train_tfidf_invokes_underlying_trainer(tmp_path: Path) -> None:
    """The @task wrapper downloads splits + builds TrainConfig + calls trainer."""
    splits_dir = tmp_path / "splits"

    with (
        _stub_split_download(splits_dir),
        patch("ru_jailbreak_guard.models.tfidf_logreg.train_tfidf_logreg") as m,
    ):
        m.return_value = {"run_id": "abc", "val_metrics": {"f1": 0.9}, "test_metrics": {"f1": 0.91}}
        result = _task_fn(tasks.train_tfidf)(data_version="v123")
    assert result["run_id"] == "abc"
    assert result["model_family"] == "tfidf_logreg"
    m.assert_called_once()
    cfg = m.call_args.kwargs["cfg"]
    assert str(cfg.train_path) == str(splits_dir / "train.parquet")
    assert cfg.data_version == "v123"


def test_train_lgbm_invokes_underlying_trainer(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"

    with (
        _stub_split_download(splits_dir),
        patch("ru_jailbreak_guard.models.lgbm_emb.train_lgbm_emb") as m,
    ):
        m.return_value = {
            "run_id": "xyz",
            "val_metrics": {"f1": 0.95},
            "test_metrics": {"f1": 0.96},
        }
        result = _task_fn(tasks.train_lgbm)(data_version="v123")
    assert result["model_family"] == "lgbm_emb"
    m.assert_called_once()


def test_train_rubert_invokes_underlying_trainer(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"

    with (
        _stub_split_download(splits_dir),
        patch("ru_jailbreak_guard.models.rubert_ft.train_rubert_ft") as m,
    ):
        m.return_value = {
            "run_id": "rrr",
            "val_metrics": {"f1": 0.99},
            "test_metrics": {"f1": 0.99},
        }
        result = _task_fn(tasks.train_rubert)(data_version="v123")
    assert result["model_family"] == "rubert_ft"
    m.assert_called_once()
