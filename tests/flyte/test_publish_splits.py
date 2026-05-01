"""Tests for scripts/publish_splits.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# scripts/ is not a package; import via path manipulation
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import publish_splits  # noqa: E402  # ty: ignore[unresolved-import]


def test_data_version_from_dvc_lock(tmp_path: Path) -> None:
    """Reads md5 of split stage outs from a dvc.lock fixture."""
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        """schema: '2.0'
stages:
  split:
    cmd: foo
    outs:
      - path: data/splits/train.parquet
        md5: aaaa
      - path: data/splits/val.parquet
        md5: bbbb
      - path: data/splits/test.parquet
        md5: cccc
"""
    )
    version = publish_splits.compute_data_version(lock)
    # Version is deterministic hash of the split stage outs
    assert isinstance(version, str)
    assert len(version) == 12  # short hash
    # Same input → same version
    assert publish_splits.compute_data_version(lock) == version


def test_publish_splits_uploads_three_parquets(tmp_path: Path) -> None:
    """Each of train/val/test parquet is uploaded to s3://splits/<version>/."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name in ("train", "val", "test"):
        (splits_dir / f"{name}.parquet").write_bytes(b"fake parquet")

    s3 = MagicMock()
    publish_splits.upload(
        s3_client=s3,
        bucket="splits",
        data_version="v123abc456",
        splits_dir=splits_dir,
    )
    # Three put_object calls, one per file, with correct keys
    assert s3.put_object.call_count == 3
    keys = sorted(c.kwargs["Key"] for c in s3.put_object.call_args_list)
    assert keys == [
        "v123abc456/test.parquet",
        "v123abc456/train.parquet",
        "v123abc456/val.parquet",
    ]


def test_publish_splits_idempotent(tmp_path: Path) -> None:
    """If objects already exist with same ETag, skip upload."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.parquet").write_bytes(b"fake")
    (splits_dir / "val.parquet").write_bytes(b"fake")
    (splits_dir / "test.parquet").write_bytes(b"fake")

    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": 4}  # same size = "exists"
    publish_splits.upload(
        s3_client=s3,
        bucket="splits",
        data_version="v123",
        splits_dir=splits_dir,
        skip_if_exists=True,
    )
    assert s3.put_object.call_count == 0


def test_main_missing_dvc_lock_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing dvc.lock → exits with code 1."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        publish_splits.main(argv=[])
    assert exc.value.code == 1
