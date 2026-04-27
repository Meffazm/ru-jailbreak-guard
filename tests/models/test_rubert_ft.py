"""Tests for ruBERT-tiny2 fine-tune trainer.

The actual training is slow (downloads + 3 epochs); marked `slow`. A small
unit covers TrainConfig defaults + the lazy dataset class to keep CI fast.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from ru_jailbreak_guard.models.rubert_ft import TrainConfig, _TextDataset


def test_train_config_defaults() -> None:
    cfg = TrainConfig(
        train_path=Path("/tmp/t.parquet"),
        val_path=Path("/tmp/v.parquet"),
        test_path=Path("/tmp/te.parquet"),
        artifacts_dir=Path("/tmp/a"),
    )
    assert cfg.backbone == "cointegrated/rubert-tiny2"
    assert cfg.num_epochs == 3
    assert cfg.batch_size == 16
    assert cfg.register_as == "ru-jailbreak-rubert-ft"


def test_text_dataset_yields_tokenized_dict() -> None:
    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {
        "input_ids": MagicMock(squeeze=lambda dim: f"ids_{dim}"),
        "attention_mask": MagicMock(squeeze=lambda dim: f"mask_{dim}"),
    }
    ds = _TextDataset(
        texts=["hello", "world"],
        labels=np.array([0, 1]),
        tokenizer=fake_tokenizer,
        max_length=32,
    )
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


@pytest.mark.slow
def test_real_train_minimal_smoke(tmp_path: Path) -> None:
    """End-to-end smoke: 1 epoch, 16 examples. Hits real ruBERT-tiny2."""
    from ru_jailbreak_guard.models.rubert_ft import train_rubert_ft

    rows = []
    for i in range(16):
        rows.append(
            {
                "text": f"jailbreak attempt {i}" if i % 2 else f"benign news {i}",
                "label": i % 2,
                "source": "test",
                "subcategory": None,
                "lang": "ru",
            }
        )
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name, rs in [("train", rows[:8]), ("val", rows[8:12]), ("test", rows[12:])]:
        pl.DataFrame(rs).write_parquet(splits_dir / f"{name}.parquet")

    cfg = TrainConfig(
        train_path=splits_dir / "train.parquet",
        val_path=splits_dir / "val.parquet",
        test_path=splits_dir / "test.parquet",
        artifacts_dir=tmp_path / "artifacts",
        num_epochs=1,
        batch_size=4,
        max_length=32,
        mlflow_tracking_uri=None,
        data_version="test-fixture",
    )
    result = train_rubert_ft(cfg=cfg)
    assert "val_metrics" in result
    assert "test_metrics" in result
    assert (tmp_path / "artifacts" / "model" / "config.json").exists()
