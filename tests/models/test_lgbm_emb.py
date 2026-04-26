"""Tests for LightGBM on ruBERT embeddings."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from ru_jailbreak_guard.models.lgbm_emb import TrainConfig, train_lgbm_emb


def _make_splits(tmp: Path) -> dict[str, Path]:
    """Same shape fixture as TF-IDF; LightGBM needs a few more rows."""
    rows = []
    for i in range(40):
        label = i % 2
        rows.append(
            {
                "text": f"prompt {i} label {label}",
                "label": label,
                "source": "test",
                "subcategory": None,
                "lang": "ru",
            }
        )
    train, val, test = rows[:24], rows[24:32], rows[32:]
    splits_dir = tmp / "splits"
    splits_dir.mkdir()
    paths = {
        "train": splits_dir / "train.parquet",
        "val": splits_dir / "val.parquet",
        "test": splits_dir / "test.parquet",
    }
    pl.DataFrame(train).write_parquet(paths["train"])
    pl.DataFrame(val).write_parquet(paths["val"])
    pl.DataFrame(test).write_parquet(paths["test"])
    return paths


def test_train_with_mocked_encoder(tmp_path: Path) -> None:
    paths = _make_splits(tmp_path)

    fake_encoder = MagicMock()

    # 8-dim embeddings; correlated with label so model can learn something.
    def fake_encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = 1.0 if "label 1" in t else -1.0
        return out

    fake_encoder.encode.side_effect = fake_encode

    cfg = TrainConfig(
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        artifacts_dir=tmp_path / "artifacts",
        embedding_cache_dir=tmp_path / "cache",
        embedding_model_name="fake-model",
        embedding_revision=None,
        embedding_device="cpu",
        embedding_max_length=64,
        num_leaves=15,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=50,
        min_child_samples=2,
        seed=42,
        mlflow_tracking_uri=None,
        experiment_name="test_lgbm",
        data_version="test-fixture",
        encoder=fake_encoder,
    )
    result = train_lgbm_emb(cfg=cfg)
    assert result["val_metrics"]["f1"] >= 0.5  # mock encoder is informative
    assert (tmp_path / "artifacts" / "lgbm.txt").exists()
