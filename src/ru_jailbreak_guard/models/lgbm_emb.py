"""LightGBM on ruBERT-tiny2 mean-pooled embeddings."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
import yaml

from ru_jailbreak_guard.evaluate.metrics import (
    binary_metrics,
    confusion_matrix_png,
    latency_benchmark,
    per_source_f1,
)
from ru_jailbreak_guard.features.embeddings import (
    EmbeddingCache,
    Encoder,
    embed_texts_cached,
)


@dataclass
class TrainConfig:
    train_path: Path
    val_path: Path
    test_path: Path
    artifacts_dir: Path
    embedding_cache_dir: Path
    embedding_model_name: str = "cointegrated/rubert-tiny2"
    embedding_revision: str | None = None
    embedding_device: str = "cpu"
    embedding_max_length: int = 256
    num_leaves: int = 31
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 5
    seed: int = 42
    mlflow_tracking_uri: str | None = None
    experiment_name: str = "lgbm_embeddings"
    data_version: str = "unknown"
    register_as: str = "ru-jailbreak-lgbm-emb"
    # injected for testability — production uses None and constructs Encoder lazily
    encoder: Any = field(default=None)


def _load_split(path: Path) -> tuple[list[str], np.ndarray, pl.DataFrame]:
    df = pl.read_parquet(path)
    return df["text"].to_list(), df["label"].to_numpy(), df


def _embed(texts: list[str], encoder: Any, cache: EmbeddingCache, key: str) -> np.ndarray:
    return embed_texts_cached(texts=texts, encoder=encoder, cache=cache, cache_key=key)


def train_lgbm_emb(*, cfg: TrainConfig) -> dict[str, Any]:
    """Train LightGBM on ruBERT embeddings, log to MLflow, save artifacts.

    Args:
        cfg: Training configuration with hyperparameters and IO paths.

    Returns:
        Dict with keys: run_id, val_metrics, test_metrics, test_per_source_f1,
        latency, booster_path, confusion_matrix_path.
    """
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(root=cfg.embedding_cache_dir)
    encoder = cfg.encoder or Encoder(
        model_name=cfg.embedding_model_name,
        revision=cfg.embedding_revision,
        device=cfg.embedding_device,
        max_length=cfg.embedding_max_length,
    )

    train_texts, y_train, _ = _load_split(cfg.train_path)
    val_texts, y_val, _ = _load_split(cfg.val_path)
    test_texts, y_test, test_df = _load_split(cfg.test_path)

    rev_key = cfg.embedding_revision or "latest"
    base = f"{cfg.data_version}__{cfg.embedding_model_name}__{rev_key}"
    x_train = _embed(train_texts, encoder, cache, key=f"{base}__train")
    x_val = _embed(val_texts, encoder, cache, key=f"{base}__val")
    x_test = _embed(test_texts, encoder, cache, key=f"{base}__test")

    train_set = lgb.Dataset(x_train, label=y_train)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "min_child_samples": cfg.min_child_samples,
        "verbose": -1,
        "seed": cfg.seed,
    }
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    val_score = np.asarray(booster.predict(x_val), dtype=np.float32)
    val_pred = (val_score >= 0.5).astype(int)
    test_score = np.asarray(booster.predict(x_test), dtype=np.float32)
    test_pred = (test_score >= 0.5).astype(int)

    val_metrics = binary_metrics(y_true=y_val, y_pred=val_pred, y_score=val_score)
    test_metrics = binary_metrics(y_true=y_test, y_pred=test_pred, y_score=test_score)
    test_per_source = per_source_f1(df=test_df, y_pred=test_pred)

    cm_path = cfg.artifacts_dir / "confusion_matrix.png"
    confusion_matrix_png(
        y_true=y_test, y_pred=test_pred, out_path=cm_path, title="LGBM+ruBERT-emb test"
    )

    booster_path = cfg.artifacts_dir / "lgbm.txt"
    booster.save_model(str(booster_path))

    def _predict_batch(texts: list[str]) -> list[int]:
        emb = encoder.encode(texts)
        return list((np.asarray(booster.predict(emb), dtype=np.float32) >= 0.5).astype(int))

    latency = latency_benchmark(
        predict_fn=_predict_batch, sample_texts=test_texts[: min(50, len(test_texts))], batch_size=1
    )

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "model_family": "lgbm_emb",
                    "embedding_model": cfg.embedding_model_name,
                    "embedding_revision": cfg.embedding_revision or "latest",
                    "num_leaves": cfg.num_leaves,
                    "max_depth": cfg.max_depth,
                    "learning_rate": cfg.learning_rate,
                    "n_estimators": cfg.n_estimators,
                    "min_child_samples": cfg.min_child_samples,
                    "seed": cfg.seed,
                }
            )
            mlflow.set_tags(
                {
                    "data_version": cfg.data_version,
                    "git_sha": os.environ.get("GIT_SHA", "unknown"),
                    "git_branch": os.environ.get("GIT_BRANCH", "unknown"),
                    "model_family": "lgbm_emb",
                }
            )
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            for k, v in latency.items():
                mlflow.log_metric(f"latency_{k}", v)
            for src, f1 in test_per_source.items():
                mlflow.log_metric(f"test_f1_source_{src}", f1)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(booster_path))
            mlflow.lightgbm.log_model(
                lgb_model=booster,
                artifact_path="model",
                registered_model_name=cfg.register_as,
            )
            run_id = run.info.run_id
    else:
        run_id = "no-mlflow"

    return {
        "run_id": run_id,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_per_source_f1": test_per_source,
        "latency": latency,
        "booster_path": str(booster_path),
        "confusion_matrix_path": str(cm_path),
    }


def _params_from_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return yaml.safe_load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM on ruBERT embeddings")
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.parquet"))
    parser.add_argument("--val", type=Path, default=Path("data/splits/val.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts/lgbm_emb"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--data-version", type=str, default="unknown")
    args = parser.parse_args()

    params = _params_from_yaml(args.params)
    ml = params["model_lgbm"]
    emb = params["embedding"]
    mlflow_uri = args.mlflow_uri or params["mlflow"]["tracking_uri"]

    cfg = TrainConfig(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        artifacts_dir=args.artifacts,
        embedding_cache_dir=Path(emb["cache_dir"]),
        embedding_model_name=emb["model_name"],
        embedding_revision=emb["revision"],
        embedding_device=emb["device"],
        embedding_max_length=emb["max_length"],
        num_leaves=ml["num_leaves"],
        max_depth=ml["max_depth"],
        learning_rate=ml["learning_rate"],
        n_estimators=ml["n_estimators"],
        min_child_samples=ml["min_child_samples"],
        seed=params["seed"],
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=params["mlflow"]["experiment_lgbm"],
        data_version=args.data_version,
    )
    result = train_lgbm_emb(cfg=cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
