"""TF-IDF + Logistic Regression trainer with MLflow logging."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import polars as pl
import yaml
from sklearn.linear_model import LogisticRegression

from ru_jailbreak_guard.evaluate.metrics import (
    binary_metrics,
    confusion_matrix_png,
    latency_benchmark,
    per_source_f1,
)
from ru_jailbreak_guard.features.tfidf import TfidfWrapper


@dataclass
class TrainConfig:
    """Hyperparameters and IO paths."""

    train_path: Path
    val_path: Path
    test_path: Path
    artifacts_dir: Path
    max_features: int = 50_000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    C: float = 1.0
    max_iter: int = 1000
    seed: int = 42
    mlflow_tracking_uri: str | None = None
    experiment_name: str = "tfidf_logreg"
    data_version: str = "unknown"
    register_as: str = "ru-jailbreak-tfidf-logreg"


def _load_split(path: Path) -> tuple[list[str], np.ndarray, pl.DataFrame]:
    df = pl.read_parquet(path)
    return df["text"].to_list(), df["label"].to_numpy(), df


def train_tfidf_logreg(*, cfg: TrainConfig) -> dict[str, Any]:
    """Train, log, register. Returns dict of metrics + artifact paths.

    Args:
        cfg: Training configuration with hyperparameters and IO paths.

    Returns:
        Dict with keys: run_id, val_metrics, test_metrics, test_per_source_f1,
        latency, tfidf_path, logreg_path, confusion_matrix_path.
    """
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_texts, y_train, _ = _load_split(cfg.train_path)
    val_texts, y_val, _ = _load_split(cfg.val_path)
    test_texts, y_test, test_df = _load_split(cfg.test_path)

    tfidf = TfidfWrapper(
        max_features=cfg.max_features,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
    )
    x_train = tfidf.fit_transform(train_texts)
    x_val = tfidf.transform(val_texts)
    x_test = tfidf.transform(test_texts)

    model = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.seed, n_jobs=1)
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    val_score = model.predict_proba(x_val)[:, 1]
    test_pred = model.predict(x_test)
    test_score = model.predict_proba(x_test)[:, 1]

    val_metrics = binary_metrics(y_true=y_val, y_pred=val_pred, y_score=val_score)
    test_metrics = binary_metrics(y_true=y_test, y_pred=test_pred, y_score=test_score)
    test_per_source = per_source_f1(df=test_df, y_pred=test_pred)

    cm_path = cfg.artifacts_dir / "confusion_matrix.png"
    confusion_matrix_png(
        y_true=y_test,
        y_pred=test_pred,
        out_path=cm_path,
        title="TF-IDF+LogReg test",
    )

    tfidf_path = cfg.artifacts_dir / "tfidf.joblib"
    logreg_path = cfg.artifacts_dir / "logreg.joblib"
    tfidf.save(tfidf_path)
    joblib.dump(model, logreg_path)

    def _predict_batch(texts: list[str]) -> list[int]:
        return list(map(int, model.predict(tfidf.transform(texts))))

    latency = latency_benchmark(
        predict_fn=_predict_batch,
        sample_texts=test_texts[: min(50, len(test_texts))],
        batch_size=1,
    )

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "model_family": "tfidf_logreg",
                    "max_features": cfg.max_features,
                    "ngram_range": f"({cfg.ngram_min},{cfg.ngram_max})",
                    "min_df": cfg.min_df,
                    "C": cfg.C,
                    "max_iter": cfg.max_iter,
                    "seed": cfg.seed,
                    "vocab_size": tfidf.vocab_size,
                }
            )
            mlflow.set_tags(
                {
                    "data_version": cfg.data_version,
                    "git_sha": os.environ.get("GIT_SHA", "unknown"),
                    "git_branch": os.environ.get("GIT_BRANCH", "unknown"),
                    "model_family": "tfidf_logreg",
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
            mlflow.log_artifact(str(tfidf_path))
            mlflow.log_artifact(str(logreg_path))
            mlflow.sklearn.log_model(
                sk_model=model,
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
        "tfidf_path": str(tfidf_path),
        "logreg_path": str(logreg_path),
        "confusion_matrix_path": str(cm_path),
    }


def _params_from_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return yaml.safe_load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogReg")
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.parquet"))
    parser.add_argument("--val", type=Path, default=Path("data/splits/val.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts/tfidf_logreg"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--data-version", type=str, default="unknown")
    args = parser.parse_args()

    params = _params_from_yaml(args.params)
    mt = params["model_tfidf"]
    mlflow_uri = args.mlflow_uri or params["mlflow"]["tracking_uri"]

    cfg = TrainConfig(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        artifacts_dir=args.artifacts,
        max_features=mt["max_features"],
        ngram_min=mt["ngram_min"],
        ngram_max=mt["ngram_max"],
        min_df=mt["min_df"],
        C=mt["C"],
        max_iter=mt["max_iter"],
        seed=params["seed"],
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=params["mlflow"]["experiment_tfidf"],
        data_version=args.data_version,
    )
    result = train_tfidf_logreg(cfg=cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
