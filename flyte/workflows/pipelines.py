"""Flyte workflows wiring the @task primitives in `tasks.py` into pipelines."""

from __future__ import annotations

from flytekit import workflow

from flyte.workflows.tasks import (
    download_splits,
    evaluate,
    promote,
    train_lgbm,
    train_rubert,
    train_tfidf,
)


@workflow
def cheap_train_pipeline(data_version: str) -> tuple[dict, dict]:
    """Retrain TF-IDF + LightGBM."""
    splits = download_splits(data_version=data_version)
    tfidf = train_tfidf(splits_dir=splits, data_version=data_version)
    lgbm = train_lgbm(splits_dir=splits, data_version=data_version)
    return tfidf, lgbm  # ty: ignore[invalid-return-type]


@workflow
def gpu_train_pipeline(data_version: str) -> dict:
    """Retrain the ruBERT-tiny2 fine-tune."""
    splits = download_splits(data_version=data_version)
    rubert = train_rubert(splits_dir=splits, data_version=data_version)
    return rubert  # ty: ignore[invalid-return-type]


@workflow
def evaluate_and_promote(data_version: str) -> dict:
    """Read MLflow runs for this data_version, set @production + @champion."""
    eval_result = evaluate(data_version=data_version)
    promotion = promote(eval_result=eval_result)
    return promotion  # ty: ignore[invalid-return-type]
