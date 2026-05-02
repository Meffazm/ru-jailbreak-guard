"""Flyte workflows wiring the @task primitives in `tasks.py` into pipelines."""

from __future__ import annotations

from flytekit import workflow

from flyte.workflows.tasks import (
    evaluate,
    promote,
    train_lgbm,
    train_rubert,
    train_tfidf,
)


@workflow
def cheap_train_pipeline(data_version: str) -> tuple[dict, dict]:
    """Retrain TF-IDF + LightGBM in parallel. Each task downloads splits itself."""
    tfidf = train_tfidf(data_version=data_version)
    lgbm = train_lgbm(data_version=data_version)
    return tfidf, lgbm  # ty: ignore[invalid-return-type]


@workflow
def gpu_train_pipeline(data_version: str) -> dict:
    """Retrain the ruBERT-tiny2 fine-tune."""
    rubert = train_rubert(data_version=data_version)
    return rubert  # ty: ignore[invalid-return-type]


@workflow
def evaluate_and_promote(data_version: str) -> dict:
    """Read MLflow runs for this data_version, set @production + @champion."""
    eval_result = evaluate(data_version=data_version)
    promotion = promote(eval_result=eval_result)
    return promotion  # ty: ignore[invalid-return-type]
