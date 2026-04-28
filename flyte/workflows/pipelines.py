"""Phase 5 Flyte workflows.

Three workflows wire the @task primitives in `tasks.py` into pipelines:
- `cheap_train_pipeline`: TF-IDF + LightGBM
- `gpu_train_pipeline`:   ruBERT fine-tune (CPU-runnable; GPU resource pending)
- `evaluate_and_promote`: read MLflow → set @production / @champion aliases
"""

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
def cheap_train_pipeline(data_version: str) -> dict:
    """Retrain the two cheap families (tfidf, lgbm) and return both run results.

    Cron: weekly (Sunday 03:00 UTC) — see launchplans.py.
    """
    splits = download_splits(data_version=data_version)
    tfidf = train_tfidf(splits_dir=splits, data_version=data_version)
    lgbm = train_lgbm(splits_dir=splits, data_version=data_version)
    return {"tfidf": tfidf, "lgbm": lgbm}


@workflow
def gpu_train_pipeline(data_version: str) -> dict:
    """Retrain the ruBERT-tiny2 fine-tune.

    Currently CPU-runnable (Phase 3 measured ~2 min on smoke data).
    Cron: monthly (1st @ 04:00 UTC) — see launchplans.py.
    """
    splits = download_splits(data_version=data_version)
    rubert = train_rubert(splits_dir=splits, data_version=data_version)
    return {"rubert": rubert}


@workflow
def evaluate_and_promote(data_version: str) -> dict:
    """Read MLflow runs for this data_version → set @production + @champion."""
    eval_result = evaluate(data_version=data_version)
    promotion = promote(eval_result=eval_result)
    return promotion  # ty: ignore[invalid-return-type]
