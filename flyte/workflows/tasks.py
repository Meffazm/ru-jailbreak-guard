"""Flyte @task wrappers around existing trainers + MLflow ops.

Each task is a thin wrapper. The actual training logic stays in
`src/ru_jailbreak_guard/models/<family>.py` so it remains usable from
`make train-*` (no Flyte) and from a Flyte workflow alike.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flytekit import Resources, task


def _make_s3_client() -> Any:
    """Build a boto3 S3 client from in-pod env vars (set via Flyte default-env-vars)."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL", "http://minio.minio.svc.cluster.local:9000"
        ),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


@task(
    cache=True,
    cache_version="1",
    retries=3,
    requests=Resources(cpu="100m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def download_splits(
    data_version: str,
    target_dir: str = "/tmp/splits",
    bucket: str = "splits",
) -> str:
    """Download train/val/test parquets from MinIO into a local dir.

    Returns the path to the directory containing the three parquet files.
    """
    s3 = _make_s3_client()
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        s3.download_file(
            Bucket=bucket,
            Key=f"{data_version}/{name}.parquet",
            Filename=str(target / f"{name}.parquet"),
        )
    return str(target)


# ---------------------------------------------------------------------------
# Trainer wrappers — each calls into src/ru_jailbreak_guard/models/<family>.py
# ---------------------------------------------------------------------------

_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000")


@task(
    cache=True,
    cache_version="1",
    retries=1,
    timeout=60 * 10,
    requests=Resources(cpu="500m", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
)
def train_tfidf(splits_dir: str, data_version: str) -> dict:
    """Train TF-IDF + LogReg on the splits at `splits_dir`. Logs to MLflow."""
    from ru_jailbreak_guard.models.tfidf_logreg import TrainConfig, train_tfidf_logreg

    splits = Path(splits_dir)
    cfg = TrainConfig(
        train_path=splits / "train.parquet",
        val_path=splits / "val.parquet",
        test_path=splits / "test.parquet",
        artifacts_dir=splits / "artifacts" / "tfidf",
        mlflow_tracking_uri=_MLFLOW_URI,
        data_version=data_version,
    )
    result = train_tfidf_logreg(cfg=cfg)
    return {**result, "model_family": "tfidf_logreg"}


@task(
    cache=True,
    cache_version="1",
    retries=1,
    timeout=60 * 10,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
)
def train_lgbm(splits_dir: str, data_version: str) -> dict:
    """Train LightGBM on ruBERT embeddings."""
    from ru_jailbreak_guard.models.lgbm_emb import TrainConfig, train_lgbm_emb

    splits = Path(splits_dir)
    artifacts = splits / "artifacts" / "lgbm"
    cfg = TrainConfig(
        train_path=splits / "train.parquet",
        val_path=splits / "val.parquet",
        test_path=splits / "test.parquet",
        artifacts_dir=artifacts,
        embedding_cache_dir=artifacts / "embcache",
        mlflow_tracking_uri=_MLFLOW_URI,
        data_version=data_version,
    )
    result = train_lgbm_emb(cfg=cfg)
    return {**result, "model_family": "lgbm_emb"}


@task(
    cache=True,
    cache_version="1",
    retries=1,
    timeout=60 * 30,
    requests=Resources(cpu="1", mem="3Gi"),
    limits=Resources(cpu="4", mem="6Gi"),
)
def train_rubert(splits_dir: str, data_version: str) -> dict:
    """Fine-tune ruBERT-tiny2."""
    from ru_jailbreak_guard.models.rubert_ft import TrainConfig, train_rubert_ft

    splits = Path(splits_dir)
    cfg = TrainConfig(
        train_path=splits / "train.parquet",
        val_path=splits / "val.parquet",
        test_path=splits / "test.parquet",
        artifacts_dir=splits / "artifacts" / "rubert",
        mlflow_tracking_uri=_MLFLOW_URI,
        data_version=data_version,
    )
    result = train_rubert_ft(cfg=cfg)
    return {**result, "model_family": "rubert_ft"}
