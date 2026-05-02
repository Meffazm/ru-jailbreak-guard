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


def _download_splits(data_version: str, target_dir: str = "/tmp/splits") -> Path:
    """Pull the three split parquets from `s3://splits/<data_version>/` into target_dir.

    Each task runs in its own pod with an isolated filesystem, so each trainer
    must fetch splits itself rather than receiving a path from an upstream task.
    """
    s3 = _make_s3_client()
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        s3.download_file(
            Bucket="splits",
            Key=f"{data_version}/{name}.parquet",
            Filename=str(target / f"{name}.parquet"),
        )
    return target


# ---------------------------------------------------------------------------
# Trainer wrappers — each calls into src/ru_jailbreak_guard/models/<family>.py
# ---------------------------------------------------------------------------

_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000")


@task(
    cache=True,
    cache_version="2",
    retries=1,
    timeout=60 * 10,
    requests=Resources(cpu="500m", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
)
def train_tfidf(data_version: str) -> dict:
    """Download splits + train TF-IDF + LogReg + log to MLflow."""
    from ru_jailbreak_guard.models.tfidf_logreg import TrainConfig, train_tfidf_logreg

    splits = _download_splits(data_version)
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
    cache_version="2",
    retries=1,
    timeout=60 * 30,
    requests=Resources(cpu="1", mem="4Gi"),
    limits=Resources(cpu="2", mem="16Gi"),
)
def train_lgbm(data_version: str) -> dict:
    """Download splits + train LightGBM on ruBERT embeddings."""
    from ru_jailbreak_guard.models.lgbm_emb import TrainConfig, train_lgbm_emb

    splits = _download_splits(data_version)
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
    cache_version="2",
    retries=1,
    timeout=60 * 90,
    requests=Resources(cpu="1", mem="4Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
)
def train_rubert(data_version: str) -> dict:
    """Download splits + fine-tune ruBERT-tiny2."""
    from ru_jailbreak_guard.models.rubert_ft import TrainConfig, train_rubert_ft

    splits = _download_splits(data_version)
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


# ---------------------------------------------------------------------------
# Evaluation + promotion
# ---------------------------------------------------------------------------


def _mlflow_client():
    """Build an MlflowClient from env. Helper exists so tests can patch it."""
    import mlflow

    mlflow.set_tracking_uri(_MLFLOW_URI)
    from mlflow.tracking import MlflowClient

    return MlflowClient()


@task(
    cache=False,
    retries=2,
    timeout=60 * 2,
    requests=Resources(cpu="100m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def evaluate(data_version: str) -> dict:
    """Find the best run per model_family for this data_version.

    Returns:
        {
          "per_family": {family: {"run_id": str, "val_f1": float, ...}},
          "champion":   {"family": str, "run_id": str, "val_f1": float, ...},
        }
    """
    client = _mlflow_client()
    # Search across all experiments, filter by tag
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in client.search_experiments()],
        filter_string=f"tags.data_version = '{data_version}'",
        max_results=200,
    )
    if not runs:
        raise RuntimeError(f"No successful runs found for data_version={data_version}")

    per_family: dict[str, dict] = {}
    for r in runs:
        family = r.data.tags.get("model_family")
        if not family:
            continue
        val_f1 = r.data.metrics.get("val_f1")
        if val_f1 is None:
            continue
        current_best = per_family.get(family)
        if current_best is None or val_f1 > current_best["val_f1"]:
            per_family[family] = {
                "run_id": r.info.run_id,
                "val_f1": val_f1,
                "family": family,
            }

    if not per_family:
        raise RuntimeError(
            f"No successful runs with model_family tag for data_version={data_version}"
        )

    champion = max(per_family.values(), key=lambda x: x["val_f1"])
    return {"per_family": per_family, "champion": champion}


_FAMILY_TO_MODEL_NAME = {
    "tfidf_logreg": "ru-jailbreak-tfidf-logreg",
    "lgbm_emb": "ru-jailbreak-lgbm-emb",
    "rubert_ft": "ru-jailbreak-rubert-ft",
}


@task(
    cache=False,
    retries=2,
    timeout=60 * 2,
    requests=Resources(cpu="100m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def promote(eval_result: dict) -> dict:
    """Set MLflow registry aliases: @production per family, @champion overall.

    Returns the version numbers chosen (logged as Flyte task output).
    """
    client = _mlflow_client()
    per_family_versions: dict[str, str] = {}

    for family, info in eval_result["per_family"].items():
        model_name = _FAMILY_TO_MODEL_NAME.get(family)
        if model_name is None:
            continue
        # Find the registered model version that wraps this run_id.
        versions = client.search_model_versions(
            filter_string=f"run_id = '{info['run_id']}' and name = '{model_name}'",
        )
        if not versions:
            continue
        version = versions[0].version
        per_family_versions[family] = version
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=version,
        )

    champion = eval_result["champion"]
    champion_family = champion["family"]
    champion_model = _FAMILY_TO_MODEL_NAME[champion_family]
    champion_version = per_family_versions[champion_family]
    client.set_registered_model_alias(
        name=champion_model,
        alias="champion",
        version=champion_version,
    )

    return {
        "per_family_versions": per_family_versions,
        "champion_family": champion_family,
        "champion_version": champion_version,
    }
