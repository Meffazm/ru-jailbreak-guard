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
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio.minio.svc.cluster.local:9000"),
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
