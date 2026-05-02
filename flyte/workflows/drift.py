"""Drift detection Flyte tasks + workflow.

Reads recent production samples from MinIO `s3://predictions/`, compares
against training data at the deployed `data_version`, emits a drift gauge
to Pushgateway.
"""

from __future__ import annotations

import datetime as dt
import io
import os
from typing import Any

import numpy as np
import polars as pl
from flytekit import Resources, task, workflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from ru_jailbreak_guard.evaluate.drift import compute_drift


def _make_s3_client() -> Any:
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


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Cheap proxy embedding: char-feature counts. Avoids loading transformers."""
    if not texts:
        return np.zeros((0, 4), dtype=float)
    feats = np.zeros((len(texts), 4), dtype=float)
    for i, t in enumerate(texts):
        feats[i, 0] = len(t)
        feats[i, 1] = sum(1 for c in t if c.isalpha())
        feats[i, 2] = sum(1 for c in t if "Ѐ" <= c <= "ӿ")
        feats[i, 3] = sum(1 for c in t if c.isdigit())
    return feats


def _read_production_texts(family: str, days: int = 7) -> list[str]:
    s3 = _make_s3_client()
    today = dt.datetime.now(tz=dt.UTC).date()
    texts: list[str] = []
    for offset in range(days):
        date = (today - dt.timedelta(days=offset)).isoformat()
        prefix = f"{date}/{family}/"
        try:
            resp = s3.list_objects_v2(Bucket="predictions", Prefix=prefix)
        except Exception:
            continue
        for obj in resp.get("Contents", []):
            try:
                body = s3.get_object(Bucket="predictions", Key=obj["Key"])["Body"].read()
                df = pl.read_parquet(io.BytesIO(body))
                texts.extend(df["text"].to_list())
            except Exception:
                continue
    return texts


def _read_training_texts(data_version: str, *, max_rows: int = 5000) -> list[str]:
    s3 = _make_s3_client()
    body = s3.get_object(Bucket="splits", Key=f"{data_version}/train.parquet")["Body"].read()
    df = pl.read_parquet(io.BytesIO(body))
    if len(df) > max_rows:
        df = df.sample(n=max_rows, seed=42)
    return df["text"].to_list()


@task(
    cache=False,
    retries=1,
    timeout=60 * 5,
    requests=Resources(cpu="200m", mem="1Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
)
def compute_drift_task(data_version: str, family: str) -> dict:
    """Fetch prod + training texts, compute drift, return small report dict.

    Both fetches happen inside one task to avoid Flyte's 2 MB inter-task output
    limit (training set serialized as list[str] is ~33 MB).
    """
    production_texts = _read_production_texts(family=family)
    training_texts = _read_training_texts(data_version=data_version)
    if len(production_texts) < 100:
        return {
            "drift_detected": False,
            "p_value_length": 1.0,
            "p_value_charset": 1.0,
            "cosine_distance": 0.0,
            "n_production": len(production_texts),
            "n_training": len(training_texts),
            "skipped_reason": "insufficient_samples",
            "family": family,
        }
    p_emb = _embed_texts(production_texts)
    t_emb = _embed_texts(training_texts)
    report = compute_drift(
        production_texts=production_texts,
        training_texts=training_texts,
        production_embeddings=p_emb,
        training_embeddings=t_emb,
    )
    return {
        "drift_detected": report.drift_detected,
        "p_value_length": report.p_value_length,
        "p_value_charset": report.p_value_charset,
        "cosine_distance": report.cosine_distance,
        "n_production": report.n_production,
        "n_training": report.n_training,
        "family": family,
    }


@task(
    cache=False,
    retries=2,
    timeout=60 * 2,
    requests=Resources(cpu="100m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
)
def push_drift_metric(report: dict, family: str) -> None:
    """Push the drift gauge to Pushgateway. Tolerates unreachable gateway."""
    reg = CollectorRegistry()
    g = Gauge(
        "ru_jailbreak_drift_detected",
        "1 if drift detected for this family, 0 otherwise.",
        ["family"],
        registry=reg,
    )
    g.labels(family=family).set(1.0 if report.get("drift_detected") else 0.0)

    pushgateway_url = os.environ.get(
        "PUSHGATEWAY_URL",
        "http://pushgateway.pushgateway.svc.cluster.local:9091",
    )
    try:
        push_to_gateway(pushgateway_url, job=f"drift-{family}", registry=reg)
    except Exception as e:
        print(f"Pushgateway unavailable, skipping: {e}")


@workflow
def drift_detect(data_version: str, family: str = "tfidf_logreg") -> dict:
    """Per-family drift detection. Run via launchplan or pyflyte run."""
    report = compute_drift_task(data_version=data_version, family=family)
    push_drift_metric(report=report, family=family)
    return report  # ty: ignore[invalid-return-type]
