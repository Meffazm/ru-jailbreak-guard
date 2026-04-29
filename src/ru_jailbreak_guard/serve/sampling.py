"""Predictor input sampling buffer that writes 1% of inputs to MinIO."""

from __future__ import annotations

import datetime as dt
import io
import logging
import os
import random
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "predictions"
_DEFAULT_SAMPLE_RATE = 0.01
_DEFAULT_FLUSH_SIZE = 100


def cyrillic_aware_record(*, text: str, label: str, confidence: float) -> dict[str, Any]:
    return {
        "text": text,
        "label": label,
        "confidence": float(confidence),
        "ts": dt.datetime.now(tz=dt.UTC).isoformat(),
    }


@dataclass
class SamplingBuffer:
    """Per-family in-memory buffer that flushes to s3://predictions/<date>/<family>/<uuid>.parquet."""

    family: str
    flush_size: int = _DEFAULT_FLUSH_SIZE
    bucket: str = _DEFAULT_BUCKET
    s3_client: Any = None
    sample_rate: float = _DEFAULT_SAMPLE_RATE
    _rows: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def date_prefix(self) -> str:
        return dt.datetime.now(tz=dt.UTC).date().isoformat()

    def maybe_append(self, *, text: str, label: str, confidence: float) -> None:
        if random.random() >= self.sample_rate:
            return
        rec = cyrillic_aware_record(text=text, label=label, confidence=confidence)
        with self._lock:
            self._rows.append(rec)
            if len(self._rows) >= self.flush_size:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._rows:
            return
        rows = self._rows
        self._rows = []
        try:
            df = pl.DataFrame(rows)
            buf = io.BytesIO()
            df.write_parquet(buf)
            buf.seek(0)
            key = f"{self.date_prefix}/{self.family}/{uuid.uuid4().hex}.parquet"
            self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
        except Exception:
            logger.exception("sampling flush failed; dropping %d rows", len(rows))


def make_default_buffer(family: str) -> SamplingBuffer:
    """Build a buffer using env-driven boto3 client. Lazy import keeps tests light."""
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL", "http://minio.minio.svc.cluster.local:9000"
        ),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return SamplingBuffer(family=family, s3_client=s3)
