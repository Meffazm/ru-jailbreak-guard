"""Publish DVC-versioned splits to MinIO so Flyte tasks can pull them.

Reads `dvc.lock`, derives `data_version` from the split stage outs, and uploads
`data/splits/{train,val,test}.parquet` to `s3://splits/<data_version>/`.

Run via `make publish-splits`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import boto3
import yaml


def compute_data_version(lock_path: Path) -> str:
    """Hash the md5s of the split stage outs into a short data_version string."""
    with lock_path.open() as fp:
        lock = yaml.safe_load(fp)
    split_stage = lock["stages"]["split"]
    md5s = sorted(out["md5"] for out in split_stage["outs"])
    digest = hashlib.sha256("|".join(md5s).encode()).hexdigest()
    return digest[:12]


def upload(
    *,
    s3_client: Any,
    bucket: str,
    data_version: str,
    splits_dir: Path,
    skip_if_exists: bool = False,
) -> None:
    """Upload three parquet files to s3://<bucket>/<data_version>/."""
    for name in ("train", "val", "test"):
        local = splits_dir / f"{name}.parquet"
        key = f"{data_version}/{name}.parquet"
        if skip_if_exists:
            try:
                head = s3_client.head_object(Bucket=bucket, Key=key)
                if head.get("ContentLength") == local.stat().st_size:
                    continue
            except Exception:
                pass
        with local.open("rb") as fp:
            s3_client.put_object(Bucket=bucket, Key=key, Body=fp.read())


def _make_s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("dvc.lock"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--bucket", default="splits")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if not args.lock.exists():
        print(f"ERROR: {args.lock} not found. Run `dvc repro` first.", file=sys.stderr)
        sys.exit(1)
    if not args.splits.is_dir():
        print(f"ERROR: {args.splits} not found.", file=sys.stderr)
        sys.exit(1)

    data_version = compute_data_version(args.lock)
    s3 = _make_s3_client()
    upload(
        s3_client=s3,
        bucket=args.bucket,
        data_version=data_version,
        splits_dir=args.splits,
        skip_if_exists=args.skip_existing,
    )
    print(f"Published splits → s3://{args.bucket}/{data_version}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
