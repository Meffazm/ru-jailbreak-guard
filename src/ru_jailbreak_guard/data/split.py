"""Group-aware stratified train/val/test split.

Two rows whose normalized text hashes to the same group end up in the same split,
preventing train/test leakage from near-duplicates that survived dedup.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema


def group_hash(text: str) -> str:
    """Stable group identifier: sha256 of normalized (lowercase, whitespace-collapsed) text."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def stratified_split(
    df: pl.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split df into (train, val, test). Groups (same normalized text) stay together.

    Bucket assignment uses the group_hash (deterministic given seed and text).
    Stratification by `label` is approximate - the bucket-by-hash approach
    preserves class proportions in expectation.

    Args:
        df: DataFrame conforming to CANONICAL_SCHEMA.
        ratios: (train, val, test) fractions; must sum to 1.0.
        seed: Reserved for future seeded shuffles (current impl is deterministic by hash).

    Returns:
        Tuple of (train, val, test) DataFrames.

    Raises:
        ValueError: If ratios do not sum to 1.0.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    if df.height == 0:
        empty = pl.DataFrame(schema=CANONICAL_SCHEMA)
        return empty, empty, empty

    df_with_grp = df.with_columns(
        pl.col("text").map_elements(group_hash, return_dtype=pl.String).alias("_grp")
    )
    df_with_bucket = df_with_grp.with_columns(
        (
            pl.col("_grp").map_elements(lambda h: int(h, 16) % 10000, return_dtype=pl.Int64)
            / 10000.0
        ).alias("_bucket")
    )

    train_p, val_p, _ = ratios
    train = df_with_bucket.filter(pl.col("_bucket") < train_p).drop(["_grp", "_bucket"])
    val = df_with_bucket.filter(
        (pl.col("_bucket") >= train_p) & (pl.col("_bucket") < train_p + val_p)
    ).drop(["_grp", "_bucket"])
    test = df_with_bucket.filter(pl.col("_bucket") >= train_p + val_p).drop(["_grp", "_bucket"])

    for d in (train, val, test):
        validate_schema(d)
    return train, val, test


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-val", type=Path, required=True)
    parser.add_argument("--out-test", type=Path, required=True)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pl.read_parquet(args.input)
    ratios = (args.ratios[0], args.ratios[1], args.ratios[2])
    train, val, test = stratified_split(df, ratios=ratios, seed=args.seed)
    for path, d in [(args.out_train, train), (args.out_val, val), (args.out_test, test)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        d.write_parquet(path)
    print(f"train={train.height} val={val.height} test={test.height}")


if __name__ == "__main__":
    main()
