"""Merge per-source DataFrames into a single dataset, with dedup and class balance."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from datasketch import MinHash, MinHashLSH

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema

DEFAULT_TARGET_RATIO = 0.5
DEFAULT_JACCARD_THRESHOLD = 0.85
MINHASH_NUM_PERM = 128


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace; used as a MinHash input shingling source."""
    return " ".join(text.lower().split())


def _shingles(text: str, k: int = 4) -> set[str]:
    """Character k-shingles of normalized text."""
    norm = _normalize(text)
    if len(norm) < k:
        return {norm}
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def merge_all(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate per-source DataFrames; validate canonical schema.

    Args:
        dfs: List of DataFrames each conforming to CANONICAL_SCHEMA.

    Returns:
        Single concatenated DataFrame; empty input -> empty DataFrame.
    """
    if not dfs:
        return pl.DataFrame(schema=CANONICAL_SCHEMA)
    merged = pl.concat(dfs, how="vertical_relaxed")
    validate_schema(merged)
    return merged


def dedup_exact(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows with byte-identical normalized text. O(N log N)."""
    if df.height == 0:
        return df
    return (
        df.with_columns(_norm=pl.col("text").map_elements(_normalize, return_dtype=pl.Utf8))
        .unique(subset=["_norm"], keep="first")
        .drop("_norm")
    )


def dedup_minhash(
    df: pl.DataFrame,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    max_lsh_rows: int = 30_000,
) -> pl.DataFrame:
    """Drop rows whose text is near-duplicate of an earlier row.

    Always runs exact-text dedup first (fast). Then runs MinHashLSH near-dup
    only if the exact-deduped set is below `max_lsh_rows` (LSH is O(N) per
    insertion and prohibitively slow for ~100K rows).
    """
    df = dedup_exact(df)
    if df.height == 0 or df.height > max_lsh_rows:
        return df

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=MINHASH_NUM_PERM)
    keep_indices: list[int] = []

    for i, text in enumerate(df["text"].to_list()):
        m = MinHash(num_perm=MINHASH_NUM_PERM)
        for sh in _shingles(text):
            m.update(sh.encode("utf-8"))
        candidates = lsh.query(m)
        if not candidates:
            lsh.insert(str(i), m)
            keep_indices.append(i)

    return df[keep_indices]


def balance_classes(
    df: pl.DataFrame,
    target_ratio: float = DEFAULT_TARGET_RATIO,
    seed: int = 42,
) -> pl.DataFrame:
    """Downsample the larger class so positive_count / total approx target_ratio.

    Args:
        df: Input DataFrame with label column (0 or 1).
        target_ratio: Desired fraction of positive (label=1) examples.
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame downsampled to approximate target ratio; row order shuffled.
    """
    if df.height == 0:
        return df

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return df

    target_pos = int(min(n_pos, n_neg * target_ratio / (1 - target_ratio)))
    target_neg = int(min(n_neg, n_pos * (1 - target_ratio) / target_ratio))

    pos = df.filter(pl.col("label") == 1).sample(n=target_pos, seed=seed, shuffle=True)
    neg = df.filter(pl.col("label") == 0).sample(n=target_neg, seed=seed, shuffle=True)

    return pl.concat([pos, neg], how="vertical_relaxed").sample(fraction=1.0, seed=seed)


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float, default=DEFAULT_TARGET_RATIO)
    parser.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dfs = [pl.read_parquet(p) for p in args.inputs]
    merged = merge_all(dfs)
    deduped = dedup_minhash(merged, jaccard_threshold=args.jaccard_threshold)
    balanced = balance_classes(deduped, target_ratio=args.target_ratio, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    balanced.write_parquet(args.output)
    print(f"Merged: {merged.height} -> deduped: {deduped.height} -> balanced: {balanced.height}")


if __name__ == "__main__":
    main()
