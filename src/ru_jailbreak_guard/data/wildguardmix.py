"""Loader for WildGuardMix Russian subset.

WildGuardMix is a multilingual safety dataset. We extract Russian rows only and
use the is_adversarial flag as the label (True → 1, False → 0).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, empty_canonical_df, validate_schema

SOURCE_NAME = "wildguardmix_ru"
HF_DATASET_ID = "allenai/wildguardmix"


def parse_wildguardmix_parquet(parquet_path: Path) -> pl.DataFrame:
    """Parse a WildGuardMix parquet snapshot into canonical schema (Russian rows only).

    Args:
        parquet_path: Path to a parquet file with columns prompt, lang,
            prompt_harm_category, is_adversarial.

    Returns:
        DataFrame with rows where lang == "ru"; label is 1 when is_adversarial else 0.
    """
    raw = pl.read_parquet(parquet_path).filter(pl.col("lang") == "ru")
    if raw.height == 0:
        return empty_canonical_df()

    df = pl.DataFrame(
        {
            "text": raw["prompt"].to_list(),
            "label": [int(b) for b in raw["is_adversarial"].to_list()],
            "source": [SOURCE_NAME] * raw.height,
            "subcategory": raw["prompt_harm_category"].to_list(),
            "lang": ["ru"] * raw.height,
            # Polars Struct({}) cannot be serialised to Parquet; populate one field.
            "meta": [{"orig_source": HF_DATASET_ID} for _ in range(raw.height)],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    df = df.cast({"label": pl.Int8})
    validate_schema(df)
    return df


def fetch_and_save(output_path: Path, revision: str | None = None) -> int:
    """Download WildGuardMix from HF and write Russian-only canonical parquet.

    The dataset is gated on HuggingFace; access requires HF_TOKEN with approved
    license terms. When unauthenticated, this emits an empty canonical-schema
    parquet and prints a warning so the pipeline can still complete (treats
    WildGuardMix as an optional contributor rather than a hard dependency).

    Args:
        output_path: Destination .parquet file.
        revision: Optional HF dataset revision SHA.

    Returns:
        Number of Russian rows written (0 if unauthenticated).
    """
    import sys

    from datasets import Dataset, load_dataset
    from datasets.exceptions import DatasetNotFoundError

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(HF_DATASET_ID, "wildguardtest", split="test", revision=revision)
    except (DatasetNotFoundError, PermissionError) as exc:
        print(
            f"WARN: WildGuardMix is gated and not accessible ({type(exc).__name__}: {exc}). "
            "Emitting empty parquet. Set HF_TOKEN and accept the dataset terms to enable.",
            file=sys.stderr,
        )
        empty = empty_canonical_df()
        empty.write_parquet(output_path)
        return 0

    assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds).__name__}"
    tmp_parquet = output_path.with_suffix(".raw.parquet")
    ds.to_parquet(str(tmp_parquet))
    df = parse_wildguardmix_parquet(tmp_parquet)
    df.write_parquet(output_path)
    tmp_parquet.unlink()
    return df.height


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()
    n = fetch_and_save(args.output, revision=args.revision)
    print(f"Wrote {n} Russian rows to {args.output}")


if __name__ == "__main__":
    main()
