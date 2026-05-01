"""Loader for dmtrdr/russian_prompt_injections (HuggingFace, 22K Russian prompt injections).

Always emits label=1 (positive class). Uses prompt_ru as the canonical text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema

SOURCE_NAME = "dmtrdr_ru"
HF_DATASET_ID = "dmtrdr/russian_prompt_injections"


def parse_dmtrdr_json(raw: list[dict[str, Any]]) -> pl.DataFrame:
    """Parse a list of dmtrdr/russian_prompt_injections records into canonical schema.

    Args:
        raw: List of dicts with keys prompt_ru, prompt_en, class, source.

    Returns:
        Polars DataFrame conforming to CANONICAL_SCHEMA. Empty input gives empty DataFrame.
    """
    if not raw:
        return pl.DataFrame(schema=CANONICAL_SCHEMA)

    df = pl.DataFrame(
        {
            "text": [r["prompt_ru"] for r in raw],
            "label": [1] * len(raw),
            "source": [SOURCE_NAME] * len(raw),
            "subcategory": [r.get("class") for r in raw],
            "lang": ["ru"] * len(raw),
            "meta": [
                {"prompt_en": r.get("prompt_en", ""), "orig_source": r.get("source", "")}
                for r in raw
            ],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    validate_schema(df)
    return df


def fetch_and_save(output_path: Path, revision: str | None = None) -> int:
    """Fetch the dataset from HF and write canonical parquet to output_path.

    Args:
        output_path: Destination .parquet file.
        revision: Optional HF dataset revision SHA for reproducibility.

    Returns:
        Number of rows written.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split="train", revision=revision)
    raw = [dict(row) for row in ds]
    df = parse_dmtrdr_json(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df.height


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch dmtrdr/russian_prompt_injections")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()
    n = fetch_and_save(args.output, revision=args.revision)
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
