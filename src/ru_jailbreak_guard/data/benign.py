"""Loader for benign Russian text from Wikipedia / news.

Benign passages serve as the negative class. Wikipedia abstracts (lang="ru") and
news passages are filtered for length and Cyrillic ratio.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ru_jailbreak_guard.data.filters import is_russian_text, text_length_ok
from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, empty_canonical_df, validate_schema

DEFAULT_SOURCE = "benign_wiki"
HF_WIKIPEDIA_ID = "wikimedia/wikipedia"
WIKIPEDIA_RU_CONFIG = "20231101.ru"


def parse_benign_text_lines(
    lines: list[str],
    source: str = DEFAULT_SOURCE,
    min_chars: int = 30,
    max_chars: int = 2000,
    min_cyrillic_ratio: float = 0.7,
) -> pl.DataFrame:
    """Convert a list of Russian text lines into canonical schema with label=0.

    Filters:
        - length within [min_chars, max_chars]
        - Cyrillic ratio >= min_cyrillic_ratio

    Args:
        lines: Raw input lines (one passage per line).
        source: Source tag (default "benign_wiki").
        min_chars: Minimum length in characters.
        max_chars: Maximum length in characters.
        min_cyrillic_ratio: Minimum fraction of Cyrillic letters.

    Returns:
        DataFrame conforming to CANONICAL_SCHEMA; label=0 for all rows.
    """
    kept = [
        line.strip()
        for line in lines
        if line.strip()
        and text_length_ok(line.strip(), min_chars, max_chars)
        and is_russian_text(line.strip(), min_cyrillic_ratio)
    ]
    if not kept:
        return empty_canonical_df()

    df = pl.DataFrame(
        {
            "text": kept,
            "label": [0] * len(kept),
            "source": [source] * len(kept),
            "subcategory": [None] * len(kept),
            "lang": ["ru"] * len(kept),
            # Polars Struct({}) cannot be serialised to Parquet; populate one field.
            "meta": [{"orig_source": source} for _ in kept],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    df = df.cast({"label": pl.Int8})
    validate_schema(df)
    return df


def fetch_and_save_wikipedia(
    output_path: Path,
    n_samples: int,
    revision: str | None = None,
    seed: int = 42,
) -> int:
    """Sample n_samples Russian Wikipedia article first-paragraphs into canonical parquet.

    Args:
        output_path: Destination .parquet file.
        n_samples: Target number of benign passages to write.
        revision: Optional HF dataset revision SHA.
        seed: Random seed for shuffle.

    Returns:
        Number of rows written (may be slightly less than n_samples after filters).
    """
    from datasets import Dataset, load_dataset

    ds = load_dataset(HF_WIKIPEDIA_ID, WIKIPEDIA_RU_CONFIG, split="train", revision=revision)
    assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds).__name__}"
    sampled = ds.shuffle(seed=seed).select(range(min(n_samples * 4, len(ds))))
    lines: list[str] = []
    for row in sampled:
        text = row["text"]
        first_para = text.split("\n\n")[0] if text else ""
        if first_para:
            lines.append(first_para)
        if len(lines) >= n_samples:
            break

    df = parse_benign_text_lines(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df.height


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    n = fetch_and_save_wikipedia(args.output, args.n_samples, seed=args.seed)
    print(f"Wrote {n} benign rows to {args.output}")


if __name__ == "__main__":
    main()
