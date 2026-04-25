"""Loader for HiveTrace/HiveTraceRed/datasets/system_prompt_extraction_ru.csv.

Russian system-prompt-extraction attack examples. Always label=1.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema

SOURCE_NAME = "hivetrace_sysprompt"
RAW_URL = (
    "https://raw.githubusercontent.com/HiveTrace/HiveTraceRed/master/"
    "datasets/system_prompt_extraction_ru.csv"
)


def parse_hivetrace_csv(csv_path: Path) -> pl.DataFrame:
    """Parse system_prompt_extraction_ru.csv into canonical schema (Russian rows only).

    Args:
        csv_path: Path to the HiveTraceRed system-prompt-extraction CSV.

    Returns:
        DataFrame with rows where language == "ru"; label=1, source="hivetrace_sysprompt".
    """
    raw = pl.read_csv(csv_path).filter(pl.col("language") == "ru")
    if raw.height == 0:
        return pl.DataFrame(schema=CANONICAL_SCHEMA)

    df = pl.DataFrame(
        {
            "text": raw["prompt"].to_list(),
            "label": [1] * raw.height,
            "source": [SOURCE_NAME] * raw.height,
            "subcategory": raw["subcategory"].to_list(),
            "lang": ["ru"] * raw.height,
            "meta": [{"category": c} for c in raw["category"].to_list()],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    df = df.cast({"label": pl.Int8})
    validate_schema(df)
    return df


def fetch_and_save(output_path: Path, revision: str = "master") -> int:
    """Download CSV from the pinned HiveTraceRed git revision and save canonical parquet.

    Args:
        output_path: Destination .parquet file.
        revision: Git ref / SHA on HiveTraceRed (default "master").

    Returns:
        Number of Russian rows written.
    """
    import urllib.request

    url = RAW_URL.replace("master", revision) if revision != "master" else RAW_URL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_path.with_suffix(".csv")
    urllib.request.urlretrieve(url, tmp_csv)
    df = parse_hivetrace_csv(tmp_csv)
    df.write_parquet(output_path)
    tmp_csv.unlink()
    return df.height


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", type=str, default="master")
    args = parser.parse_args()
    n = fetch_and_save(args.output, revision=args.revision)
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
