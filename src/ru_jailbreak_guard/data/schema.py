"""Canonical schema for all per-source DataFrames in the data pipeline.

Every loader output and every intermediate DataFrame conforms to CANONICAL_SCHEMA.
The merge stage relies on this guarantee.
"""

import polars as pl

CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "text": pl.String,
    "label": pl.Int8,
    "source": pl.String,
    "subcategory": pl.String,
    "lang": pl.String,
    "meta": pl.Struct({}),
}


def validate_schema(df: pl.DataFrame) -> None:
    """Raise ValueError if df does not conform to CANONICAL_SCHEMA."""
    missing = set(CANONICAL_SCHEMA.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.schema["label"] != pl.Int8:
        raise ValueError(f"label must be Int8, got {df.schema['label']}")
    if df.schema["text"] != pl.String:
        raise ValueError(f"text must be String, got {df.schema['text']}")
