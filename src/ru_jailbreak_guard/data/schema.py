"""Canonical schema for all per-source DataFrames in the data pipeline.

Every loader output and every intermediate DataFrame conforms to CANONICAL_SCHEMA.
The merge stage relies on this guarantee.
"""

import polars as pl

# Polars accepts both DataType instances (e.g. pl.Struct({})) and the corresponding
# DataTypeClass (e.g. pl.String, pl.Int8) as schema values; ty needs the union to
# permit both forms.
CANONICAL_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "text": pl.String,
    "label": pl.Int8,
    "source": pl.String,
    "subcategory": pl.String,
    "lang": pl.String,
    "meta": pl.Struct({}),
}

# Polars cannot serialise an empty Struct{} to Parquet ("struct type with no
# child field"). For the empty-DataFrame case we use a one-field placeholder
# struct so the file can round-trip; concat with non-empty meta structs uses
# vertical_relaxed in the merge stage and tolerates the schema widening.
EMPTY_PARQUET_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    **CANONICAL_SCHEMA,
    "meta": pl.Struct({"_placeholder": pl.String}),
}


def empty_canonical_df() -> pl.DataFrame:
    """Return a 0-row DataFrame conforming to CANONICAL_SCHEMA, safe to write to Parquet."""
    return pl.DataFrame(schema=EMPTY_PARQUET_SCHEMA)


def validate_schema(df: pl.DataFrame) -> None:
    """Raise ValueError if df does not conform to CANONICAL_SCHEMA.

    Checks every column declared in CANONICAL_SCHEMA. Extra columns are tolerated
    (forward-compatibility for source-specific lineage tags).

    Args:
        df: Polars DataFrame to validate.

    Raises:
        ValueError: If a required column is missing or its dtype differs from the
            canonical declaration. The message indicates which column and why.
    """
    missing = set(CANONICAL_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for column, expected in CANONICAL_SCHEMA.items():
        actual = df.schema[column]
        if column == "meta":
            # meta is allowed to be any Struct; sources may put any fields in it.
            if not isinstance(actual, pl.Struct):
                raise ValueError(f"meta must be a Struct, got {actual}")
            continue
        if actual != expected:
            raise ValueError(f"{column} must be {expected}, got {actual}")
