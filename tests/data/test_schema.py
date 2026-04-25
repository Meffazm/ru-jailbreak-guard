"""Tests for the canonical data schema."""

import polars as pl
import pytest

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema


def test_canonical_schema_has_required_columns() -> None:
    expected = {"text", "label", "source", "subcategory", "lang", "meta"}
    assert set(CANONICAL_SCHEMA.keys()) == expected


def test_validate_schema_accepts_valid_dataframe() -> None:
    df = pl.DataFrame(
        {
            "text": ["привет"],
            "label": [0],
            "source": ["test"],
            "subcategory": [None],
            "lang": ["ru"],
            "meta": [{}],
        },
        schema=CANONICAL_SCHEMA,
    )
    validate_schema(df)


def test_validate_schema_rejects_missing_column() -> None:
    df = pl.DataFrame({"text": ["x"], "label": [0]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(df)


def test_validate_schema_rejects_wrong_label_dtype() -> None:
    df = pl.DataFrame(
        {
            "text": ["x"],
            "label": ["yes"],
            "source": ["test"],
            "subcategory": [None],
            "lang": ["ru"],
            "meta": [{}],
        }
    )
    with pytest.raises(ValueError, match="label must be Int8"):
        validate_schema(df)
