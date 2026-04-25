"""Tests for stratified group-aware train/val/test split."""

import polars as pl

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema
from ru_jailbreak_guard.data.split import group_hash, stratified_split


def make_df(n: int) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "text": [f"sample {i}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "source": [f"src_{i % 3}" for i in range(n)],
            "subcategory": [None] * n,
            "lang": ["ru"] * n,
            "meta": [{} for _ in range(n)],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    return df.cast({"label": pl.Int8})


def test_split_returns_three_dataframes() -> None:
    df = make_df(100)
    train, val, test = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    assert train.height + val.height + test.height == 100


def test_split_validates_schema() -> None:
    df = make_df(20)
    train, val, test = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    for d in (train, val, test):
        validate_schema(d)


def test_split_preserves_class_balance_approximately() -> None:
    df = make_df(200)  # 50/50 by construction
    train, val, test = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        if d.height > 0:
            n_pos = float((d["label"] == 1).sum())
            assert abs(n_pos / d.height - 0.5) < 0.20, f"{name} class balance off"


def test_split_no_text_leak_between_splits() -> None:
    """If two rows share normalized text, they must end up in the same split."""
    df = pl.DataFrame(
        {
            "text": ["one"] * 10 + ["TWO"] * 10 + ["two"] * 10,
            "label": [1] * 30,
            "source": ["s"] * 30,
            "subcategory": [None] * 30,
            "lang": ["ru"] * 30,
            "meta": [{} for _ in range(30)],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    ).cast({"label": pl.Int8})
    train, val, test = stratified_split(df, ratios=(0.34, 0.33, 0.33), seed=42)
    train_groups = {group_hash(t) for t in train["text"].to_list()}
    val_groups = {group_hash(t) for t in val["text"].to_list()}
    test_groups = {group_hash(t) for t in test["text"].to_list()}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_group_hash_normalizes_case_and_whitespace() -> None:
    assert group_hash("Hello WORLD") == group_hash("hello world")
    assert group_hash("a  b") == group_hash("a b")
    assert group_hash("a") != group_hash("b")


def test_split_empty_input() -> None:
    df = pl.DataFrame(schema=CANONICAL_SCHEMA)
    train, val, test = stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42)
    assert train.height == val.height == test.height == 0


def test_split_rejects_invalid_ratios() -> None:
    import pytest

    df = make_df(10)
    with pytest.raises(ValueError, match="ratios must sum to 1"):
        stratified_split(df, ratios=(0.5, 0.3, 0.3), seed=42)
