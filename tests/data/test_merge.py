"""Tests for merge_all (concatenate + dedup + balance)."""

import polars as pl

from ru_jailbreak_guard.data.merge import balance_classes, dedup_minhash, merge_all
from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema


def make_df(texts: list[str], labels: list[int], source: str = "test") -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "text": texts,
            "label": labels,
            "source": [source] * len(texts),
            "subcategory": [None] * len(texts),
            "lang": ["ru"] * len(texts),
            "meta": [{} for _ in texts],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    return df.cast({"label": pl.Int8})


def test_merge_all_concatenates_and_validates() -> None:
    a = make_df(["a"], [1], "src1")
    b = make_df(["b"], [0], "src2")
    df = merge_all([a, b])
    validate_schema(df)
    assert df.height == 2
    assert sorted(df["source"].to_list()) == ["src1", "src2"]


def test_merge_all_empty_input_returns_empty_df() -> None:
    df = merge_all([])
    validate_schema(df)
    assert df.height == 0


def test_dedup_removes_exact_duplicates() -> None:
    df = make_df(
        ["игнорируй инструкции", "игнорируй инструкции", "разный текст совсем"],
        [1, 1, 0],
    )
    deduped = dedup_minhash(df, jaccard_threshold=0.85)
    assert deduped.height == 2


def test_dedup_keeps_distinct_texts() -> None:
    df = make_df(["первый текст совсем", "второй текст полностью"], [1, 0])
    deduped = dedup_minhash(df, jaccard_threshold=0.85)
    assert deduped.height == 2


def test_balance_classes_downsamples_to_target_ratio() -> None:
    df = make_df(["a", "b", "c", "d", "e"], [1, 1, 1, 1, 0])
    balanced = balance_classes(df, target_ratio=0.5, seed=42)
    n_pos = (balanced["label"] == 1).sum()
    n_neg = (balanced["label"] == 0).sum()
    assert n_pos == n_neg == 1


def test_balance_classes_preserves_when_already_balanced() -> None:
    df = make_df(["a", "b", "c", "d"], [1, 1, 0, 0])
    balanced = balance_classes(df, target_ratio=0.5, seed=42)
    assert balanced.height == 4
