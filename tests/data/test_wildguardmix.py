"""Tests for WildGuardMix Russian subset loader."""

from pathlib import Path

import polars as pl

from ru_jailbreak_guard.data.schema import validate_schema
from ru_jailbreak_guard.data.wildguardmix import parse_wildguardmix_parquet

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "data"


def test_parse_returns_canonical_schema() -> None:
    df = parse_wildguardmix_parquet(FIXTURES_DIR / "wildguardmix_sample.parquet")
    validate_schema(df)


def test_parse_keeps_only_russian_rows() -> None:
    df = parse_wildguardmix_parquet(FIXTURES_DIR / "wildguardmix_sample.parquet")
    # 2 of 3 fixture rows are Russian
    assert df.height == 2


def test_parse_label_is_adversarial_flag() -> None:
    df = parse_wildguardmix_parquet(FIXTURES_DIR / "wildguardmix_sample.parquet")
    # First Russian row is adversarial=True → label=1
    # Second Russian row is adversarial=False → label=0
    assert sorted(df["label"].to_list()) == [0, 1]


def test_parse_records_source_wildguardmix_ru() -> None:
    df = parse_wildguardmix_parquet(FIXTURES_DIR / "wildguardmix_sample.parquet")
    assert (df["source"] == "wildguardmix_ru").all()


def test_parse_records_subcategory_from_harm_category() -> None:
    df = parse_wildguardmix_parquet(FIXTURES_DIR / "wildguardmix_sample.parquet")
    # The adversarial Russian row has prompt_harm_category="privacy_attack"
    adversarial = df.filter(pl.col("label") == 1)
    assert adversarial["subcategory"][0] == "privacy_attack"
