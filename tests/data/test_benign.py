"""Tests for benign Russian text loader."""

from pathlib import Path

from ru_jailbreak_guard.data.benign import parse_benign_text_lines
from ru_jailbreak_guard.data.schema import validate_schema

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "data"


def test_parse_returns_canonical_schema() -> None:
    lines = (FIXTURES_DIR / "benign_sample.txt").read_text(encoding="utf-8").splitlines()
    df = parse_benign_text_lines(lines)
    validate_schema(df)


def test_parse_assigns_label_zero() -> None:
    lines = (FIXTURES_DIR / "benign_sample.txt").read_text(encoding="utf-8").splitlines()
    df = parse_benign_text_lines(lines)
    assert (df["label"] == 0).all()


def test_parse_records_source_benign_wiki() -> None:
    lines = (FIXTURES_DIR / "benign_sample.txt").read_text(encoding="utf-8").splitlines()
    df = parse_benign_text_lines(lines, source="benign_wiki")
    assert (df["source"] == "benign_wiki").all()


def test_parse_filters_too_short() -> None:
    df = parse_benign_text_lines(
        ["короткий", "Это нормальная строка длиной более десяти символов."]
    )
    assert df.height == 1


def test_parse_filters_non_russian() -> None:
    df = parse_benign_text_lines(
        [
            "This is English text with sufficient length to pass.",
            "Это русский текст с достаточной длиной.",  # noqa: RUF001
        ]
    )
    assert df.height == 1
    assert df["text"][0] == "Это русский текст с достаточной длиной."  # noqa: RUF001
