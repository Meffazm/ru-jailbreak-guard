"""Tests for data profile HTML report."""

from pathlib import Path

import polars as pl

from ru_jailbreak_guard.data.profile import write_profile_report
from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA


def make_df() -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "text": ["короткий", "немного длиннее", "ещё чуть подлиннее"],
            "label": [1, 0, 1],
            "source": ["s1", "s2", "s1"],
            "subcategory": [None, None, "x"],
            "lang": ["ru"] * 3,
            "meta": [{} for _ in range(3)],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    return df.cast({"label": pl.Int8})


def test_write_profile_creates_html_file(tmp_path: Path) -> None:
    out = tmp_path / "profile.html"
    write_profile_report(make_df(), out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>") or "<html" in content


def test_profile_contains_class_counts(tmp_path: Path) -> None:
    out = tmp_path / "profile.html"
    write_profile_report(make_df(), out)
    content = out.read_text(encoding="utf-8")
    assert "label" in content.lower()
    assert "source" in content.lower()


def test_profile_contains_sample_rows(tmp_path: Path) -> None:
    out = tmp_path / "profile.html"
    write_profile_report(make_df(), out)
    content = out.read_text(encoding="utf-8")
    assert "короткий" in content or "немного длиннее" in content


def test_profile_handles_empty_dataframe(tmp_path: Path) -> None:
    empty_df = pl.DataFrame(schema=CANONICAL_SCHEMA)
    out = tmp_path / "profile_empty.html"
    write_profile_report(empty_df, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "0 rows" in content or "0 строк" in content


def test_profile_creates_parent_directory(tmp_path: Path) -> None:
    nested_out = tmp_path / "nested" / "subdir" / "profile.html"
    write_profile_report(make_df(), nested_out)
    assert nested_out.exists()
