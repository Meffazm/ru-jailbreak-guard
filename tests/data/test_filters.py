"""Tests for text filtering utilities."""

from ru_jailbreak_guard.data.filters import (
    cyrillic_ratio,
    is_russian_text,
    text_length_ok,
)


def test_cyrillic_ratio_pure_russian() -> None:
    assert cyrillic_ratio("привет мир") > 0.9


def test_cyrillic_ratio_pure_english() -> None:
    assert cyrillic_ratio("hello world") < 0.1


def test_cyrillic_ratio_mixed() -> None:
    ratio = cyrillic_ratio("hello привет")
    assert 0.3 < ratio < 0.7


def test_cyrillic_ratio_empty_string_returns_zero() -> None:
    assert cyrillic_ratio("") == 0.0


def test_is_russian_text_threshold() -> None:
    assert is_russian_text("это русский текст", min_ratio=0.5)
    assert not is_russian_text("this is english", min_ratio=0.5)


def test_text_length_ok_within_bounds() -> None:
    assert text_length_ok("short text", min_chars=5, max_chars=100)


def test_text_length_ok_too_short() -> None:
    assert not text_length_ok("hi", min_chars=5, max_chars=100)


def test_text_length_ok_too_long() -> None:
    assert not text_length_ok("x" * 200, min_chars=5, max_chars=100)
