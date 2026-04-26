"""Tests for TF-IDF feature wrapper."""

from pathlib import Path

import pytest

from ru_jailbreak_guard.features.tfidf import TfidfWrapper


def test_fit_transform_produces_sparse_matrix() -> None:
    texts = ["привет мир", "тест проверка", "привет тест"]
    wrap = TfidfWrapper(max_features=20, ngram_range=(1, 2))
    matrix = wrap.fit_transform(texts)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] <= 20
    assert wrap.fitted is True


def test_transform_after_fit_uses_same_vocab() -> None:
    train = ["один два три", "четыре пять"]
    test = ["один два", "новый текст"]
    # min_df=1 so the 2-doc fixture vocabulary survives pruning.
    wrap = TfidfWrapper(max_features=10, min_df=1)
    wrap.fit_transform(train)
    matrix_test = wrap.transform(test)
    assert matrix_test.shape[0] == 2
    assert matrix_test.shape[1] == wrap.vocab_size


def test_transform_before_fit_raises() -> None:
    wrap = TfidfWrapper()
    with pytest.raises(RuntimeError, match="not fitted"):
        wrap.transform(["text"])


def test_save_load_round_trip(tmp_path: Path) -> None:
    texts = ["русский текст один", "русский текст два", "другой пример"]
    wrap = TfidfWrapper(max_features=15)
    wrap.fit_transform(texts)
    path = tmp_path / "tfidf.joblib"
    wrap.save(path)
    loaded = TfidfWrapper.load(path)
    assert loaded.vocab_size == wrap.vocab_size
    matrix1 = wrap.transform(["русский"])
    matrix2 = loaded.transform(["русский"])
    assert (matrix1 != matrix2).nnz == 0
