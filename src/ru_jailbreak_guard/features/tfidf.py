"""TF-IDF feature extractor with persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfWrapper:
    """Thin wrapper around sklearn TfidfVectorizer with save/load.

    Defaults are tuned for short Russian prompts: word + bigram, sublinear TF.
    """

    def __init__(
        self,
        *,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            lowercase=True,
            strip_accents=None,
        )
        self.fitted: bool = False

    @property
    def vocab_size(self) -> int:
        if not self.fitted:
            raise RuntimeError("not fitted yet")
        return len(self._vectorizer.vocabulary_)

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        matrix = self._vectorizer.fit_transform(texts)
        self.fitted = True
        return matrix

    def transform(self, texts: list[str]) -> csr_matrix:
        if not self.fitted:
            raise RuntimeError("TfidfWrapper is not fitted")
        return self._vectorizer.transform(texts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self._vectorizer, "fitted": self.fitted}, path)

    @classmethod
    def load(cls, path: Path) -> TfidfWrapper:
        bundle = joblib.load(path)
        instance = cls.__new__(cls)
        instance._vectorizer = bundle["vectorizer"]
        instance.fitted = bundle["fitted"]
        return instance
