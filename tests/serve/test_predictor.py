"""Tests for predictor base + a fake concrete predictor."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ru_jailbreak_guard.serve.predictor import PredictionResponse, Predictor, build_app


class _FakePredictor(Predictor):
    family = "tfidf_logreg"

    def __init__(self, *, version: str = "1", data_version: str = "abc") -> None:
        super().__init__(version=version, data_version=data_version)
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def _predict_one(self, text: str) -> tuple[str, float]:
        time.sleep(0.001)
        # Trivial rule: contains "взлом" → jailbreak
        if "взлом" in text.lower():
            return "jailbreak", 0.95
        return "benign", 0.05


def test_health_endpoint() -> None:
    app = build_app(predictor=_FakePredictor())
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_locked_schema() -> None:
    app = build_app(predictor=_FakePredictor(version="7", data_version="dv-xyz"))
    client = TestClient(app)
    resp = client.post("/predict", json={"text": "взлом сейчас"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "jailbreak"
    assert body["confidence"] >= 0.5
    assert body["model_family"] == "tfidf_logreg"
    assert body["model_version"] == "7"
    assert body["data_version"] == "dv-xyz"
    assert body["latency_ms"] >= 0


def test_predict_benign() -> None:
    app = build_app(predictor=_FakePredictor())
    client = TestClient(app)
    resp = client.post("/predict", json={"text": "опиши погоду в москве"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "benign"


def test_response_model_is_PredictionResponse_compatible() -> None:  # noqa: N802
    pr = PredictionResponse(
        label="jailbreak",
        confidence=0.9,
        model_family="tfidf_logreg",
        model_version="1",
        data_version="dv",
        latency_ms=2.5,
    )
    assert pr.label == "jailbreak"
    assert pr.confidence == 0.9


def test_tfidf_logreg_predictor_loads_from_local_files(tmp_path: Path) -> None:
    """Build a tiny TF-IDF + LogReg artifact bundle on disk, point predictor at it."""
    from ru_jailbreak_guard.serve.tfidf_logreg_predictor import TfidfLogregPredictor

    vec = TfidfVectorizer(min_df=1)
    x_mat = vec.fit_transform(["jailbreak text", "benign text"])
    model = LogisticRegression().fit(x_mat, [1, 0])

    art_dir = tmp_path / "art"
    art_dir.mkdir()
    joblib.dump({"vectorizer": vec, "fitted": True}, art_dir / "tfidf.joblib")
    joblib.dump(model, art_dir / "logreg.joblib")

    predictor = TfidfLogregPredictor.from_local(
        artifacts_dir=art_dir, version="1", data_version="dv"
    )
    predictor.load()
    label, conf = predictor._predict_one("jailbreak text")
    assert label in ("jailbreak", "benign")
    assert 0.0 <= conf <= 1.0


def test_lgbm_emb_predictor_loads_from_local_files(tmp_path: Path) -> None:
    """Build a tiny LightGBM model + fake encoder, point predictor at it."""
    from ru_jailbreak_guard.serve.lgbm_emb_predictor import LgbmEmbPredictor

    rng = np.random.default_rng(0)
    x_mat = rng.normal(size=(40, 8)).astype(np.float32)
    y = (x_mat[:, 0] > 0).astype(int)
    booster = lgb.train(
        params={"objective": "binary", "verbose": -1, "num_leaves": 8},
        train_set=lgb.Dataset(x_mat, label=y),
        num_boost_round=20,
    )

    art_dir = tmp_path / "art"
    art_dir.mkdir()
    booster.save_model(str(art_dir / "lgbm.txt"))

    fake_encoder = MagicMock()
    fake_encoder.encode.return_value = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    predictor = LgbmEmbPredictor.from_local(
        booster_path=art_dir / "lgbm.txt",
        encoder=fake_encoder,
        version="1",
        data_version="dv",
    )
    predictor.load()
    label, conf = predictor._predict_one("any text")
    assert label in ("jailbreak", "benign")
    assert 0.0 <= conf <= 1.0


@pytest.mark.slow
def test_rubert_ft_predictor_loads_real_model(tmp_path: Path) -> None:
    """Build a tiny ruBERT-tiny2 model + tokenizer dir on disk, point predictor at it."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ru_jailbreak_guard.serve.rubert_ft_predictor import RubertFtPredictor

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tok = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModelForSequenceClassification.from_pretrained(
        "cointegrated/rubert-tiny2", num_labels=2
    )
    tok.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    predictor = RubertFtPredictor.from_local(model_dir=model_dir, version="1", data_version="dv")
    predictor.load()
    label, conf = predictor._predict_one("привет мир")
    assert label in ("jailbreak", "benign")
    assert 0.0 <= conf <= 1.0
