"""Tests for auto_promote.py - diffing MLflow vs gitops modelVersion pin."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# scripts/ is not a package; import via path manipulation
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import auto_promote  # noqa: E402  # ty: ignore[unresolved-import]


def test_read_pinned_version_extracts_value(tmp_path: Path) -> None:
    values = tmp_path / "values.yaml"
    values.write_text('image:\n  tag: sha-abc\nmodelVersion: "1"\n')
    result = auto_promote.read_pinned_version(values)
    assert result == "1"


def test_read_pinned_version_handles_missing(tmp_path: Path) -> None:
    values = tmp_path / "values.yaml"
    values.write_text("image:\n  tag: sha-abc\n")
    result = auto_promote.read_pinned_version(values)
    assert result is None


def test_write_pinned_version_replaces_in_place(tmp_path: Path) -> None:
    values = tmp_path / "values.yaml"
    values.write_text('image:\n  tag: sha-abc\nmodelVersion: "1"\n')
    auto_promote.write_pinned_version(values, new_version="7")
    text = values.read_text()
    assert 'modelVersion: "7"' in text
    assert "sha-abc" in text  # Other content preserved


def test_get_mlflow_production_version_success() -> None:
    client = MagicMock()
    mv = MagicMock()
    mv.version = "5"
    client.get_model_version_by_alias.return_value = mv
    v = auto_promote.get_mlflow_production_version(
        client=client, model_name="ru-jailbreak-tfidf-logreg"
    )
    assert v == "5"


def test_get_mlflow_production_version_returns_none_on_error() -> None:
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = Exception("not found")
    v = auto_promote.get_mlflow_production_version(
        client=client, model_name="ru-jailbreak-tfidf-logreg"
    )
    assert v is None
