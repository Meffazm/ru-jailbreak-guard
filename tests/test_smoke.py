"""Smoke tests verifying the package is importable and version is exposed."""

from ru_jailbreak_guard import __version__


def test_package_imports_and_exposes_version() -> None:
    assert __version__ == "0.1.0"
