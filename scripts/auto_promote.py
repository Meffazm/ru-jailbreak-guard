"""Auto-promotion: diff MLflow @production aliases against gitops modelVersion.

Run hourly by `.github/workflows/promote.yaml`. If any pin lags MLflow's
@production alias, opens a PR with updates. No-ops when MLFLOW_TRACKING_URI
is unset.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_FAMILY_TO_VALUES_PATH = {
    "ru-jailbreak-tfidf-logreg": "gitops/apps/model-tfidf/values.yaml",
    "ru-jailbreak-lgbm-emb": "gitops/apps/model-lgbm/values.yaml",
    "ru-jailbreak-rubert-ft": "gitops/apps/model-rubert-ft/values.yaml",
}

_PIN_RE = re.compile(r'modelVersion:\s*"?([^"\n]+)"?')


def read_pinned_version(values_path: Path) -> str | None:
    """Extract the modelVersion value from a Helm values.yaml file."""
    text = values_path.read_text()
    m = _PIN_RE.search(text)
    return m.group(1).strip() if m else None


def write_pinned_version(values_path: Path, new_version: str) -> None:
    """Replace the modelVersion value in-place; preserves surrounding YAML."""
    text = values_path.read_text()
    new_text = _PIN_RE.sub(f'modelVersion: "{new_version}"', text)
    values_path.write_text(new_text)


def get_mlflow_production_version(*, client: Any, model_name: str) -> str | None:
    """Resolve the MLflow registered-model version aliased to @production.

    Returns None on any client error (alias missing, registry unreachable,
    auth failure) so the caller can skip cleanly.
    """
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="production")
        return str(mv.version)
    except Exception:
        return None


def _build_mlflow_client(mlflow_uri: str) -> Any | None:
    """Construct MlflowClient or return None if mlflow is missing/unreachable."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("mlflow not installed; skipping.", file=sys.stderr)
        return None

    try:
        return MlflowClient(tracking_uri=mlflow_uri)
    except Exception as e:
        print(f"MLflow client init failed: {e}; skipping.", file=sys.stderr)
        return None


def collect_diffs(
    *, client: Any, repo_root: Path, apply_writes: bool
) -> list[tuple[str, str, str]]:
    """For each model family, compare MLflow @production vs pinned modelVersion.

    Returns a list of (model_name, old_pinned, new_prod) tuples for all models
    whose pin lags MLflow. When `apply_writes` is True, the values.yaml is
    rewritten in place; otherwise only the diff list is returned.
    """
    diffs: list[tuple[str, str, str]] = []
    for model_name, rel_path in _FAMILY_TO_VALUES_PATH.items():
        values_path = repo_root / rel_path
        if not values_path.exists():
            continue
        prod = get_mlflow_production_version(client=client, model_name=model_name)
        pinned = read_pinned_version(values_path)
        if prod is None or pinned is None or prod == pinned:
            continue
        diffs.append((model_name, pinned, prod))
        if apply_writes:
            write_pinned_version(values_path, prod)
    return diffs


def open_promotion_pr(diffs: list[tuple[str, str, str]]) -> None:
    """Create branch, commit, push, and open a PR for the queued promotions."""
    branch = f"auto-promote/{os.environ.get('GITHUB_RUN_ID', 'manual')}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore: auto-promote {len(diffs)} model(s)"],
        check=True,
    )
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)

    body = "\n".join(f"- {n}: `{o}` -> `{p}`" for n, o, p in diffs)
    pr_body = f"## Auto-promotion\n\n{body}\n\nReady to merge if CI green."
    subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            f"chore: auto-promote {len(diffs)} model version(s)",
            "--body",
            pr_body,
            "--label",
            "auto-promote",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.mlflow_uri:
        print("MLFLOW_TRACKING_URI not set; skipping (no-op).", file=sys.stderr)
        return 0

    client = _build_mlflow_client(args.mlflow_uri)
    if client is None:
        return 0

    diffs = collect_diffs(client=client, repo_root=args.repo_root, apply_writes=not args.dry_run)

    if not diffs:
        print("No promotion needed.")
        return 0

    print(f"Promotions to apply ({'dry-run' if args.dry_run else 'live'}):")
    for name, old, new in diffs:
        print(f"  {name}: {old} -> {new}")

    if args.dry_run:
        return 0

    open_promotion_pr(diffs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
