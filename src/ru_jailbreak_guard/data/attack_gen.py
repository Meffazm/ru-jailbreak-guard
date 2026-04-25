"""Wrapper for hivetracered adversarial-prompt generation.

Architecture: this module does NOT import hivetracered (which has a hard
pyarrow<20 conflict with datasets). Instead, it shells out to a bridge script
that runs hivetracered in an isolated ephemeral venv via `uv run --with`.

The bridge script writes a JSON list of {seed_text, attack_name, attack_text};
this module parses that JSON into the canonical Polars schema.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import polars as pl

from ru_jailbreak_guard.data.schema import CANONICAL_SCHEMA, validate_schema

SOURCE_NAME = "hivetracered_attack"
HIVETRACERED_VERSION = "1.0.15"
BRIDGE_SCRIPT = (
    Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "run_hivetracered_bridge.py"
)

# Informational — actual category set is determined at runtime by the bridge script
# from hivetracered's ATTACK_TYPES registry. This list is the agreed Phase 1 subset
# of non-LLM attack categories (AlgoAttack/TemplateAttack subclasses only, no
# ModelAttack which would require live LLM API calls).
AVAILABLE_ATTACK_CATEGORIES: list[str] = [
    "roleplay",
    "context_switching",
    "token_smuggling",
    "text_structure_modification",
    "output_formatting",
    "task_deflection",
    "irrelevant_information",
    "in_context_learning",
    "simple_instructions",
]


def parse_attack_bridge_output(items: list[dict[str, Any]]) -> pl.DataFrame:
    """Parse the bridge script's JSON output into canonical schema.

    Args:
        items: List of dicts with keys seed_text, attack_name, attack_text.

    Returns:
        DataFrame conforming to CANONICAL_SCHEMA; label=1 for all rows.
    """
    if not items:
        return pl.DataFrame(schema=CANONICAL_SCHEMA)

    df = pl.DataFrame(
        {
            "text": [item["attack_text"] for item in items],
            "label": [1] * len(items),
            "source": [SOURCE_NAME] * len(items),
            "subcategory": [item["attack_name"] for item in items],
            "lang": ["ru"] * len(items),
            "meta": [{"seed_text": item.get("seed_text", "")} for item in items],
        },
        schema={**CANONICAL_SCHEMA, "meta": pl.Struct},
    )
    df = df.cast({"label": pl.Int8})
    validate_schema(df)
    return df


def run_bridge_and_parse(
    seeds_path: Path,
    output_path: Path,
    categories: list[str] | None = None,
) -> int:
    """Invoke the bridge script in an isolated venv and parse its JSON output.

    Args:
        seeds_path: Parquet file with seed prompts (column "text" required).
        output_path: Destination .parquet for canonical-schema attacks.
        categories: Optional list of attack category names to enable.

    Returns:
        Number of attack rows written.
    """
    cats = categories or AVAILABLE_ATTACK_CATEGORIES
    bridge_in = output_path.with_suffix(".bridge_in.json")
    bridge_out = output_path.with_suffix(".bridge_out.json")

    seeds_df = pl.read_parquet(seeds_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bridge_in.write_text(
        json.dumps({"seeds": seeds_df["text"].to_list(), "categories": cats}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "uv",
            "run",
            "--with",
            f"hivetracered=={HIVETRACERED_VERSION}",
            "--isolated",
            "python",
            str(BRIDGE_SCRIPT),
            "--in",
            str(bridge_in),
            "--out",
            str(bridge_out),
        ],
        check=True,
    )

    items = json.loads(bridge_out.read_text(encoding="utf-8"))
    df = parse_attack_bridge_output(items)
    df.write_parquet(output_path)
    bridge_in.unlink()
    bridge_out.unlink()
    return df.height


def main() -> None:
    """CLI entry point for DVC stage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    args = parser.parse_args()
    n = run_bridge_and_parse(args.seeds_input, args.output, args.categories)
    print(f"Wrote {n} attacks to {args.output}")


if __name__ == "__main__":
    main()
