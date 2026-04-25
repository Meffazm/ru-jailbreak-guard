"""Standalone bridge to run hivetracered in an isolated venv.

Invoked by ru_jailbreak_guard.data.attack_gen.run_bridge_and_parse via:
    uv run --with hivetracered==1.0.15 --isolated python scripts/run_hivetracered_bridge.py

Reads input JSON ({seeds: [...], categories: [...]}) and writes output JSON
([{seed_text, attack_name, attack_text}, ...]).

Selects only AlgoAttack/TemplateAttack subclasses (no ModelAttack — those
require a live LLM API). Filters by attack_type membership in `categories`.
The seed prompt is the harmful instruction; each attack wraps/transforms it
into an adversarial variant.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def collect_attacks(
    hivetracered_module: Any,
    requested_categories: list[str],
) -> list[tuple[str, str, Any]]:
    """Build (attack_name, attack_type, instance) triples for non-LLM attacks.

    Filters to AlgoAttack/TemplateAttack subclasses (excludes ModelAttack).
    Skips classes whose constructor demands additional positional args.
    """
    attacks: list[tuple[str, str, Any]] = []
    has_model_attack = hasattr(hivetracered_module, "ModelAttack")
    for attack_name, info in hivetracered_module.ATTACK_CLASSES.items():
        attack_class = info["attack_class"]
        attack_type = info["attack_type"]
        is_algo = issubclass(attack_class, hivetracered_module.AlgoAttack)
        is_template = issubclass(attack_class, hivetracered_module.TemplateAttack)
        is_model = has_model_attack and issubclass(attack_class, hivetracered_module.ModelAttack)
        if not (is_algo or is_template) or is_model:
            continue
        if requested_categories and attack_type not in requested_categories:
            continue
        try:
            instance = attack_class()
        except Exception as exc:
            print(
                f"WARN: failed to instantiate {attack_name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        attacks.append((attack_name, attack_type, instance))
    return attacks


def generate_variants(
    seeds: list[str],
    attacks: list[tuple[str, str, Any]],
) -> list[dict[str, str]]:
    """Apply every attack to every seed; tolerate per-attack failures."""
    output: list[dict[str, str]] = []
    for seed in seeds:
        for attack_name, attack_type, instance in attacks:
            try:
                result = instance.apply(seed)
            except Exception as exc:
                print(
                    f"WARN: attack {attack_name} failed on seed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(result, str) and result:
                output.append(
                    {
                        "seed_text": seed,
                        "attack_name": attack_type,
                        "attack_class": attack_name,
                        "attack_text": result,
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--out", dest="output_path", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input_path.read_text(encoding="utf-8"))
    seeds: list[str] = payload["seeds"]
    requested_categories: list[str] = payload.get("categories") or []

    import hivetracered  # only available in the isolated venv

    attacks = collect_attacks(hivetracered, requested_categories)
    print(
        f"Bridge: {len(attacks)} attacks ready, {len(seeds)} seeds — "
        f"will produce up to {len(attacks) * len(seeds)} variants",
        file=sys.stderr,
    )

    output = generate_variants(seeds, attacks)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Bridge: wrote {len(output)} attack variants to {args.output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
