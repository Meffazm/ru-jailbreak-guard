# ADR-0010: Use HiveTrace data, tooling, and zero-shot baseline

**Status:** Accepted
**Date:** 2026-04-25
**Deciders:** Dmitrii Velibekov

## Context

The project needs Russian-language jailbreak/prompt-injection examples for training.
Original plan was to translate English datasets (jailbreak_llms, AdvBench) via LLM API
and synthesize additional Russian examples via custom prompt templates.

During Phase 0, we discovered the HiveTrace ecosystem (ITMO University + AI Talent Hub):

- **`dmtrdr/russian_prompt_injections`** on HuggingFace — 22,394 bilingual (ru+en)
  prompt injection examples, 6 injection technique classes, Apache 2.0 license.
  The dataset author overlaps with the HiveTrace team and the dataset aligns with
  HiveTrace's red-teaming taxonomy.
- **`HiveTrace/HiveTraceRed`** GitHub repo — Apache 2.0, contains
  `datasets/system_prompt_extraction_ru.csv` plus the `hivetracered` PyPI package:
  80+ adversarial attack templates across 10 categories (roleplay, persuasion,
  token smuggling, etc.) that programmatically generate variants from base prompts.
- **`hivetrace/gliner-guard-v1`** HuggingFace collection — three GLiNER2-based
  zero-shot classifiers for PII + content safety + prompt attacks detection,
  multi-language (incl. Russian).

All components are Apache 2.0, MIT-compatible.

## Decision

Three integration choices:

1. **Adopt `dmtrdr/russian_prompt_injections` as the PRIMARY training data source.**
   Retire the translation pipeline (jailbreak_llms + AdvBench → Russian via LLM API).
   Pull the dataset via DVC, pinned to a specific HF revision SHA for reproducibility.

2. **Use the `hivetracered` package for adversarial synthesis** instead of bespoke
   LLM-template synthesis. Apply its 80+ attack templates programmatically to
   Russian seed prompts. Pin the package version in `params.yaml`.

3. **Use `hivetrace/gliner-guard-uniencoder` as a zero-shot baseline reference**,
   evaluated offline alongside our trained models. Do NOT deploy as a 4th
   `InferenceService` — keep offline-only to control scope. Defense narrative:
   "our domain-specific fine-tune of ruBERT-tiny2 achieves +X% F1 over the leading
   Russian zero-shot guardrail."

## Consequences

### Positive

- 22K real Russian examples available immediately, eliminating translation
  pipeline complexity
- LLM API budget drops from ~150 RUB to ~50 RUB (only spot-quality sampling remains)
- Phase 1 effort drops from ~1 week to ~4-5 days
- Adversarial synthesis becomes more diverse (80+ programmatic templates vs.
  handful of hand-written prompts)
- GLiNER baseline strengthens defense narrative — direct comparison to the
  leading Russian guardrail
- Apache 2.0 licensing is fully MIT-compatible; no redistribution or fork required

### Negative

- New dependency on the `hivetracered` PyPI package (Apache 2.0; pinned via
  uv.lock + params.yaml)
- HiveTrace team's roadmap could change — datasets/repos could be archived or
  moved (mitigated: mirror to S3 at first fetch, pin SHAs)
- Some `hivetracered` attack templates may not generate valid Russian (mitigated:
  per-category sampling check at Phase 1; disable categories that produce
  English/garbage)
- Defense committee may ask "did you train your own dataset?" — answer: "we built
  atop the strongest available Russian-language source rather than translating
  from English, which is honest data engineering"

### Neutral

- Translation pipeline (jailbreak_llms, AdvBench) is **deferred, not deleted** —
  could be re-introduced if HiveTrace data quality is insufficient
- WildGuardMix's Russian slice still pulled, but used only for held-out
  cross-validation (not training), as documented in the design spec §5.1

## Alternatives considered

- **Stick to original translation-heavy plan** — rejected because translating
  from English is wasted work when 22K Apache-2.0-licensed Russian examples
  already exist
- **Datasets only, skip `hivetracered` package** — rejected because programmatic
  adversarial generation gives more diverse coverage than hand-written templates,
  and adds a one-time install dep with no ongoing cost
- **Fork HiveTrace components into our repo** — rejected because (a) Apache 2.0
  doesn't require it, (b) we'd own maintenance burden of code we didn't write,
  (c) `dvc pull` from HuggingFace is the cleaner reproducibility story
- **Deploy GLiNER as a 4th InferenceService for runtime comparison** — rejected
  because it expands scope (4 deployed models vs. 3) without commensurate
  portfolio gain; offline comparison via MLflow is sufficient for the defense
  narrative
