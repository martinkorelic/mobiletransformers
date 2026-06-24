# Documentation Set & Compatibility Matrix

**Priority #30 | Prerequisites: #22–#26 (`03_code_plans/*`), #18 (`02_code_plans/01`), #12 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`) | Blocks: #31 (`05_code_plans/05`, release)**

> Author each doc as its underlying contract stabilizes — not before, to avoid drift.

## Purpose

Produce the public user-facing documentation set under `docs/` (separate from `agent_docs/`, which stays research/planning) and a living compatibility matrix, so the framework is adoptable and citable. Each page is written when its contract is locked, sourced from the corresponding code plan.

## Touched / new files

- NEW `docs/ARCHITECTURE.md` — training/export/runtime data flow; native graph I/O contract from `03_code_plans/01`.
- NEW `docs/PUBLIC_API.md` — Kotlin + Python public API; HF mapping table from `03_code_plans/02` and `03_tier2...md`.
- NEW `docs/MODEL_FORMAT.md` — the `mobiletransformers_manifest.json` + `weight_handoff_map.json` contract (`00_code_plans/06`, `00_code_plans/07`).
- NEW `docs/CONFIGURATION.md` — the public config contract from `00_code_plans/09`: the enum vocabulary (mirrored Python ↔ Kotlin), the Pydantic config models + their generated `schemas/*.schema.json` (the cross-boundary JSON contract for `training_config.json`/`generation_config.json`/`rag_config.json`), and the PEFT/architecture/merger **registries** as the public extension points ("to add a method/architecture/merger, add a registry entry"). Source of truth: `00_code_plans/09`.
- NEW `docs/ANDROID_SDK.md` — Gradle/AAR setup, permissions, ABI, local Maven (`05_code_plans/03`).
- NEW `docs/RAG.md` — embedding model, ingestion, vector store semantics (the `1 - score`, 8-dim constraint), retrieval, grounded generation (`03_code_plans/03`–`05`).
- NEW `docs/EXPORT.md` — one-command export + toolchain notes (`02_code_plans/05`, `00_code_plans/03`).
- NEW `docs/COMPATIBILITY_MATRIX.md` — the matrix below.
- NEW `docs/RELEASE_CHECKLIST.md` + `CHANGELOG.md` (owned/finalized by #31; created here).
- Existing `docs/mobile_evaluation.md` — referenced for the measurement style; not rewritten.

## Data contracts / interfaces

### Compatibility matrix schema (`docs/COMPATIBILITY_MATRIX.md`)

| Axis | Values |
| --- | --- |
| Model family | Qwen2, SmolLM2, TinyLlama, Phi, Gemma(2/3), BERT/MiniLM |
| Task | text-generation, feature-extraction, classification, RAG-embedding |
| PEFT | enumerated from `09.PEFT_REGISTRY` (LoRA, LoRA-XS, MARS, …) |
| Quantization | `09.QuantizationType` (none, QInt8, QUInt8, int4 where supported) |
| Merger variant | enumerated from `09.MERGER_REGISTRY` (resolved `MergerVariant`) |
| Export path | legacy Optimum, optimum-onnx, direct `torch.onnx` |
| Engine | native, genai |
| Device | Pixel 6, Galaxy S21 FE, emulator |
| Status | supported / experimental / blocked / not-tested |
| Evidence | link to CI run / issue / doc note |

The axes are **enumerated from the registries/enums in `00_code_plans/09`** (no hand-maintained lists that can drift from the code), and the matrix is generated from / cross-checked against `model_support_matrix.json` (`02_code_plans/02`), so it doesn't drift from reality. Model family + architecture coverage comes from `09.ARCHITECTURE_REGISTRY`.

## Implementation steps

1. Stand up `docs/` with stubs for each page; add a markdown link-check to CI (#28).
2. Fill each page as its contract locks (ARCHITECTURE + PUBLIC_API after #22/#23; MODEL_FORMAT after #12/#8; RAG after #24–#26; ANDROID_SDK after #29; EXPORT after #14).
3. Generate `COMPATIBILITY_MATRIX.md` from the support matrix; mark untested combos honestly as `not-tested`.
4. Create `RELEASE_CHECKLIST.md` + `CHANGELOG.md` skeletons (finalized in #31).
5. Keep `agent_docs/` clearly delineated as planning docs (note in README).

## Interactions

- **#22–#26 / #18 / #12 / #29 / #14**: each is the source of truth for one page.
- **`02_code_plans/02` (support matrix)**: feeds `COMPATIBILITY_MATRIX.md`.
- **#28 (CI)**: link-check + (optionally) snippet checks run on docs.
- **#31 (release)**: docs completeness is a checklist gate.

## Tests & smokes

- Markdown link-check passes for all `docs/` pages (and `agent_docs/` cross-refs).
- Every public API in `PUBLIC_API.md` resolves to a real symbol (spot-check).
- `COMPATIBILITY_MATRIX.md` rows have a status + evidence; no row left blank.
- Code snippets in docs compile/parse (where feasible) before release.
