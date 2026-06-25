# Documentation Set & Compatibility Matrix

**Priority #31 | Prerequisites: #23–#27 (`03_code_plans/*`), #19 (`02_code_plans/01`), #13 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`) | Blocks: #32 (`05_code_plans/05`, release)**

> Author each doc as its underlying contract stabilizes — not before, to avoid drift.

## Purpose

Produce the public user-facing documentation set under `docs/` (separate from `agent_docs/`, which stays research/planning) and a living compatibility matrix, so the framework is adoptable and citable. Each page is written when its contract is locked, sourced from the corresponding code plan.

## Touched / new files

- NEW `docs/ARCHITECTURE.md` — training/export/runtime data flow; native graph I/O contract from `03_code_plans/01`.
- NEW `docs/PUBLIC_API.md` — Kotlin + Python public API; HF mapping table from `03_code_plans/02` and `03_tier2...md`. Per **F5**, the Python side documents exactly the importable surface declared in `mobiletransformers.__all__` (owned by `00_code_plans/10`), alongside the Kotlin facade + CLI names — these three together are the SemVer-governed public surface (#32).
- NEW `docs/MODEL_FORMAT.md` — the `mobiletransformers_manifest.json` + `weight_handoff_map.json` contract (`00_code_plans/06`, `00_code_plans/07`).
- NEW `docs/CONFIGURATION.md` — the public config contract from `00_code_plans/09`: the enum vocabulary (mirrored Python ↔ Kotlin), the Pydantic config models + their generated `schemas/*.schema.json` (the cross-boundary JSON contract for `training_config.json`/`generation_config.json`/`rag_config.json`), and the PEFT/architecture/merger **registries** as the public extension points ("to add a method/architecture/merger, add a registry entry"). Source of truth: `00_code_plans/09`.
- NEW `docs/ANDROID_SDK.md` — Gradle/AAR setup, permissions, ABI, local Maven (`05_code_plans/03`).
- NEW `docs/RAG.md` — embedding model, ingestion, vector store semantics (the `1 - score`, 8-dim constraint), retrieval, grounded generation (`03_code_plans/03`–`05`).
- NEW `docs/EXPORT.md` — one-command export + toolchain notes (`02_code_plans/05`, `00_code_plans/03`).
- NEW `docs/COMPATIBILITY_MATRIX.md` — the matrix below.
- NEW `docs/RELEASE_CHECKLIST.md` + `CHANGELOG.md` (owned/finalized by #32; created here).
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

The axes are **enumerated from the registries/enums in `00_code_plans/09`** (no hand-maintained lists that can drift from the code). Per **F6**, `model_support_matrix.json` (`02_code_plans/02`) is the generated source of truth → `docs/COMPATIBILITY_MATRIX.md` is **rendered FROM it** (not hand-maintained) → a per-package manifest declares only that package's realized capabilities; don't hand-maintain the three. Model family + architecture coverage comes from `09.ARCHITECTURE_REGISTRY`.

## Implementation steps

1. Stand up `docs/` with stubs for each page; add a markdown link-check to CI (#29).
2. Fill each page as its contract locks (ARCHITECTURE + PUBLIC_API after #23/#24; MODEL_FORMAT after #13/#9; RAG after #25–#27; ANDROID_SDK after #30; EXPORT after #15).
3. Generate `COMPATIBILITY_MATRIX.md` from the support matrix; mark untested combos honestly as `not-tested`.
4. Create `RELEASE_CHECKLIST.md` + `CHANGELOG.md` skeletons (finalized in #32).
5. Keep `agent_docs/` clearly delineated as planning docs (note in README).

## Interactions

- **#23–#27 / #19 / #13 / #30 / #15**: each is the source of truth for one page.
- **`02_code_plans/02` (support matrix)**: feeds `COMPATIBILITY_MATRIX.md`.
- **#29 (CI)**: link-check + (optionally) snippet checks run on docs.
- **#32 (release)**: docs completeness is a checklist gate.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- Markdown link-check passes for all `docs/` pages (and `agent_docs/` cross-refs) — wired into CI (#29).
- Every public API in `PUBLIC_API.md` resolves to a real symbol (spot-check the Python `__all__` members, Kotlin facade, and CLI names — F5).
- `COMPATIBILITY_MATRIX.md` rows have a status + evidence; no row left blank.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `COMPATIBILITY_MATRIX.md` regenerates from `model_support_matrix.json` and matches the committed copy (F6: rendered, not hand-edited; fails on drift).
- Code snippets in docs compile/parse (where feasible) before release.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Full read-through of each page once its underlying contract locks, to confirm it matches the shipped behaviour.

**Definition of done** — every `docs/` page is filled when its contract locks (no premature drift), the link-check is green, `PUBLIC_API.md` covers the `__all__` + Kotlin facade + CLI surface, `COMPATIBILITY_MATRIX.md` is rendered from `model_support_matrix.json` with every row carrying status + evidence, and `RELEASE_CHECKLIST.md` + `CHANGELOG.md` skeletons exist for #32.
