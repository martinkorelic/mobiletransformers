# Staged CI Pipeline

**Priority #28 | Prerequisites: #27 (`05_code_plans/01`), #1 (`00_code_plans/01`), #3 (`01_code_plans/06_source_built_ort_training_pipeline.md`) | Blocks: #31 (`05_code_plans/05`, release)**

> Runs alongside the tiers. The CI is the standing proof of "it actually works" — the edge over a smoke-test-only competitor.

## Purpose

Add a staged CI that compiles the core, validates packages, and runs an end-to-end smoke, **without** downloading large models on every PR. Stages are ordered cheapest-first so failures surface fast. The repo currently has **no CI config** (`.github/workflows` absent).

## Touched / new files

- NEW `.github/workflows/ci.yml` — fast checks + export smoke + Android assemble.
- NEW `.github/workflows/device.yml` — manual/scheduled device + model-zoo job.
- `Makefile` (#27) — CI calls `make` targets, not raw commands.
- `pyproject.toml` / `uv.lock` (#1) — pinned, reproducible install.

## Data contracts / interfaces

### Stage design

| Stage | Trigger | Content | Time budget |
| --- | --- | --- | --- |
| Fast checks | every PR | python import checks, `make test` (config/settings/manifest unit tests), Kotlin/JVM unit tests, markdown link check | minutes |
| Export smoke | every PR | `make test-smoke`: one tiny `generate_artifacts` (#3) + manifest validation + 1-token desktop smoke | < ~10 min |
| Android build | every PR | pre-rename `:ORTransformersMobile:assembleDebug` + `:app:assembleDebug`; post-rename `:MobileTransformers:assembleDebug` + `:MobileTransformersApp:assembleDebug` | medium |
| Device / zoo | manual + nightly | physical device: train 1 step → merge → generate 1 token → ingest/query 1 RAG doc; build starter zoo; record time/memory | long |

Use `matrix`, `fail-fast: false`, and per-job `timeout-minutes` (GitHub Actions workflow syntax). Do **not** make zoo builds mandatory on PRs.

### Rename transition

The Android assemble job must work **before and after** the `00_code_plans/04` module rename. Gate the task names on a repo flag / detect the module, or run both and allow one to be skipped, until the rename lands.

## Implementation steps

1. `ci.yml`: three jobs (fast / export-smoke / android-build) wired to `make` targets; cache `uv`/Gradle.
2. Respect dependency-profile isolation (#2): the export-smoke job uses the source-built ORT-training env from #3; the Android job is independent.
3. `device.yml`: `workflow_dispatch` + `schedule` (nightly); self-hosted or device-farm runner; emits time/memory artifacts in the `docs/mobile_evaluation.md` style.
4. Keep the markdown link check scoped (it can validate the `agent_docs` cross-references too).
5. On tags, optionally build the release AAR (#29).

## Interactions

- **#27 (Makefile)**: CI = `make` orchestration.
- **#3 (source-built ORT)**: export-smoke proves the training toolchain is alive.
- **#29 (AAR)**: tag builds publish artifacts.
- **#31 (release)**: green CI is a release-checklist gate.

## Tests & smokes

- A PR with a broken unit test fails the fast stage.
- A PR that breaks export fails the export-smoke stage with the tiny fixture.
- The Android assemble job succeeds for current module names; updates cleanly after rename.
- The nightly device job produces train→merge→generate→RAG metrics artifacts.
- No PR job downloads a large model (assert via job logs / cache policy).
