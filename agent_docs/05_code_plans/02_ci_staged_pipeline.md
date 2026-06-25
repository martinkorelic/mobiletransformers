# Staged CI Pipeline

**Priority #29 | Prerequisites: #28 (`05_code_plans/01`), #1 (`00_code_plans/01`), #3 (`01_code_plans/06_source_built_ort_training_pipeline.md`) | Blocks: #32 (`05_code_plans/05`, release)**

> Runs alongside the tiers. The CI is the standing proof of "it actually works" — the edge over a smoke-test-only competitor.

## Purpose

Add a staged CI that compiles the core, validates packages, and runs an end-to-end smoke, **without** downloading large models on every PR. Stages are ordered cheapest-first so failures surface fast. The repo currently has **no CI config** (`.github/workflows` absent).

## Touched / new files

- NEW `.github/workflows/ci.yml` — fast checks + export smoke + Android assemble.
- NEW `.github/workflows/device.yml` — manual/scheduled device + model-zoo job.
- `Makefile` (#28) — CI calls `make` targets, not raw commands.
- `pyproject.toml` / `uv.lock` (#1) — pinned, reproducible install.

## Data contracts / interfaces

### Stage design

| Stage | Trigger | Content | Time budget |
| --- | --- | --- | --- |
| Fast checks | every PR | python import checks, `make test` (config/settings/manifest unit tests), Kotlin/JVM unit tests, markdown link check | minutes |
| Export smoke | every PR | `make test-smoke`: one tiny `generate_artifacts` (#3) + manifest validation + 1-token desktop smoke | < ~10 min |
| Android build | every PR | pre-rename `:ORTransformersMobile:assembleDebug` + `:app:assembleDebug`; post-rename `:MobileTransformers:assembleDebug` + `:MobileTransformersApp:assembleDebug` | medium |
| Device / zoo | manual + nightly | physical device: train 1 step → merge → generate 1 token → ingest/query 1 RAG doc; build starter zoo; record time/memory | long |

Use `matrix`, `fail-fast: false`, and per-job `timeout-minutes` (GitHub Actions workflow syntax). Do **not** make zoo builds mandatory on PRs. Where the Android matrix pins runtime libraries (e.g. WorkManager for #34), keep the pinned versions explicit in the workflow.

**Parity gate (F2).** The fast stage runs a `parity` job: the Pydantic models (`config/models.py`) + enums (`config/constants.py`) are the single source of truth, so the job regenerates `schemas/*.schema.json` + a golden `enums.json` from them (`python -m mobiletransformers.codegen.enums --check`) and **fails on drift** vs the checked-in Kotlin/C++ mirrors. The same fast stage also gates `make lint` + `make typecheck` (the #5 / #28 tooling).

### Rename transition

The Android assemble job must work **before and after** the `00_code_plans/04` module rename. Gate the task names on a repo flag / detect the module, or run both and allow one to be skipped, until the rename lands.

## Implementation steps

1. `ci.yml`: jobs (fast / export-smoke / android-build) wired to `make` targets; cache `uv`/Gradle. The fast job also runs the `parity` gate (F2: regenerate schemas + golden `enums.json`, fail on Kotlin/C++ drift) and `make lint` / `make typecheck`.
2. Respect dependency-profile isolation (#2): the export-smoke job uses the source-built ORT-training env from #3; the Android job is independent.
3. `device.yml`: `workflow_dispatch` + `schedule` (nightly); self-hosted or device-farm runner; emits time/memory artifacts in the `docs/mobile_evaluation.md` style.
4. Keep the markdown link check scoped (it can validate the `agent_docs` cross-references too).
5. On tags, optionally build the release AAR (#30).

## Interactions

- **#28 (Makefile)**: CI = `make` orchestration.
- **#3 (source-built ORT)**: export-smoke proves the training toolchain is alive.
- **#30 (AAR)**: tag builds publish artifacts.
- **#32 (release)**: green CI is a release-checklist gate.

## References

- `https://developer.android.com/jetpack/androidx/releases/work` — WorkManager versions for the Android build matrix (#34).

## Worked example

`ci.yml` stage skeleton — fast → export-smoke → android-assemble, `fail-fast: false`, per-job `timeout-minutes`:

```yaml
jobs:
  fast:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: make lint && make typecheck
      - run: python -m mobiletransformers.codegen.enums --check   # F2 parity gate
      - run: make test

  export-smoke:
    needs: fast
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - run: make test-smoke   # tiny generate_artifacts + manifest validate + 1-token desktop

  android-assemble:
    needs: fast
    strategy:
      fail-fast: false
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      # rename transition (#16): assemble both module names until the rename lands
      - run: ./gradlew :ORTransformersMobile:assembleDebug || true
      - run: ./gradlew :MobileTransformers:assembleDebug
```

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- A PR with a broken unit test fails the fast stage.
- The `parity` gate (F2) fails the fast stage if a checked-in Kotlin/C++ enum/schema mirror drifts from the regenerated golden.
- `make lint` / `make typecheck` failures fail the fast stage.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- A PR that breaks export fails the export-smoke stage with the tiny fixture.
- The Android assemble job succeeds for current module names and updates cleanly after the #16 rename (both `:ORTransformersMobile` and `:MobileTransformers` covered during transition).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- The nightly `device.yml` job produces train→merge→generate→RAG metrics artifacts on a real device/farm runner.

**Definition of done** — `ci.yml` runs the fast → export-smoke → android-assemble jobs with `fail-fast: false` and per-job `timeout-minutes`; no PR job downloads a large model (assert via job logs / cache policy); the `parity`, lint, and typecheck gates are wired; `device.yml` runs on `workflow_dispatch` + nightly schedule.
