# Cross-cutting release — 05_code_plans audit

> ## ⚠️ SNAPSHOT — 2026-08-07, at HEAD `54e0a8e`. NOT a live defect list. (Banner added 2026-08-14.)
>
> This audit is a **point-in-time photograph**, and it carries **no closure annotations of its own** —
> nothing in this file was ever struck out as findings were fixed. Six later cycles of work landed on
> top of it (the 2026-08-07 remediation pass, then 08-08 / 08-09 / 08-10 device acceptance, then the
> 08-14 cleaning phase). **Reading it as a to-do list generates phantom work**, which is the specific
> failure this banner exists to prevent.
>
> Spot-verification on 2026-08-14 found the audits materially **over-report** what is open. Every one
> of these, recorded here as a defect, is fixed in the tree with the fix documented at the site:
>
> | Audit finding | Where it is fixed |
> | --- | --- |
> | #21 "installer deletes live cache before rename" | `ModelPackageInstaller.kt:47-75` — renames aside, publishes, rolls back on failure ("#21 crash safety") |
> | #17 "`ORT*` leak in public `TrainingResult`" | `runtime/Results.kt:25-26` records the retype |
> | #27 "config override applies only on the FIRST retrieve" | `RagRepository.kt:36` — "A changed config now always applies" |
> | #24 "GenAI carries its own private method map with silent-greedy fallback" | `ORTGeneratorGenAI.kt:78` uses the shared `SamplingMethod.fromWire(...).nativeOrdinal` |
> | #26 "`maxTextLength` silently dropped" | threaded through `ConfigMappers.kt:135` / `ORTRagConfig.kt:45` |
> | #34 "`ORTScheduler.kt` TODO still open" | fixed; `ORTScheduler.kt:161-162` records it |
> | #25 "`SearchType` String→enum swap never landed" | done 2026-08-07 |
> | #6 "grep-guard DoD fails, `build_adapter_mapping` missing" | done 2026-08-07 (see #6's self-check) |
> | #22 "Mode-1 never writes `adapter_model.safetensors`" | fixed; see #22's self-check |
> | #15 "`--validate` missing entirely" | it exists |
>
> **The authoritative list of what is actually open is `agent_docs/HANDOFF.md`**, whose numbers are
> re-measured each cycle. Use this file for its *reasoning* — why a finding mattered, what the failure
> shape was — not for its verdicts.


Audit date 2026-08-07. Branch `restructure`, HEAD `54e0a8e "Restructure: Code complete phases 1-29"`.
**Tree state note:** `git status` is *clean* and every artifact in scope (`Makefile`, `scripts/`,
`.github/workflows/*`, `docs/*`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE.md`, `pyproject.toml`) is
**tracked in HEAD** (`git ls-tree -r HEAD`). So the "nothing is committed yet" premise no longer holds —
the restructure *is* committed. The commit message claim "code complete phases 1-29" is **accurate for
#28** and **optimistic for #29** (see below); #30/#31/#32 are outside its claim and are indeed incomplete.

`make check` (= `lint` + `typecheck` + `parity` + `test`) was run locally and is **green**:
`215 passed, 10 skipped` in 1.05s. That is real, verified evidence for the #29 fast stage.

## Summary table

| # | Plan | Claimed | Verified | % | Verdict |
| --- | --- | --- | --- | --- | --- |
| 28 | Makefile & CLI entrypoints | `[x]` done 2026-07-14 | Substantially TRUE | **90%** | Done; acceptance *tests* for `help`/`clean-generated` never written; README happy-path undocumented |
| 29 | Staged CI pipeline | `[x]` done 2026-07-14 | PARTIAL — over-claimed | **70%** | fast+export-smoke real & green; android job self-skips forever; no JVM tests, no link-check, no tag-AAR; `device.yml` body is an `echo` |
| 30 | AAR & Maven publication | not started | NOT STARTED (confirmed) | **10%** | Zero `maven-publish` anywhere; both scripts `exit 1`; no `examples/consumer-app/`. Only step 1 (native inputs) is locally satisfied |
| 31 | Docs set & compat matrix | `[ ]` partial | PARTIAL, and now *stale* | **65%** | 8/10 pages exist; `ARCHITECTURE.md` + `ANDROID_SDK.md` missing; `RAG.md`/`PUBLIC_API.md` drifted behind #26/#27/#17/#19/#35 |
| 32 | Versioning, license, release | not started | NOT STARTED (confirmed) | **5%** | Still CC-BY-NC-4.0, no SPDX, no `THIRD_PARTY_NOTICES.md`, no license in `pyproject`, version sites disagree, 0 tags |

**Tier estimate: ~48% complete.**

---

## Per-plan findings

### #28 — Makefile & CLI entrypoints (`05_code_plans/01`)

**Required:** root `Makefile` with the 11 canonical targets, each ≤ ~3 lines and a thin wrapper over the
`mobiletransformers` CLI or Gradle; `.PHONY` + `##` self-documentation; profile-isolated `setup*` honoring
#2 conflicts; non-destructive `clean-generated`; `scripts/{android_build_aar,publish_local_maven,run_smoke}.sh`;
happy path documented in `README.md` + `docs/EXPORT.md`.

**Verified present:**
- `Makefile:8-10` — `.PHONY` lists 22 targets; **all 22** carry `##` docs and appear in `make help`
  (`Makefile:20-22`, run confirmed, exit 0).
- Thin wrappers confirmed: `Makefile:65` (`mobiletransformers export …`), `Makefile:68` (`package-model`),
  `Makefile:72` (`$(GRADLE) :MobileTransformers:assembleDebug :app:assembleDebug`), `Makefile:18` defines
  `GRADLE := cd android/MobileTransformersApp && ./gradlew`. No target exceeds 3 lines.
- Profile isolation: `Makefile:25-35` — `setup` = `uv sync --frozen --group dev`; `setup-export` =
  `--extra export`; `setup-train` = `--python 3.12 --group ort-training-local`; `setup-genai` =
  `--group genai-smoke`. Each syncs its own env; never combines the colliding pairs (matches
  `pyproject.toml:26-70` `[tool.uv] conflicts` intent).
- `clean-generated` (`Makefile:90-92`) removes only `build/ dist/ onnx_models/ *.egg-info src/*.egg-info`
  plus `-maxdepth 2` `__pycache__`. **No `cache_dir/`, no `$HOME`, no absolute paths** — non-destructive
  by inspection.
- Console script exists: `pyproject.toml:23-24` `mobiletransformers = "mobiletransformers.cli.main:main"`,
  dispatcher real at `src/mobiletransformers/cli/main.py:25`.
- `scripts/run_smoke.sh:9` — real (`exec make test-smoke`).
- Bonus beyond plan: `device-package` / `device-test` targets (`Makefile:74-78`) backed by a **real**
  `scripts/device_package.sh` (81 lines, export → reshape → `adb push`).

**Gaps:**
- (b) implemented-but-untested — the plan's own unit acceptance ("`make help` lists every target
  (grep-checkable)", "`make clean-generated` removes only generated artifacts, assert `cache_dir/`
  untouched") has **no automated test**: `grep -rl "clean-generated\|Makefile" tests/` returns nothing.
  Both hold by inspection, neither is guarded against regression.
- (a) genuinely missing — implementation step 5, "document the happy path in `README.md`": `README.md`
  contains **zero** `make` references (`grep -n "make " README.md` → empty). `docs/EXPORT.md:51-58` does
  document the `setup*` matrix, so only the README half is missing.
- (d) deferred with owner — `scripts/android_build_aar.sh:11-13` and `scripts/publish_local_maven.sh:10-12`
  are fail-closed stubs (`exit 1` after printing the interim manual command). This is **exactly what the
  plan specifies** ("bodies owned by #30"), so it is not a #28 defect.
- `docs` target (`Makefile:86-87`) is an `echo` no-op; no `mkdocs.yml` exists although
  `pyproject.toml:39` declares a `docs = ["mkdocs>=1.6"]` group. Deferred to #31 by the target's own text.

**Drift / doubtful claims:** none material. The HANDOFF `[x]` for #28 is honest.

---

### #29 — Staged CI pipeline (`05_code_plans/02`)

**Required:** `ci.yml` with fast → export-smoke → android-assemble, `fail-fast: false`, per-job
`timeout-minutes`, CI calling `make` targets; fast stage = lint + typecheck + parity (F2) + unit tests +
**Kotlin/JVM unit tests** + **markdown link check**; export-smoke = tiny `generate_artifacts` (#3) +
manifest validation + **1-token desktop smoke**, using the source-built ORT-training env; android job works
before *and* after the rename; `device.yml` on `workflow_dispatch` + nightly `schedule`; optional tag AAR
build; no PR job downloads a large model.

**Verified present:**
- `.github/workflows/ci.yml:24-87` — three jobs, `needs: fast` chain (`:46`, `:62`), `timeout-minutes`
  10/15/25 (`:28,:48,:64`), `fail-fast: false` (`:65-66`).
- Fast job calls the Makefile, not raw commands: `ci.yml:36,38,40,42` → `make lint`, `make typecheck`,
  `make parity`, `make test`. Locally verified green (215 passed / 10 skipped).
- `make parity` (`Makefile:50`) → `python -m mobiletransformers.codegen.enums --check` — the F2 gate is
  genuinely wired and passes.
- Export-smoke syncs the export profile under 3.12 (`ci.yml:56`) then `make test-smoke` (`ci.yml:58`).
- `device.yml:12-16` — `workflow_dispatch` + nightly `cron: '17 3 * * *'`; `runs-on: [self-hosted,
  android-device]` (`:21`), 120-min timeout, artifact upload (`:40-47`).
- No PR job pulls a large model — `make test`/`make test-smoke` (`Makefile:53,56`) run only
  fixture-scoped suites.

**Gaps:**
- (a) genuinely missing — **markdown link check** (implementation step 4, and #31's acceptance gate) is
  absent from `ci.yml` entirely. `grep -i "link" .github/workflows/*.yml` → nothing.
- (a) genuinely missing — **Kotlin/JVM unit tests** are in the plan's fast stage but no job runs
  `testDebugUnitTest`. A substantial JVM suite exists
  (`android/MobileTransformersApp/MobileTransformers/src/test/java/com/martinkorelic/mobiletransformers/`
  → `facade/`, `hub/`, `packages/`, `rag/`, `runtime/`, `training/`, `internal/`). These likely do **not**
  need the vendored native deps (unit-test compile does not trigger `externalNativeBuild` link tasks),
  so this is the cheapest recoverable CI win in the whole tier.
- (a) genuinely missing — **tag-triggered AAR build** (implementation step 5): `ci.yml:19-22` triggers only
  on `push` to `main`/`restructure` and `pull_request`; no `tags:` filter, no release job.
- (c) host-done, external leg outstanding — **android-assemble never actually runs.** `ci.yml:74-83` gates
  on `[ -d MobileTransformers/src/main/aarLibs ] && [ -d …/jniLibs ]`; `aarLibs/` is git-ignored
  (`android/MobileTransformersApp/MobileTransformers/.gitignore:39`) and `git ls-files` shows **no tracked
  files** under either dir, so on any hosted runner the condition is permanently false and the job emits a
  warning and exits green. The "Android build works before and after the rename" acceptance is therefore
  **unproven**, not proven.
- (a) drift — the gate is also **stale**: `build.gradle.kts:77-80` documents that the genai **AAR is no
  longer a dependency** (the `.so` ships from `jniLibs`), so gating on `aarLibs/` presence tests a dir the
  build no longer consumes. The real missing inputs on a bare runner are `src/main/jniLibs/{arm64-v8a,
  x86_64}` (ignored at `.gitignore:36-37`) and `src/main/cpp/includes/` (ignored at `:38`).
- (a) drift — **export-smoke does not do what the plan says.** Plan: "one tiny `generate_artifacts` (#3)
  + manifest validation + 1-token desktop smoke". Actual: `Makefile:56` → `pytest tests/export tests/hub
  tests/cli -q`. `generate_artifacts` exists only in `tests/integration/test_ort_training_smoke.py`, which
  is **not** in `make test-smoke` and runs only via the `workflow_dispatch`-only
  `ort-training-smoke.yml:14`. There is no 1-token desktop smoke anywhere. Implementation step 2 ("the
  export-smoke job uses the source-built ORT-training env from #3") is **not implemented** — it uses the
  `export` extra instead.
- (a) `device.yml:36-39` — the end-to-end step is `echo "Device e2e harness: …"`. It is a workflow shell,
  not a harness; nothing is executed. `Upload metrics artifacts` uploads only `adb-devices.txt`.
- (d) deferred — `ort-training-smoke.yml:1-9` documents the wheel-provisioning question as owned by #29
  and left open. #29 did not close it.

**Drift / doubtful claims:** the `[x]` in `IMPLEMENTATION_ORDER.md:89` is over-claimed. Two of three CI
stages are real; the third is a permanent no-op, and two named fast-stage contents (JVM tests, link check)
plus the entire export-smoke *content contract* were never implemented. Honest status: **partial**.

---

### #30 — AAR & local-Maven publication (`05_code_plans/03`) · checkpoint

**Required:** resolve missing native inputs; apply `maven-publish` to `:MobileTransformers`; coordinates
`com.martinkorelic.mobiletransformers:mobiletransformers-android:<version from -Pversion>`; POM metadata
incl. Apache-2.0 license; ABI packaging for `arm64-v8a` + `x86_64`; real bodies for
`scripts/android_build_aar.sh` + `scripts/publish_local_maven.sh`; `examples/consumer-app/`;
`publishToMavenLocal` succeeds and an external consumer app resolves it.

**Verified present:**
- Step 1 (native inputs) is **locally satisfied**: `src/main/aarLibs/onnxruntime-genai.aar` and
  `src/main/jniLibs/{arm64-v8a,x86_64}/*.so` exist on this machine, and the dangling
  `implementation(files("./src/main/aarLibs/onnxruntime-genai.aar"))` reference is **gone** —
  `build.gradle.kts:77-80` replaces it with a comment explaining the `.so`-from-`jniLibs` decision. The
  plan's "missing-native regression: no dangling `files(...)` reference" acceptance is met.
- ABI filters configured: `build.gradle.kts:35-37` `abiFilters += listOf("arm64-v8a", "x86_64")`.
- `packaging { jniLibs { pickFirsts … } }` at `build.gradle.kts:57-64` dedupes the genai `.so`s.

**Gaps:**
- (a) **No publishing block of any kind.** `grep -rn "maven-publish|publishing|publishToMavenLocal|groupId|
  artifactId" android/ --include=*.kts --include=*.toml --include=*.properties` (excluding `build/`)
  returns **zero hits**. `MobileTransformers/build.gradle.kts:1-5` applies only
  `android.library` + `kotlin.android` + `objectbox`.
- (a) **No version/group on the library module at all.** `gradle.properties` (23 lines) has no `version`
  or `group`; root `build.gradle.kts` (6 lines) sets neither; only the *sample app* carries
  `versionName = "1.0"` (`app/build.gradle.kts:18`), which is not the SDK artifact version. There is no
  `-Pversion` plumbing for #32 to drive.
- (a) both scripts are fail-closed stubs: `scripts/android_build_aar.sh:11-13` and
  `scripts/publish_local_maven.sh:10-12` print and `exit 1`.
- (a) `examples/consumer-app/` does **not exist** (`ls examples` → no such directory). The #30 workflow
  checkpoint (external consumer app resolves from `mavenLocal()`) is entirely unstarted.
- (a) no POM metadata, hence no place for the Apache-2.0 license the plan requires (blocked on #32 anyway).
- (a) stale leftover: `build.gradle.kts:13` `jniLibs.srcDirs("libs")` points at a module-root `libs/`
  directory that does not exist (module root has no `libs`). Harmless (AGP tolerates it, and `srcDirs`
  *adds* to the default `src/main/jniLibs`) but it is the last remnant of the plan's "missing `libs` JNI
  dir" note and should be deleted or repointed.

**Drift / doubtful claims:** none — the `[ ]` is honest. Note the plan's premise "the genai AAR must be
restored" was resolved by a *different* decision (drop the AAR dependency, ship the patched `.so`); the
plan text is stale on that point, and the third-party-AAR question it raises ("not transitively published
via mavenLocal") is consequently *simplified*, not eliminated — the `.so`s must be vendored into the
published AAR, and nothing does that yet.

---

### #31 — Documentation set & compatibility matrix (`05_code_plans/04`)

**Required (10 artifacts):** `docs/ARCHITECTURE.md`, `PUBLIC_API.md`, `MODEL_FORMAT.md`,
`CONFIGURATION.md`, `ANDROID_SDK.md`, `RAG.md`, `EXPORT.md`, `COMPATIBILITY_MATRIX.md` (rendered from
`model_support_matrix.json`, every row with status + evidence), `RELEASE_CHECKLIST.md`, `CHANGELOG.md`;
markdown link-check in CI; `PUBLIC_API.md` covers `__all__` + Kotlin facade + CLI; README notes the
`agent_docs/` delineation.

**Verified present (8/10):**
- `docs/EXPORT.md` (3.7 KB), `docs/RAG.md` (3.2 KB), `docs/PUBLIC_API.md` (2.7 KB),
  `docs/MODEL_FORMAT.md` (8.5 KB), `docs/CONFIGURATION.md` (4.9 KB), `docs/COMPATIBILITY_MATRIX.md`
  (1.5 KB), `docs/RELEASE_CHECKLIST.md` (1.1 KB), `CHANGELOG.md` (863 B).
- F6 satisfied: `docs/COMPATIBILITY_MATRIX.md:3` declares it generated from `model_support_matrix.json`;
  renderer is `src/mobiletransformers/support/render.py`; drift guarded by `tests/support/test_render.py`
  (2 tests, passing in the run above). Every row (`:18-20`) carries status glyphs **and** an evidence /
  blocker string — no blank rows.
- `CONFIGURATION.md:8-27` enumerates the 10 mirrored enums against `config/constants.py` and names the
  parity gate; `MODEL_FORMAT.md:1-20` documents both JSON contracts + `check_compat()` versioning.
- `CHANGELOG.md:1-5` uses Keep-a-Changelog format with an `## [Unreleased]` section.

**Gaps:**
- (a) **`docs/ARCHITECTURE.md` MISSING.** Its blockers (#23 `01_inference_handoff_alignment`, #24
  `sampling_and_streaming`) are both **code-complete as of 2026-07-15**
  (`IMPLEMENTATION_ORDER.md:78-79`), so this page is now *unblocked and overdue* — it is no longer a
  legitimate "deferred until the contract locks".
- (a) **`docs/ANDROID_SDK.md` MISSING.** Legitimately blocked on #30 (Gradle/AAR/local-Maven setup), which
  is not started. Category (d) with a documented owner.
- (a) **`docs/RAG.md` is now factually wrong.** `RAG.md:7-11` states ingestion/chunking (#26) and grounded
  generation + `RagConfig` (#27) "are not yet implemented" — but both are code-complete per
  `IMPLEMENTATION_ORDER.md` (`DocumentChunker`, `IngestionPipeline`, `PromptAssembler`,
  `facade.generateWithRag`). This is exactly the drift the plan's "author each doc as its contract
  stabilizes" rule was meant to prevent, in the opposite direction.
- (a) **`PUBLIC_API.md` incomplete on two axes.** (1) `PUBLIC_API.md:7-8` defers the Kotlin facade section
  to #17/#19 — both code-complete 2026-07-15, so overdue. (2) The CLI table (`:34-43`) omits the
  `federated` subcommand, which **is** registered (`cli/main.py:16,25`, and in the parser metavar at
  `:37`). F5's "every public API in PUBLIC_API.md resolves to a real symbol" holds, but the converse
  (every real CLI name is documented) does not.
- (a) markdown link-check not wired into CI (owned by #29, absent there too). In practice `docs/*.md`
  contain **no relative cross-links at all** (only two image links in `README.md:10,100`), so a link-check
  would be trivially green — but it also means the docs set is not interlinked/navigable.
- (a) implementation step 5 — README does not delineate `agent_docs/` as planning docs; `README.md` has no
  reference to `agent_docs/` at all.
- (b) no `mkdocs.yml` despite the `docs` dependency group; `make docs` is a stub echo.

**Drift / doubtful claims:** the `partial` label is correct, but the HANDOFF rationale ("remaining pages
await #23/#24/#30") is **out of date** — only `ANDROID_SDK.md` still has a live blocker.

---

### #32 — Versioning, license & v1.0 release (`05_code_plans/05`) · checkpoint

**Required:** relicense to Apache-2.0; SPDX headers on first-party source only (with an explicit vendored
exclusion list); `THIRD_PARTY_NOTICES.md`; `pyproject.toml` license expression + classifier; reconcile
`CITATION.cff` version/date to the tag; fill `CHANGELOG.md` v1.0.0 with non-goals; bump all five version
sites together and assert them equal; `.gitignore` `agent_docs/` decision; annotated `v1.0.0` tag; full
release gate green.

**Verified present:**
- `docs/RELEASE_CHECKLIST.md:6-20` — the gate list exists (created by #31) and correctly enumerates CI,
  parity, AAR/consumer, docs, version sites, license/SPDX/`THIRD_PARTY_NOTICES.md`, CHANGELOG. Every box
  is unticked, which is accurate.
- `.gitignore` decision **made**: `agent_docs/` is no longer ignored (`.gitignore` has 26 lines, none
  matching `agent_docs`; `git check-ignore agent_docs` → not ignored; the dir is tracked in HEAD). One
  checklist item genuinely closed.

**Gaps:**
- (a) **License unchanged.** `LICENSE.md:1` is still `# Attribution-NonCommercial 4.0 International`
  (CC-BY-NC-4.0). The Apache-2.0 target (`IMPLEMENTATION_ORDER.md:16`) is not applied.
- (a) **Zero SPDX headers.** `grep -rln "SPDX-License-Identifier" src/ scripts/ android/.../java` → no
  hits. Neither the positive pass nor the negative (vendored-dirs-untouched) assertion exists as a test.
- (a) **`THIRD_PARTY_NOTICES.md` does not exist.** Vendored trees needing enumeration are present and
  substantial: `…/src/main/cpp/{onnxruntime,onnxruntime-genai,tokenizers,proto}` plus
  `jniLibs/*/lib{protobuf-lite,tokenizers_c,tokenizers_cpp}.a`.
- (a) `pyproject.toml:21` still carries the deferral comment
  `# license intentionally omitted until the licensing decision (target: Apache-2.0).` — no `license`
  expression, no classifier. #32's stated job of "closing the loop" on #1's deferral is unstarted.
- (a) **Version sites disagree** (table below) and there is **no invariant test**. The plan requires
  `__version__ == importlib.metadata.version("mobiletransformers")`; what exists is
  `tests/unit/test_import_compat.py:8` asserting a **hardcoded** `== "0.1.0"` — it would pass even if the
  installed distribution reported something different, so it does not implement the invariant.
- (a) `CHANGELOG.md` has only `## [Unreleased]` (`:10`); no `## [1.0.0]` section, and the "Non-goals"
  block (`:19-20`) covers a device-parity note rather than the plan's required non-goals (GPU/NPU
  training, multimodal training, "leaner trainer" race).
- (a) **No git tag at all** (`git tag -l` empty). `git describe --tags` acceptance cannot pass.
- (c/manual) rights-holder relicense sign-off (two authors in `CITATION.cff:3-7`) is a values decision, not
  engineering — outstanding and unflagged in any tracked file other than the checklist.
- (c) "≥1 starter model package published OR documented how to build it" — `docs/EXPORT.md` documents the
  build path, so this item is arguably satisfiable on paper; nothing published.

**Drift / doubtful claims:** none — nothing is claimed. Note `CITATION.cff` is *actively wrong today*: it
declares `version: 1.0.0` / `date-released: 2025-10-18` for a repo whose package version is `0.1.0` with no
tag, i.e. it advertises a release that does not exist.

---

## Version-site consistency table

| Site | Evidence | Value | Agrees? |
| --- | --- | --- | --- |
| `pyproject.toml` `[project] version` | `pyproject.toml:7` | `0.1.0` | baseline |
| `src/mobiletransformers/__init__.py` `__version__` | `src/mobiletransformers/__init__.py:26` | `"0.1.0"` (hardcoded, not read via `importlib.metadata`) | ✅ value matches, ❌ mechanism |
| Gradle library `version` / `-Pversion` | **absent** — no `version`/`group` in `MobileTransformers/build.gradle.kts`, `gradle.properties`, or root `build.gradle.kts` | *(none)* | ❌ site does not exist |
| Android **sample app** `versionName` | `app/build.gradle.kts:18` | `"1.0"` | ❌ (unrelated to the SDK artifact, but visually contradictory) |
| `CITATION.cff` `version` / `date-released` | `CITATION.cff:9-10` | `1.0.0` / `2025-10-18` | ❌ **conflicts** with `0.1.0`, and the date predates any release |
| `CHANGELOG.md` | `CHANGELOG.md:10` | `[Unreleased]` only | ❌ no versioned entry |
| Git tag | `git tag -l` → empty | *(none)* | ❌ |
| `mobiletransformers_manifest.json` `mobiletransformersVersion` | `src/mobiletransformers/export/pipeline.py:333` → `_pkg_version("mobiletransformers") or "0.0.0"` | derived from the installed dist | ✅ correctly derived (the one site wired the right way) |

**Verdict:** 3 of the 5 plan-mandated sites are wrong or missing. The manifest site is the only one
implemented per the plan's "single write-site" rule. A version-agreement test does not exist.

## License status vs Apache-2.0 target

| Item | Target (#32 / `IMPLEMENTATION_ORDER.md:16`) | Actual | Status |
| --- | --- | --- | --- |
| `LICENSE.md` | Apache-2.0 | CC-BY-NC-4.0 (`LICENSE.md:1`) | ❌ NOT STARTED |
| SPDX headers, first-party only | present on `src/`, Kotlin, first-party C++, CMake, shell | zero occurrences repo-wide | ❌ NOT STARTED |
| Vendored-code exclusion list | explicit list maintained | none; no negative test | ❌ NOT STARTED |
| `THIRD_PARTY_NOTICES.md` | exists, referenced from `LICENSE.md` | does not exist | ❌ NOT STARTED |
| `pyproject.toml` license expr + classifier | `license = "Apache-2.0"` + classifier | deferral comment at `pyproject.toml:21` | ❌ NOT STARTED |
| AAR POM license | Apache-2.0 in the publication block | no publication block exists (#30) | ❌ blocked on #30 |
| README license section | — | README has no license mention at all | ⚠️ minor |
| Rights-holder agreement | both authors sign off | not recorded anywhere | ❌ manual/external |

The non-commercial clause remains the single hardest release blocker: it is incompatible with the
"consumable AAR on Maven" positioning that #30 exists to deliver.

## CI native-dep provisioning blocker

**Current state: OPEN, unchanged since 2026-07-14, and now blocking three plans.**

The three git-ignored inputs a hosted runner cannot obtain:

| Input | Ignore rule | Present locally? | Needed by |
| --- | --- | --- | --- |
| `…/MobileTransformers/src/main/jniLibs/{arm64-v8a,x86_64}` (`libonnxruntime.so`, `libonnxruntime-genai*.so`, `libort_gen.so`, `libtokenizers_*.a`, `libprotobuf-lite.a`, `libobjectbox-jni.so`) | `MobileTransformers/.gitignore:36-37` | yes (10 + 4 files) | Android assemble, AAR (#29/#30) |
| `…/src/main/cpp/includes/` (protobuf/google headers) | `MobileTransformers/.gitignore:38` | yes | CMake `target_include_directories` (`cpp/CMakeLists.txt:42`) |
| `…/src/main/aarLibs/onnxruntime-genai.aar` | `MobileTransformers/.gitignore:39` | yes | **no longer a build input** (`build.gradle.kts:77-80` dropped the dependency) — but `ci.yml:78` still gates on it |
| `third_party/wheels/onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl` | `.gitignore:21` | yes | `ort-training-smoke.yml`, `make test-train`, real export-smoke content |

`git ls-files` confirms **nothing** under `aarLibs/` or `jniLibs/` is tracked, so on GitHub-hosted runners
all of it is absent. Both affected workflows respond by self-skipping rather than failing:
`ci.yml:74-83` (Android) and `ort-training-smoke.yml:25-33` (wheel). The result is a CI that is *green by
construction* on exactly the legs that most need proving.

**No decision has been recorded** among the three candidate options the plans name (rebuild-in-CI /
private index or package registry / cached-artifact restore). Nothing in the tree — no
`third_party/README`, no workflow step, no doc page — commits to one.

**Plans blocked:** #29 (android-assemble + a real export-smoke are unprovable), #30 (the AAR cannot be
built or published in CI, and the `.so`s must be vendored *into* the AAR for consumers), #32 (the release
checklist's "CI green (fast + export-smoke + android-build)" line can never be honestly ticked while the
android leg is a no-op).

**Cheapest partial unblock available today:** run the Kotlin/JVM unit tests
(`:MobileTransformers:testDebugUnitTest`) in CI — that suite compiles Kotlin only and should not need the
native link inputs, converting a large existing test body from "runs on the author's laptop" to "runs on
every PR".

## What stands between now and the #32 v1.0 release gate (enumerated)

1. **License relicense to Apache-2.0** — replace `LICENSE.md`; obtain both rights-holders' agreement
   (values decision, must be flagged to the user).
2. **SPDX pass** over first-party source only, with an explicit exclusion list for
   `cpp/{onnxruntime,onnxruntime-genai,tokenizers,proto}`; add the positive + negative tests.
3. **Write `THIRD_PARTY_NOTICES.md`** enumerating vendored ORT / ORT-GenAI / tokenizers / protobuf /
   nlohmann-json / ObjectBox components with their licenses; reference it from `LICENSE.md`.
4. **Set `pyproject.toml` license expression + classifier** (closes `00_code_plans/01`'s deferral at
   `pyproject.toml:21`).
5. **Create the Gradle version site** — `version`/`group` on `:MobileTransformers` driven by `-Pversion`;
   today no such property exists anywhere.
6. **Reconcile the five version sites** and add the invariant test
   (`__version__ == importlib.metadata.version(...)` plus a CITATION/pyproject/tag equality assert);
   fix `CITATION.cff`'s false `1.0.0 / 2025-10-18`.
7. **Add `maven-publish` + publication block + POM** to `:MobileTransformers` (#30, entirely unstarted).
8. **Write the two real script bodies** (`android_build_aar.sh`, `publish_local_maven.sh`) — currently
   `exit 1`.
9. **Decide + implement native-`.so` packaging** into the published AAR (or declare explicit consumer
   dependencies) and document the choice.
10. **Build `examples/consumer-app/`** and prove `mavenLocal()` resolution — the #30 workflow checkpoint.
11. **Resolve the CI native-dep provisioning question** so android-assemble stops self-skipping.
12. **Make export-smoke match its contract** — tiny `generate_artifacts` + manifest validation + 1-token
    desktop smoke, under the ORT-training profile (needs #11 too).
13. **Add the missing CI legs** — Kotlin/JVM unit tests, markdown link check, tag-triggered AAR build.
14. **Write `docs/ARCHITECTURE.md`** (unblocked: #23/#24 code-complete).
15. **Write `docs/ANDROID_SDK.md`** (blocked on #30).
16. **De-drift `docs/RAG.md`** (#26/#27 now implemented) and **finish `docs/PUBLIC_API.md`** (Kotlin facade
    section; add the `federated` CLI command).
17. **Fill `CHANGELOG.md` `## [1.0.0]`** with the plan's non-goals (GPU/NPU training, multimodal, leaner-
    trainer race).
18. **Replace `device.yml`'s echo body** with the real train→merge→generate→RAG instrumentation, and get
    one nightly artifact set — the release checklist's "device evidence" section.
19. **Publish ≥1 starter model package** (or formally accept the "documented how to build it" alternative).
20. **Tick the full `docs/RELEASE_CHECKLIST.md` and create the annotated `v1.0.0` tag.**

## Remaining work, ordered

**Host-doable now (no device, no external decision):**
- `docs/ARCHITECTURE.md`; de-drift `docs/RAG.md`; finish `PUBLIC_API.md` (Kotlin facade + `federated`).
- README happy-path (`make` commands) + `agent_docs/` delineation note.
- `maven-publish` block, coordinates, POM, Gradle `version`/`group` + `-Pversion` plumbing (#30 steps 2-3).
- Real `scripts/android_build_aar.sh` / `publish_local_maven.sh` bodies (runnable locally, where the
  native deps *are* present).
- `examples/consumer-app/` skeleton.
- `THIRD_PARTY_NOTICES.md`; SPDX pass + the two SPDX tests; `pyproject.toml` license expression.
- Version-site reconciliation + the invariant test; `CITATION.cff` fix; `CHANGELOG.md` `[1.0.0]` section.
- CI: Kotlin/JVM unit-test job, markdown link check, tag-triggered AAR job; fix the stale `aarLibs/` gate
  to check `jniLibs/` + `cpp/includes/` instead.
- Tests for `make help` completeness and `clean-generated` non-destructiveness.
- Delete or repoint the dangling `jniLibs.srcDirs("libs")` (`build.gradle.kts:13`).

**Device / native-runner required:**
- Actual AAR assemble + `publishToMavenLocal` + consumer-app resolution (the #30 checkpoint) — needs the
  vendored `.so`s, i.e. the author's machine or a provisioned runner.
- The real export-smoke content (`generate_artifacts` + 1-token desktop) — needs the cp312 source-built
  wheel.
- `device.yml` train→merge→generate→RAG evidence — needs a physical device and a self-hosted runner
  labelled `android-device`, which does not exist yet.
- `docs/ANDROID_SDK.md` (write after the AAR path actually works).

**Manual / user-run (decisions, not code):**
- **Relicense sign-off from both rights holders** — the single blocking values decision.
- The CI native-dep provisioning choice (rebuild-in-CI vs. private registry vs. cached artifact).
- Registering a device-attached self-hosted runner.
- Publishing a starter model package to the Hub.
- Running the full `docs/RELEASE_CHECKLIST.md` and pushing the annotated `v1.0.0` tag.
