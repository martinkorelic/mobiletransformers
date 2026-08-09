# Release Checklist

> Skeleton created by #31. **Finalized by #32** (`05_code_plans/05`, versioning/license/release gate).
>
> Status as of 2026-08-09: every technical item below is satisfied except CI (a policy decision, see
> below) and the licence. **The licence is the single blocker on the release gate**, and it is a
> rights-holders decision, not an engineering task.

## Gate

- [x] Parity gate green (`make parity`): enums/schemas match the Kotlin/C++ mirrors.
- [x] Lint + typecheck clean (`make lint`, `make typecheck`).
- [x] Host test gates green: Python `make check`, C++ (`make test-cpp`), Kotlin JVM
      (`:MobileTransformers:testDebugUnitTest`).
- [x] AAR builds + publishes to mavenLocal and an external consumer app builds against it (#30).
      *Proven 2026-08-08: `make publish-local && make consumer-app` → a 105 MB APK carrying all 7
      native libraries, resolved from mavenLocal alone (`FAIL_ON_PROJECT_REPOS`).*
- [x] Docs set complete for locked contracts; `COMPATIBILITY_MATRIX.md` regenerated (not stale).
- [x] All version sites agree (pyproject == `__version__` == Gradle `version` == `CITATION.cff` == tag).
      *Guarded by `tests/unit/test_version_sites.py`, so this cannot silently drift. Note the Gradle
      version lives in `android/MobileTransformers/gradle.properties` and is overridable with
      `-Pversion=`; the sample app's own `versionName` is unrelated and deliberately not tracked.*
- [x] `CHANGELOG.md` updated for the release; non-goals listed, and a `Known issues` section carries
      what a reader must not discover the hard way.
- [ ] **License finalized** with SPDX headers on first-party source. **BLOCKER — see below.**
- [ ] CI green: `fast` → `export-smoke` → `android-assemble` (`.github/workflows/ci.yml`).
      **Blocked on a policy decision, not on the code.** All three workflows are
      `workflow_dispatch`-only: their automatic triggers were removed 2026-08-08 because they were not
      in use and the native-dependency provisioning question is unresolved. Until the `on:` blocks are
      restored (they are preserved in a comment at the top of each file), **a green badge is not a
      gate** — a release must record a manual run instead. See "Open decisions".

## Device evidence

- [x] train 1 step → merge → generate 1 token → ingest/query 1 RAG doc, with time/memory artifacts.
      *Ran on a Galaxy S21 FE / Android 15 / arm64-v8a, 2026-08-08, via `make device-package` →
      `make device-test`. `device.yml` has a real body but no registered self-hosted runner, so this
      evidence is produced by a manual run, not nightly CI.*

## The licence blocker

The project is **CC-BY-NC-4.0**, which is incompatible with the consumable-AAR goal: a non-commercial
licence blocks the adoption the release is for. Relicensing is a decision for **all rights holders** —
`CITATION.cff` lists Korelič and Pejović.

Nothing technical is waiting behind it. When agreement lands, the swap is four coordinated edits plus
headers, and should go in as **one reviewed commit**:

| Site | Current | Change |
| --- | --- | --- |
| `LICENSE.md` | full CC BY-NC 4.0 text | replace with the chosen licence text |
| `pyproject.toml` (~`:21`) | `license` deliberately omitted, with a comment naming the pending decision | set the licence expression |
| `android/MobileTransformers/MobileTransformers/build.gradle.kts` (POM block) | hard-codes CC BY-NC 4.0, with a comment requiring lockstep with `LICENSE.md` | update to match |
| first-party source | **zero** SPDX headers exist anywhere | add `SPDX-License-Identifier` headers |

**Scope of the headers: first-party source only.** Vendored Microsoft / tokenizers / protobuf code is
untouched and is already enumerated in `THIRD_PARTY_NOTICES.md`, which is complete and current
(redistributed-in-AAR, build/test-only, and Python dependency tables). Model weights keep their
upstream licences regardless of this decision.

## Open decisions blocking a v1.0 tag

1. **Licence** — above. The only item the release gate genuinely waits on.
2. **CI triggers** — restore `push`/`pull_request` on `ci.yml`, or keep workflows manual and accept
   that "CI green" means a recorded manual run. Either is defensible; the checklist must match
   whichever is chosen.
3. **CI native-dep provisioning** — how `jniLibs`/`aarLibs` and the git-ignored cp312 ORT-training
   wheel reach a hosted runner (rebuild vs cached artifact vs private storage). `android-assemble` and
   `ort-training-smoke` self-skip without them today. Only worth answering if (2) restores triggers.
