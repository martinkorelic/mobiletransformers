# Release Checklist

> The versioning, licence and release gate for a tagged release.
>
> Status as of 2026-08-09: every technical item below is satisfied except CI (a policy decision, see
> below) and the licence. **The licence is the single blocker on the release gate**, and it is a
> rights-holders decision, not an engineering task.

## Gate

- [x] Parity gate green (`make parity`): enums/schemas match the Kotlin/C++ mirrors.
- [x] Lint + typecheck clean (`make lint`, `make typecheck`).
- [x] Host test gates green: Python `make check`, C++ (`make test-cpp`), Kotlin JVM
      (`:MobileTransformers:testDebugUnitTest`).
- [x] AAR builds + publishes to mavenLocal and an external consumer app builds against it.
      *Proven 2026-08-08: `make publish-local && make consumer-app` → a 105 MB APK carrying all 7
      native libraries, resolved from mavenLocal alone (`FAIL_ON_PROJECT_REPOS`).*
- [x] Docs set complete for locked contracts; `COMPATIBILITY_MATRIX.md` regenerated (not stale).
- [x] All version sites agree (pyproject == `__version__` == Gradle `version` == `CITATION.cff` ==
      the `tiny_package` fixture's `mobiletransformersVersion` == tag).
      *Guarded by `tests/unit/test_version_sites.py`, so this cannot silently drift. The Gradle
      version lives in `android/MobileTransformers/gradle.properties` and is overridable with
      `-Pversion=`. The sample app's `versionName` now derives from that root property rather than
      being a literal — it read `"1.0"`, which matched no other site in the repo and was the one
      version number a user actually sees.*
- [x] **Model shelf published** — `make publish-catalog` run, and every entry in the app's
      `assets/model_catalog.json` names a repo that actually holds a package.
      *Needs `HF_TOKEN_ORG` in `.env`. Verify against the Hub API, not the script's own log: a push
      that half-succeeded and a push that worked print the same final line. See
      [CATALOG.md](CATALOG.md).*
      *Done 2026-08-17: five repos, each carrying a `mobiletransformers_manifest.json`.*
- [x] **A fresh clone can be provisioned** — `make doctor` names every missing prerequisite with the
      command that fixes it, and `make fetch-native-deps` installs the gitignored Android natives
      against the sha256s in `third_party/android/manifest.json`.
      *`baseUrl` in that manifest is still `null` — the bundles are built but not hosted, so the
      fetch path is proven only against a `file://` mirror. Hosting them is an owner action; a GitHub
      Release keeps the URL stable.*
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
