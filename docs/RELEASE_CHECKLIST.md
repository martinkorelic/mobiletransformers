# Release Checklist

> Skeleton created by #31; **finalized by #32** (`05_code_plans/05`, versioning/license/release gate).
> Boxes below are the intended gate — do not treat as complete until #32 wires and verifies them.

## Gate

- [ ] CI green: `fast` → `export-smoke` → `android-assemble` (`.github/workflows/ci.yml`).
      **Note (2026-08-08): the workflows are `workflow_dispatch`-only** — their automatic triggers
      were removed because they are not in use and the native-dependency provisioning question is
      unresolved. Run them manually for a release, or restore the `on:` blocks (recorded in a
      comment at the top of each file) before treating a green badge as a gate.
- [ ] Parity gate green (`make parity`): enums/schemas match the Kotlin/C++ mirrors.
- [ ] Lint + typecheck clean (`make lint`, `make typecheck`).
- [ ] AAR builds + publishes to mavenLocal and an external consumer app builds against it (#30).
- [ ] Docs set complete for locked contracts; `COMPATIBILITY_MATRIX.md` regenerated (not stale).
- [ ] All version sites agree (pyproject == `__version__` == Gradle `-Pversion` == `CITATION.cff` == tag).
- [ ] License finalized (Apache-2.0) with SPDX headers on first-party source; `THIRD_PARTY_NOTICES.md`
      enumerates vendored code.
- [ ] `CHANGELOG.md` updated for the release; non-goals listed.

## Device evidence (nightly `device.yml`)

- [ ] train 1 step → merge → generate 1 token → ingest/query 1 RAG doc, with time/memory artifacts.
