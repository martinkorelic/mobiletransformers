# Versioning, License & v1.0 Release

**Priority #32 | Prerequisites: #29 (`05_code_plans/02`, CI), #30 (`05_code_plans/03`, AAR), #31 (`05_code_plans/04`, docs) | Blocks: — (terminal release gate)**

## Purpose

Turn the work into a tagged, citable, adoptable `v1.0.0`: adopt SemVer, resolve the licensing blocker (relicense code to **Apache-2.0**, the framework target), refresh citation metadata, and run a release checklist. The non-commercial CC-BY-NC-4.0 license is the single biggest adoption blocker for "de-facto framework" positioning and must be decided before tagging.

## Touched / new files

- `LICENSE.md` — currently CC-BY-NC-4.0; relicense code to Apache-2.0 (keep model weights/data under upstream licenses).
- Source headers — add SPDX identifiers (`SPDX-License-Identifier: Apache-2.0`) to code; keep docs/data licensing separate and explicit.
- `CITATION.cff` — currently `version: 1.0.0`, `date-released: 2025-10-18`; reconcile so the declared version/date match the actual tagged release.
- NEW `CHANGELOG.md` (skeleton from #31) — fill v1.0.0 notes + limitations.
- NEW `docs/RELEASE_CHECKLIST.md` (skeleton from #31) — the gate list below.
- `.gitignore` — decide whether to **un-ignore `agent_docs/`** so the plan ships with the release (currently ignored).
- Git — first annotated tag `v1.0.0`.

## Data contracts / interfaces

### SemVer policy

- `0.x`: research / pre-v1.
- `1.0.0`: first stable **public API + model-package format + CLI** release (these become the SemVer-governed compatibility surfaces — see the reimplementation-avoidance gates in `05_cross_cutting_release_modernization.md`).
- Patch: bug/doc fixes. Minor: new model families, PEFT modes, optional engines.

SemVer requires a declared public API before 1.0.0. Per **F5**, that declaration is the union of: the Python `mobiletransformers.__all__` importable surface (owned by `00_code_plans/10`), the Kotlin facade (#19), the manifest schema (#13/#9), the Maven coordinates (#30), and the CLI names (#15/#28) — documented together in `PUBLIC_API.md` (#31). A break in any of these is a major bump.

### Release checklist (`docs/RELEASE_CHECKLIST.md`)

```
[ ] License decided + applied (Apache-2.0 code) + SPDX headers
[ ] All authors agreed to relicense (values/strategy decision, flag to all)
[ ] CITATION.cff version/date == tag
[ ] CHANGELOG.md v1.0.0 with limitations + non-goals (GPU/NPU training, multimodal)
[ ] CI green (fast + export-smoke + android-build) on the release commit
[ ] AAR builds + publishes to mavenLocal; consumer-app smoke passes
[ ] ≥1 starter model package published OR documented how to build it
[ ] docs set complete + link-check green
[ ] agent_docs/ ignore decision made
[ ] tag v1.0.0 (annotated)
```

## Implementation steps

1. **License decision (blocker, not engineering):** confirm Apache-2.0 with all rights holders; apply `LICENSE.md` + SPDX headers; document model/data licensing separately.
2. Reconcile `CITATION.cff` version + date with the actual tag.
3. Fill `CHANGELOG.md` (v1.0.0, with the documented non-goals: GPU/NPU training, multimodal training, "leaner trainer" race).
4. Decide `.gitignore` `agent_docs/` (recommend un-ignoring so the design ships).
5. Run the checklist; publish at least one starter package (or document the build).
6. Tag `v1.0.0`; attach release notes + AAR/local-Maven instructions.

## Interactions

- **#29 / #30 / #31**: their green/complete state are checklist gates.
- **#13 / #15 / #19 / #30**: define the SemVer-governed public surfaces.
- **`IMPLEMENTATION_ORDER.md` canonical decision**: Apache-2.0 is already the stated target license; this plan executes it.

## Worked example

SPDX header on each source file:

```python
# SPDX-License-Identifier: Apache-2.0
```

`CHANGELOG.md` v1.0.0 skeleton:

```markdown
## [1.0.0] - 2026-06-25

### Added
- First stable public API, model-package format, and CLI.

### Non-goals
- GPU/NPU training (on-device CPU/XNNPACK training only).
- Multimodal training.
```

`CITATION.cff` reconcile — the declared version/date must equal the annotated tag:

```yaml
version: 1.0.0          # == git tag v1.0.0
date-released: 2026-06-25   # == tag date
```

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- License + SPDX present on a sample of source files; `LICENSE.md` is Apache-2.0.
- `CITATION.cff` parses; `version`/`date-released` == the tag (assert programmatically).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `git describe --tags` returns `v1.0.0` after tagging.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Run the full `docs/RELEASE_CHECKLIST.md` and confirm every item is ticked before the tag is pushed (license/relicense sign-off is a values decision flagged to all rights holders).

**Workflow (end-to-end)** — *checkpoint #32.*
- The full release gate, end-to-end: CI green (fast + export-smoke + android-build, #29) **+** AAR publishes to mavenLocal (#30) **+** consumer-app smoke passes **+** docs link-check green (#31) **+** annotated `v1.0.0` tag created.

**Definition of done** — `LICENSE.md` is Apache-2.0 with SPDX headers applied, `CITATION.cff` version/date reconcile to the tag, `CHANGELOG.md` carries v1.0.0 notes + the non-goals (GPU/NPU training, multimodal), the release checklist is fully ticked, and the annotated `v1.0.0` tag is pushed on a commit that passes the full release gate.
