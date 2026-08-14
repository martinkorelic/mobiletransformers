# Handoff — what is still open

**Branch:** `restructure` · **Reset:** 2026-08-14

This file is **only what is left to do**. Completed work is not recorded here — it lives in
`IMPLEMENTATION_ORDER.md`'s per-plan self-checks and in git. Rewritten each cycle.

> ## Read these first
>
> 1. **`IMPLEMENTATION_ORDER.md` → "Operational knowledge (permanent)"** — environment and profiles,
>    the gotchas that each cost a cycle, the ORT engine separation, the device workflow, the
>    layer-identity problem, the recurring failure shape, the gate results, and the recorded device
>    suite. Everything a cold agent needs *regardless of cycle* lives there, not here.
> 2. **`IMPLEMENTATION_ORDER.md` → "How to execute a plan"** — the implementer protocol. Follow it.
> 3. The per-plan **self-check blocks** in the same file — each records what is proven and how.
>
> **Do NOT read `agent_docs/audits/` as a to-do list.** They are 2026-08-07 snapshots with no closure
> annotations; each opens with a banner listing findings that are verifiably fixed. This file is the
> authoritative list of what is open.

---

## Where the project stands

**35 / 37 plans done · 107 / 111 self-check boxes.** The two open plans are **#32 (licence-gated, not
work)** and **#37**, whose one open box is the differentiation demo. **No known defect is outstanding.**

**Host gates (measured 2026-08-14 — do not copy forward, re-measure):**

```
Python 549 passed / 11 skipped · C++ 39 · Kotlin JVM 247 · guard 7 · parity OK · uv lock --check clean
```

```bash
uv sync --frozen --group dev --python 3.10 && make check    # ALWAYS reset the profile first
make test-cpp
cd android/MobileTransformers && JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformers:testDebugUnitTest
```

Read the real numbers from `build/test-results/testDebugUnitTest/*.xml` and the pytest summary; never
carry a figure over (a previous handoff's "guard 10" was drift against a tree producing 7).

**Device suite** (S21 FE `SM-G990B` / Android 15 / arm64-v8a) — the result depends on which package is
installed, and the cache holds one:

| package on device | result |
| --- | --- |
| SmolLM2-135M `TRAIN=1 RAG=1` (decoder) | 15 / 15 pass, 798.0 s |
| all-MiniLM-L6-v2 `TASK=text-classification TRAIN=1` (encoder) | 19 tests, 0 failures, 16 skipped |

**The device currently holds the ENCODER package.** Re-push the decoder before any generation work:
`make device-package MODEL=HuggingFaceTB/SmolLM2-135M-Instruct TRAIN=1`.

---

## 1. Code — what is actually left

### 1a. #37's differentiation demo (the only open plan-level item)

The chain is **per-user action set → on-device fine-tune → validated tool call → dry-run intent**.
**Every link is now built.** What is left is one thing: *running it*.

| link | state |
| --- | --- |
| dataset (import a corpus **or** synthesise per-user) | done — `mobiletransformers agent-dataset` |
| training-data bridge (`MobileActionsPreprocessor`, task `mobile_actions`) | done |
| tool-call validator / dry-run binder | done — `agent/FunctionCallValidator.kt`, `agent/IntentBinder.kt` |
| generated text → validator | done — `MobileTransformerModel.generateToolCall()` → `ToolCallResult` |
| the instrumented test | **WRITTEN, NEVER RUN** — `ToolCallDeviceTest` |

> ### ⚠️ `ToolCallDeviceTest` compiles but has NOT been executed on hardware.
>
> It was written 2026-08-14 with no device attached. **Do not record #37 as done on the strength of it
> existing** — that is exactly the "self-check ticked without evidence" pattern the 2026-08-07 audit
> found. Running it is the remaining work, and its outcome is a genuine unknown: it asserts the model
> emits a call its own validator accepts after ~120 steps on a 9-pair memorisable corpus, and whether a
> 135M base does that in an instrumented run has never been measured.
>
> A refusal is a **real result to record**, not a reason to weaken the assertion. If it refuses, the
> useful next questions are more steps, a smaller corpus, or the imported `google/mobile-actions` set
> for format-learning first — not asserting on the refusal path, which proves nothing (an untrained
> model also refuses).

Build the data with either or both sources:

```bash
mobiletransformers agent-dataset --source google/mobile-actions --output build/agent
mobiletransformers agent-dataset --source generated --allowlist build/agent/action_schema.json --output build/user
```

What a demo still has to solve:

- the **decoder** package must be back on the device, exported with the dataset wired to the
  `mobile_actions` task so `ORTDataCurator` picks up the new preprocessor;
- the assertion is **convergence-dependent**: a validated call means the model emitted JSON the
  validator accepts. A handful of LoRA steps on a 135M base will not do that — plan a long device run,
  and use `gradAccumSteps = 1` (a bounded round with the default 4 applies no update at all);
- **the two dataset sources answer different halves**, which is why both exist. The corpus (5,794 rows
  from `google/mobile-actions`) teaches the JSON tool-call *format* — the generic, data-hungry part.
  The generated per-user set is a short pass on top that teaches *this user's* actions, and is what the
  personalization differentiator rests on. Corpus-only drops that differentiator but still clears the
  ≥2 bar; generated-only is unlikely to converge from ~24 rows;
- a demo that "passes" via the **refusal** path shows nothing — the validator rejecting garbage is
  already covered by 15 JVM tests. Assert on an *accepted* call reaching `IntentBinder.dryRun`.

For FunctionGemma specifically, on-device training stays blocked by `ort-training-local`'s
`transformers==4.46.2` — a **#2/#3 dependency decision, not a #37 one**; floating that pin has broken
`get_peft_model` before. The demo runs on a trainable decoder plus the Gemma-3 inference graph, which
is worth stating plainly since it weakens the "FunctionGemma" framing.

**Do not chase #37's DoD gate 1 literally.** It asks for `inference_model_class = Gemma3Model` on the
Gemma-3 `ArchitectureSpec`; `config/registry/architecture.py:193-195` sets it to `None` on purpose —
that field is the vendored GenAI-builder path, and Gemma-3 inference export goes through optimum
`main_export` instead (proven 2026-08-09 with a full package). There is no `Gemma3Model` class and the
plan's `tests/inference/test_gemma3_registry.py` does not exist. Record the route; do not invent a class.

### 1b. **The showcase app rewrite** — decided 2026-08-14, not started

**Decision: a full rewrite of `MobileTransformersApp` on the public facade, covering every shipped
feature, doubling as the reference example others build from.** This is the largest remaining piece of
work in the repo and it is deliberately handed off as a plan rather than half-built. The outline below
is scoped from what this cycle verified; **a fresh agent should do its own exploration pass before
writing code** — the file inventory here is a starting point, not a survey.

#### Why: the facade is shipped but unexercised

`MobileTransformersApp` drives the **legacy repository layer** directly —
`LLMRepository`/`TrainingRepository`/`RagRepository`/`InferenceRepository`. Verified by grep over the
app module 2026-08-14: `fromPretrained`, `MobileTransformerModel`, `InferenceEngine`, `hub.`,
`pushAdapter`, `WorkManager`, `Federated`, `agent.` each return **zero** references.

So the public SDK surface #17/#19 built — the thing every external consumer is told to use — **is not
exercised by the one app that ships with it**, and `examples/consumer-app` only calls a smoke check.
That is a real debt with two costs: the facade's ergonomics have never met a real screen, and there is
no worked example of the API anyone is asked to adopt.

The current app is 3 tabs / ~2,700 lines — Inference (chat, streaming, RAG toggle + source cards),
Training (train, progress, merge-at-end), Configuration (~45 knobs). **Treat it as a specification of
required behaviour, not as a base to patch**: it is the record of what the SDK must be able to express.

#### Target

One app, facade-only, where every screen is a self-contained readable example of one capability.

| screen | covers | notes |
| --- | --- | --- |
| Models / Hub | #21, #13 | repo id → pull with progress → installed packages → variant + engine capabilities. **Start here**: today the app assumes an adb-pushed package, which no real user can do, so nothing else is reachable without it. |
| Chat | #11, #24, #27 | generate + streaming callbacks, RAG toggle with source cards, **engine picker (Native / GenAI)** showing `EngineCapabilities` and the transparent-fallback rule |
| Train | #18, #19, #34 | `trainingJob()` — status/events flows, cancel, resume, checkpoint; scheduled charging-cycle training via WorkManager |
| Tool calls | #37 | instruction → `generateToolCall` → Accepted/Rejected shown as a first-class outcome → dry-run intent with `willExecute=false` visible. **The screen that carries the differentiation argument.** |
| Federated | #35/#36 | one round against a local `federated serve`: export → upload → aggregate → import, with the consent gate visible |
| Configuration | all | the ~45 knobs, but expressed through the public config types (`GenerationConfig`/`TrainConfig`/`RagConfig`/`DatasetConfig`), not `ORT*` |

#### Constraints — do not rediscover these the hard way

1. **Facade-only is the point.** No `ORT*`, `*Native` or `*Repository` type may appear in app code. Add
   a guard (extend `tests/unit/test_guards.py`, or a Kotlin source-grep test in the style of
   `NativeLoadRegressionTest`) so the boundary cannot quietly erode. **Write the guard first** — it is
   the only thing that makes "migrated" checkable rather than claimed.
2. **The facade will be found wanting.** It has never driven a real screen. Expect gaps (progress
   granularity, cancellation, capability introspection, hub-download progress). **Fix them in the
   facade, not by reaching around it** — a reach-around is the debt reappearing. Every gap found is a
   #17/#19 finding worth recording.
3. **`adapterUpload` and federation are default-off** (`BuildConfig.ADAPTER_UPLOAD_ENABLED`,
   `FEDERATION_ENABLED`). The UI must show the disabled state honestly rather than hiding the feature.
4. **`IntentBinder` never executes.** Dry-run is the default and the app must display the intended
   action, not fire it. Executing is opt-in, allowlisted, and out of scope for the showcase.
5. **Device-only reality.** arm64-v8a only, no x86_64 emulator; most screens do nothing without a
   pushed or pulled package. Every screen needs a legible empty state — "no model installed" is the
   first thing a new user sees.
6. **Keep the sample app out of the published AAR.** It is `:MobileTransformersApp`, not the library;
   `make publish-local` must stay unaffected.

#### Suggested order

1. Guard + skeleton: new navigation, facade-only rule enforced, everything else stubbed.
2. Models/Hub screen — unblocks every other screen on a real device.
3. Chat (+ engine picker), then Train — these two replace the existing app's behaviour; the old screens
   come out only once their replacements work.
4. Tool calls — depends on `ToolCallDeviceTest` having been run, since an unconverged model makes for a
   demo that only ever shows refusals.
5. Federated, Configuration.
6. Delete the legacy ViewModels/screens once nothing references them.

#### Examples deliverable

`docs/COOKBOOK.md`: one runnable, copy-pasteable snippet per task (load → generate; train → merge;
ingest → grounded answer; import a tool-call dataset → fine-tune → validated call → dry-run intent;
pull from the Hub; select an engine), each mirroring the screen that implements it so the doc and the
app cannot describe different APIs. Link it from `README.md` and cover it with the existing docs
guards, extending `test_docs.py` so a snippet cannot name a Kotlin symbol that stopped existing —
otherwise the cookbook rots exactly the way `docs/RAG.md` did.

#### Definition of done

- No `ORT*`/`*Native`/`*Repository` reference anywhere in `MobileTransformersApp`, enforced by a test.
- Every table row above has a working screen, or a recorded reason it does not.
- `make android-build` and `make consumer-app` still pass; JVM tests cover the new ViewModels' pure
  logic (state mapping, empty/disabled states) without a device.
- `docs/COOKBOOK.md` exists, is linked from the README, and is guarded.
- Facade gaps found during the rewrite are recorded against #17/#19 rather than worked around.

### 1c. Smaller code items

- **`agent-dataset` intent mapping** — `ANDROID_INTENT_BY_ACTION` covers 5 of the corpus's 7 actions.
  Both flashlight actions are `CameraManager` torch calls with no public intent, so they are
  deliberately unmapped: trainable and validatable, never bindable. Extending it is an app decision.

*(The #31 markdown link-check closed 2026-08-14: `test_docs.py` now sweeps **every tracked markdown
file** — `agent_docs/` included, which is where the worst reference rot was — and `ci.yml`'s fast job
runs it as a named step.)*

---

## 2. Device / manual test legs still open

- **`ToolCallDeviceTest` has never been run** (§1a). It is the #37 checkpoint.
- **`PostMergeNumericsTest` has never appeared in a recorded suite run** — it was written after the last
  package was pushed. It needs its own run on a **pristine** package (the training suites mutate the
  package in place).
- #9: on-device atomic-overwrite-under-kill, and offline-vs-device byte-identical `.bin` parity.
- #21: Kotlin `VariantSelector` parity + the on-device load leg. #22: a real authenticated adapter
  upload with a checkpoint factor read.
- #34: multi-hour / Doze / FGS-quota behaviour — a recorded DEBT; the plan's own DoD does not require it.

---

## 3. Not engineering

- **The licence — the only real v1.0 blocker.** CC-BY-NC-4.0 contradicts the consumable-AAR goal. It is
  a rights-holders decision (both authors in `CITATION.cff`); the four sites are listed in
  `docs/RELEASE_CHECKLIST.md`. **The second author has NOT agreed** — do not touch licence files, do
  not add SPDX headers, do not set the `pyproject.toml` license expression. Flag it, nothing more.
  (`tests/unit/test_release_plumbing.py` asserts the POM matches `LICENSE.md`, so a one-sided change
  fails the suite.)
- **CI provisioning** — either vendor the native deps to a runner, or formally define "CI green" in
  `docs/RELEASE_CHECKLIST.md` as a recorded manual run. Today it is neither, which is the actual debt.
  All three workflows are `workflow_dispatch`-only by choice; `device.yml` targets a
  `[self-hosted, android-device]` runner that does not exist.
- **The `v1.0.0` tag** (`git tag` is empty), and publishing ≥1 starter model package — or formally
  accepting the "documented how to build it" alternative.

---

## 4. Standing debts — accepted positions, do not "fix" without deciding

- **Variant naming** — `cpu-int4` legitimately ships an fp32 inference graph, declared via the measured
  `inferenceGraphPrecision`. Renaming is a wire-contract change, deliberately not done.
- **`inference/builder.py`** carries ~10 upstream-derived TODOs and its own `load_config_from_file`
  copy. Vendored GenAI builder; treat as upstream. Allow-listed in the architecture-literal guard.
- **`peft/mapping.py` has two decoder-shaped prefix sites left, deliberately untouched.** One converts
  `base_layer_name` into checkpoint space for the **MARS** mapping only (LoRA's mapping keeps raw keys),
  so generalising it would change MARS's `peft_mapping` keys for encoders — and
  `test_mars_encoder_transfer.py` passes today against the current shape. The other
  (`module_name.replace(...)` with the result discarded) is **dead, and load-bearing by being dead**:
  the loop below it works in raw space, so assigning it would break `relative_path`. Both want a
  deliberate look with the MARS tests in hand, not a sweep.
- **The stage-path guard carries 2 allow-listed sites**, both legitimate producers that write a layout
  down once (`export/pipeline.py`, `ModelPackageInstaller.kt`).
- **`export-rocm` is a declared-but-empty group** — ROCm wheels need a dedicated AMD index.
- **Android ORT-training AAR** — `third_party/onnxruntime/manifest.json`'s Android fields are still
  null (`ndk_version`/`abis`/`aar_sha256`); that leg was never built.
- **Broadening the top-level `__all__`** — still #32's call.
- **macOS cannot run the training side** without rebuilding the `cp312-linux_x86_64` wheel for
  `macosx_arm64`. Details in `IMPLEMENTATION_ORDER.md`'s permanent section.
- **Reclaimable disk (untracked, gitignored):** `build/` 2.5 G, `.venv-genai-spike/` 5.2 G — the latter
  is recreated by `spikes/genai_external_swap/build_tiny_genai_model.sh`.

---

## Cycle protocol

Follow `IMPLEMENTATION_ORDER.md` "How to execute a plan". For this repo specifically:

1. **Reset the profile before `make check`** — `uv sync --frozen --group dev --python 3.10`. More
   "broken repo" reports have come from a leftover export/training venv than from any real defect.
   `scripts/device_package.sh` leaves the tree on the training profile.
2. **Flip a `Done` box only when the self-check holds**, with the evidence (device, OS, ABI, date)
   inline.
3. **Assert across the seam, and check the assertion can FAIL.** This keeps paying: the 08-10 cycle's
   best finding came from tightening an assertion that already passed (it was measuring clipping and
   calling it training). On 08-14 the same check found three guards asserting nothing against an empty
   scan subject, and a Gradle staleness bug where a *corrupted* cross-language oracle produced
   `BUILD SUCCESSFUL` in 670 ms without running a test. When you empty a ratchet, re-point it or delete
   it — never leave it scanning the void.
4. **When you correct an earlier claim, correct it where the claim lives**, not only in a new entry.
