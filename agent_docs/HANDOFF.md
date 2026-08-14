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
work)** and **#37**, whose one open box is the differentiation demo. **One known defect is now
outstanding** — the #37 merge-numerics failure in §1a, which is characterised but not fixed.

**Host gates (measured 2026-08-14 after the app rewrite — do not copy forward, re-measure):**

```
Python 552 passed / 11 skipped · C++ 39 · Kotlin JVM 252 (library) + 6 (app) · guard 8 · parity OK
uv lock --check clean · make android-build / publish-local / consumer-app all pass
```

```bash
uv sync --frozen --group dev --python 3.10 && make check    # ALWAYS reset the profile first
make test-cpp
# BOTH modules now: the sample app has its own JVM suite as of the facade rewrite.
cd android/MobileTransformers && JAVA_HOME=/opt/android-studio/jbr ./gradlew \
  :MobileTransformers:testDebugUnitTest :MobileTransformersApp:testDebugUnitTest
```

Read the real numbers from `build/test-results/testDebugUnitTest/*.xml` and the pytest summary; never
carry a figure over (a previous handoff's "guard 10" was drift against a tree producing 7).

**Device suite** (S21 FE `SM-G990B` / Android 15 / arm64-v8a) — the result depends on which package is
installed, and the cache holds one:

| package on device | result |
| --- | --- |
| SmolLM2-135M `TRAIN=1 RAG=1` (decoder) | 15 / 15 pass, 798.0 s |
| all-MiniLM-L6-v2 `TASK=text-classification TRAIN=1` (encoder) | 19 tests, 0 failures, 16 skipped |

**The device currently holds the train-capable SmolLM2-135M-Instruct DECODER package** (pushed
2026-08-14). It has been mutated by the `ToolCallDeviceTest` runs — re-push before any run that needs a
pristine package.

**You do not need a full re-export to re-push.** `build/pkg` holds the exported decoder; steps 2–4 of
`scripts/device_package.sh` (reshape → `adb push` → `chmod -R 777` → stage `mt_genai_spike`) are pure
file movement and take ~15 s, against ~30 min and two profile switches for a full
`make device-package`. The profile switching is also how gotcha 13's `onnxruntime` collision gets
reached, so prefer the short path unless you are testing the export itself.

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
| the instrumented test | **RUN 2026-08-14 — FAILS at generation** (`ToolCallDeviceTest`); root cause narrowed to merge numerics, see below |

> ### ✅ `ToolCallDeviceTest` ran 2026-08-14 and FAILED — and the defect it exposed is now FIXED.
>
> **The merge wrote every weight TRANSPOSED.** Root-caused, fixed and verified the same day; full
> detail in `IMPLEMENTATION_ORDER.md` → "The #37 demo run". `PostMergeNumericsTest`'s new large-delta
> test **PASSES** post-fix: memorised text 1.295 nats vs unrelated 5.529, against a 4.651 pristine
> baseline and a 10.803 uniform floor. Before the fix both were *above* the floor.
>
> **`ToolCallDeviceTest` itself has NOT been re-run since the fix** — that is the next device run, and
> it is now a fair test for the first time.
>
> The original failure analysis is kept below because its elimination was sound.
>
> *(This block previously read "compiles but has NOT been executed on hardware". That is no longer
> true; it ran twice on the S21 FE `SM-G990B` / Android 15 / arm64-v8a.)*
>
> **Result: the model emits token 198 (newline) 48 times instead of a call.** But the run excludes
> every cause the old advice pointed at:
>
> * training **converges**: loss 5.85 → 0.006 over 108 steps (99.5% drop);
> * the merge **completes** over all 60 q_proj/v_proj tensors;
> * merged weights **reach inference** — 60 "Loaded merged initializer" lines, and **zero** "loading
>   base weights" fallbacks;
> * the prompt **matches training** — generation input logged as `[1, 42037, …]`, i.e. BOS + the
>   instruction, the same sequence the curator builds.
>
> **So do NOT spend the next cycle on more steps or a smaller corpus** — that advice was right, and
> the pointer to the weight-space mapping was the right neighbourhood. The defect turned out to be one
> step further out: not the `B·A` delta's orientation but the **entire merged weight's** orientation on
> write, which is why it corrupted the model even with a zero delta. **Fixed.**
>
> `PostMergeNumericsTest` passing does not contradict this — it asserts the merge *changes* the
> computation and stays finite, not that the delta is correct.
>
> Full detail, including the two real defects fixed along the way (BOS parity, missing EOS) and one
> latent one (the chat template lives in `chat_template.jinja`, which `ORTTokenizerNative` never
> reads, so `chatTemplate` is null for this package), is in `IMPLEMENTATION_ORDER.md` →
> "The #37 demo run".

Build the data with either or both sources:

```bash
mobiletransformers agent-dataset --source google/mobile-actions --output build/agent
mobiletransformers agent-dataset --source generated --allowlist build/agent/action_schema.json --output build/user
```

What a demo still has to solve — **revised 2026-08-14 against the measurement above**:

- **the merge delta's correctness.** This is the whole remaining problem. Everything upstream of it is
  now measured working: the decoder package is on the device, `mobile_actions` dispatches at runtime,
  training converges to 0.006 loss, the merge completes, and the merged tensors load at inference;
- ~~the assertion is convergence-dependent, so plan a long device run~~ — **wrong, and measured
  wrong.** 108 steps reached a 99.5% loss drop and the model still emitted newline. Convergence is not
  the blocker. `gradAccumSteps = 1` remains correct and is pinned in the test;
- **the two dataset sources answer different halves**, which is why both exist. The corpus (5,794 rows
  from `google/mobile-actions`) teaches the JSON tool-call *format* — the generic, data-hungry part.
  The generated per-user set is a short pass on top that teaches *this user's* actions, and is what the
  personalization differentiator rests on. **Neither is worth running until the merge is fixed**: a
  model that cannot express what it learned will not express more of it;
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

### 1b. The showcase app rewrite — **DONE 2026-08-14**

*(This section previously read "decided 2026-08-14, not started" and carried a target table and a
suggested order. The rewrite has been done; what follows is what exists, not what to build.)*

`MobileTransformersApp` is now a facade-only app. All six planned screens exist:

| screen | covers | state |
| --- | --- | --- |
| Models / Hub | #21, #13 | repo id → pull with live progress → installed packages → capabilities |
| Chat | #11, #24, #27 | generate + streaming, RAG toggle with source cards + the assembled prompt, engine picker driven by `availableEngines` |
| Train | #18, #19, #34 | `trainingJob()` status/events/cancel/resume/merge + `TrainingScheduler` |
| Tool calls | #37 | instruction → `generateToolCall` → Accepted/Rejected as peers → dry-run intent with `willExecute` shown |
| Federated | #35/#36 | one round, consent gate visible, disabled state shown honestly |
| Configuration | all | the knobs, through the public config types only |

**The facade-only rule is enforced**, by `tests/unit/test_guards.py::test_the_sample_app_uses_only_the_public_facade`
(now 8 guards, was 7). It derives the banned set from the library sources — everything declared in an
`ORT*.kt` file, everything matching `*Native`/`*Repository`, and the `internal.` package — rather than
hand-listing names. That matters: a plain name grep would have missed `DeviceOptions`,
`SamplingOptions`, `SchedulerConfig`, `InferenceProgress`, `TrainingProgress` and `RagResult`, all of
which the old app imported and none of which match `ORT*`. **The guard was verified to fail first**,
reporting all 20 violations in the old app, before any screen was written.

**Six facade gaps were found and fixed in the facade** — table in `IMPLEMENTATION_ORDER.md` → #17. The
sharpest: federation had *no* public entry point at all.

`docs/COOKBOOK.md` exists, is linked from the README, and is guarded by
`test_docs.py::test_cookbook_snippets_only_name_kotlin_types_that_exist`, which parses the ```kotlin
fences and fails when a snippet names a type the facade does not declare (also verified able to fail).

**What is left on the app**, none of it blocking:

- **It has never been run on the device.** It compiles, `make android-build` passes, and its pure state
  logic has 6 JVM tests, but no screen has been walked on hardware. That is the obvious next step and
  the one thing that would find the ergonomic problems JVM tests cannot.
- The Tool calls screen shows `Rejected` for the reason in §1a — that is the model, not the screen.
- The engine picker renders `availableEngines` but switching engines requires a reload from the Models
  tab (the engine is fixed at load). A single-tap switch would need the facade to re-open a session.
- The old `ui/theme/` (`AppTheme.FRI`/`BETTER`) survived the rewrite untouched.

### 1c. Smaller code items

- **`agent-dataset` intent mapping** — `ANDROID_INTENT_BY_ACTION` covers 5 of the corpus's 7 actions.
  Both flashlight actions are `CameraManager` torch calls with no public intent, so they are
  deliberately unmapped: trainable and validatable, never bindable. Extending it is an app decision.

*(The #31 markdown link-check closed 2026-08-14: `test_docs.py` now sweeps **every tracked markdown
file** — `agent_docs/` included, which is where the worst reference rot was — and `ci.yml`'s fast job
runs it as a named step.)*

---

## 2. Device / manual test legs still open

- ~~**`ToolCallDeviceTest` has never been run**~~ — **RUN 2026-08-14**, fails at generation; see §1a.
  Still the open #37 checkpoint, but the open question is now merge numerics, not convergence.
- ~~**`PostMergeNumericsTest` has never appeared in a recorded suite run**~~ — **CLOSED 2026-08-14.**
  `mergedWeightsChangeTheComputationAndStayNumericallySane` **PASSES in 87.07 s** against a pristine
  `TRAIN=1` SmolLM2-135M-Instruct package on the S21 FE `SM-G990B` / Android 15 / arm64-v8a, run
  before any training suite touched the package. Note what it does *not* assert: that the merged
  delta is the *correct* one — see §1a.
- **RE-VALIDATE EVERY RECORDED TRAINING RESULT.** The merge wrote transposed weights until 2026-08-14,
  so the recorded device suite (15/15, 798 s) and #18/#19's train→merge→generate checkpoints were all
  measured against a system that no longer exists. They are not evidence for the current code. Highest
  priority device work.
- **Re-run `ToolCallDeviceTest`** — the #37 gate. Never run since the merge fix; now a fair test.
- #9: on-device atomic-overwrite-under-kill, and offline-vs-device byte-identical `.bin` parity.
  **Note:** `write_raw_tensor_atomic`'s comment claims offline and device merges are byte-identical.
  Given the device side transposed everything, that claim was untrue and unchecked — verify whether the
  offline Python merger has the same defect before trusting exported packages.
- **#8: `ObservedInit.transposed` is never assigned anywhere**, so `transposePolicy` is `"no_transpose"`
  by omission on all 60 entries and in the test fixtures. The contract describing weight layout was
  declared and never implemented. The device fix deliberately does not trust the field (it transposes
  and verifies against the declared shape); whether the exporter should emit
  `already_transposed_for_inference` is #8's call.
- **Strengthen the assertions that let the transpose hide for months** — `TrainMergeGenerateTest`'s
  `isNotEmpty()` (passes on 48 newlines) and `PostMergeNumericsTest`'s arbitrary-token-id `PROBE`
  (near-uniform by construction). See Operational knowledge → "Magnitude-based checks cannot detect a
  permutation".
- #21: Kotlin `VariantSelector` parity + the on-device load leg. #22: a real authenticated adapter
  upload with a checkpoint factor read.
- **The rewritten showcase app has never been run on hardware** (§1b). It compiles and its pure state
  logic is JVM-tested, but no screen has been walked on a device. Highest-value manual leg available:
  it exercises the whole public facade in one pass, including the six gaps just closed.
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
