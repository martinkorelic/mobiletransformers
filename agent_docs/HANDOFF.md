# Handoff — what is still open

**Branch:** `restructure` · **Rewritten:** 2026-08-15 (end of the device-defect cycle)

This file is **only what is left to do**. Completed work is not recorded here — it lives in
`IMPLEMENTATION_ORDER.md`'s per-plan self-checks, in the docs, and in git. Rewritten each cycle.

> ## Read these first
>
> 1. **`IMPLEMENTATION_ORDER.md` → "Operational knowledge (permanent)"** — environment and profiles,
>    the gotchas that each cost a cycle, the ORT engine separation, the device workflow, the
>    layer-identity problem, the recurring failure shape, and the recorded device suite. Everything a
>    cold agent needs *regardless of cycle* lives there, not here.
> 2. **`IMPLEMENTATION_ORDER.md` → "How to execute a plan"** — the implementer protocol. Follow it.
> 3. The per-plan **self-check blocks** in the same file — each records what is proven and how.
>
> **Do NOT read `agent_docs/audits/` as a to-do list.** They are 2026-08-07 snapshots with no closure
> annotations. This file is the authoritative list of what is open.

---

## Where the project stands

**36 / 37 plans done.** The one open plan is **#32, which is licence-gated rather than work**.

**Chat, tool calling and on-device training all work on hardware** against
`mobiletransformers/functiongemma-270m-it` (S21 FE `SM-G990B`, Android 15, arm64-v8a) as of
2026-08-15 — including the first Gemma-3 training steps ever run on a phone, which closes the
long-standing "NOT proven: on-device Gemma-3 training" note. Five defects were found and fixed by
actually using the app; each is documented where it lives (`docs/CONFIGURATION.md` → Memory profiles,
`docs/COOKBOOK.md` §3/§4/§6, and the class docs on `sampling::effectiveVocabSize`, `Tasks.resolve`,
`ModelRuntimeFactory.enginesAvailableFor`, `ToolCallSupport`,
`LLMRepository.releaseInferenceRuntime`).

> **The lesson worth carrying, because it will recur.** The previous handoff said "there is no open
> engineering item and no known outstanding defect" over a fully green host suite. Four defects then
> sat between a user and every feature of the app, and the first hour of manual use on real hardware
> found all four. **A green host suite means the host suite knows of no defect.** Nothing in it had
> ever run FunctionGemma end to end on a phone.

**Host gates (measured 2026-08-15 — do not copy forward, re-measure):**

```
Python 568 passed / 11 skipped · guard 9 · C++ 46 · uv lock --check clean
Kotlin JVM 358 = 335 library + 23 app (--rerun-tasks, read from build/test-results/**/*.xml)
make check · android-build · publish-local · consumer-app · externalNativeBuildDebug all pass
```

Read the JVM counts out of `build/test-results/**/*.xml`, never off the console: Gradle reports
`UP-TO-DATE` as success, so a suite that never executed looks identical to one that passed.

```bash
uv sync --frozen --group dev --python 3.10 && make check    # ALWAYS reset the profile first
make test-cpp
cd android/MobileTransformers && JAVA_HOME=/opt/android-studio/jbr ./gradlew --rerun-tasks \
  :MobileTransformers:testDebugUnitTest :MobileTransformersApp:testDebugUnitTest
```

**Building the APK: source `.env` first.** `make android-build` does **not**, so the APK it produces
carries no `HF_TOKEN` and silently cannot pull private repos (FunctionGemma is private):

```bash
cd android/MobileTransformers && set -a && . ../../.env && set +a && \
  JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformersApp:assembleDebug
adb install -r MobileTransformersApp/build/outputs/apk/debug/MobileTransformersApp-debug.apk
```

**Recorded device suite** — the result depends on which package is installed, and the cache holds one:

| package on device | result |
| --- | --- |
| SmolLM2-135M `TRAIN=1 RAG=1` (decoder) | 23 tests, 0 failures, 3 skipped, 3219 s (2026-08-14). **Predates every 2026-08-15 change** — re-measure. |
| all-MiniLM-L6-v2 `TASK=text-classification TRAIN=1` (encoder) | 19 tests, 0 failures, 16 skipped — predates the merge-transpose fix. Re-measure before relying on it. |

**You do not need a full re-export to re-push.** `build/pkg` holds the exported decoder and
`build/pkg-functiongemma` the Gemma-3 one; steps 2–4 of `scripts/device_package.sh` are pure file
movement (~15 s) against ~30 min and two profile switches for `make device-package`.

---

## 1. Device / manual legs still open

- **Re-run the recorded instrumented suite.** Both rows above predate the 2026-08-15 changes. The
  vocab clamp is a no-op for SmolLM2 (its declaration is correct), so any movement there is a real
  finding.
- **Prove the vocab clamp specifically.** Chat works on FunctionGemma, but that could be either the
  clamp holding *or* the repaired package. `build/pkg-functiongemma`'s tokenizer config and manifest
  checksums have been corrected in place, so pushing it tests the **exporter** fix. To test the
  **clamp** — which is what protects the package published on the Hub and every package already on a
  device — push a copy with `vocab_size` set back to 262146 (and its sha256 updated) and confirm
  generation still works, with `declared vocab_size 262146 exceeds the graph's logits width` in
  logcat.
- **GenAI is the one 2026-08-15 fix not exercised on hardware.** Expect the engine picker to offer
  NATIVE only for FunctionGemma, and a load that names `supportedEngines=[native]` if GenAI is forced.
- **`classify()` has never run on a device.** The scoring maths is JVM-tested (`ClassifierScoringTest`,
  7 cases); the forward pass is not. `ClassifierSession` deliberately bypasses
  `ORTRetriever.createEmbeddingModel` and opens `createEmbeddingSession` against **`inference/`** with
  `embeddingDim = numLabels`. The premise is that `inference::generateEmbedding` returns the raw first
  output tensor with no pooling — read `cpp/inference.cpp:235-295` and confirm before trusting it.
  Needs a `TASK=text-classification` package on device.
- **#21's on-device Hub pull** (`make device-hub-test REPO=<org>/<name>`, drives `HubPullDeviceTest`)
  is still unrun. It only became runnable when `android.permission.INTERNET` was added — it was
  declared nowhere in the tree, and MockWebServer on the JVM has no permission model, so no existing
  test could see it.
- **The scheduled-training start delay is untested on hardware.** `initialDelayMinutes` is JVM-tested
  through the codec (`SchedulerDelayTest`); whether WorkManager honours it under Doze on this device
  is exactly the thing a host test cannot answer.
- **#9: on-device atomic-overwrite-under-kill** remains untested.
- **#34: multi-hour / Doze / FGS-quota behaviour** — a recorded DEBT; the plan's own DoD does not
  require it.
- **#21: Kotlin `VariantSelector` parity. #22: a real authenticated adapter upload** with a checkpoint
  factor read.

---

## 2. Code — what is left

### 2a. `ORTTokenizerNative` never reads `chat_template.jinja` — highest-value open item

It looks for a `chat_template` key inside `tokenizer_config.json`, and the exporter writes the
template to a **sibling file**. So `tokenizer.chatTemplate` is null for every package shipped so far,
`ORTConversationState` is never constructed, and **no plain-chat prompt on the device is ever wrapped
in the model's turn format**. A model trained on `<start_of_turn>user … <start_of_turn>model` is being
handed a bare string, which is a large part of why small-model output looks worse on device than it
should.

The tool-calling path does not depend on this — `ToolPromptBuilder.prompt()` frames its own turns,
deliberately — but plain chat does.

Fixing it means reading the sibling file **and** rendering a probe conversation at load to check
Pebble can actually evaluate the template, falling back to today's behaviour when it cannot:
FunctionGemma's template uses `namespace()`, `dictsort` and macros that Pebble does not support, and a
template that throws mid-generation would be worse than none.

### 2b. `ClassifyScreen` + `ClassifyViewModel`, and re-add `Destination.Classify`

`MobileTransformerModel.classify` is implemented and the drawer already computes
`supportsClassification`; the destination was removed from `app/Navigation.kt` rather than shipped as
a stub. Re-add the enum entry, its `Availability.Hidden`-unless-`supportsClassification` branch (the
removed branch is in git), the icon in `MainActivity`, and a screen showing per-label probability bars
— ideally before/after a training run, which is the encoder story's payoff.

### 2c. `id2label` export is written but unproven

`export/pipeline.py::_read_id2label` copies the HF config's labels into
`inference/optimum_config.json` for `SEQUENCE_CLASSIFICATION` only. It has **no test** — add one to
`tests/export/`, and re-export an encoder package to confirm the file carries the names. Without them
`classify()` fails closed by design (a `LABEL_3` is not an answer).

### 2d. Training foreground-notification progress

`TrainingWorker.foregroundInfo` takes a `progress` parameter and is only ever called with `null`, so
the ongoing notification never advances. Call `setForeground` on each optimizer step with
`setProgress(totalSteps, currentStep, false)` and text naming the step and loss. The runtime
permission is now requested (`app/NotificationPermission.kt`).

### 2e. Route downloads through `PackageDownloadWorker`

It is complete, still has no caller, and is what would make a pull survive leaving the app. Promote it
to a foreground worker with progress + Cancel and have the Models screen observe its `WorkInfo`
(`KEY_PHASE`/`KEY_BYTES_*` are already emitted). Note it hardcodes `NetworkType.UNMETERED`, which
would silently never start on mobile data — expose that as a "Wi-Fi only" switch.

### 2f. Calibrate or delete `MemoryHeadroom`

It exists, is tested, and `LLMRepository` logs its estimate before opening a training session — but it
is **not shown to the user**, deliberately: it warned about a model that fits. The model did fit; the
allocator did not (see `docs/CONFIGURATION.md` → Memory profiles). Before putting it in front of
anyone, calibrate the `TRAINING_OVERHEAD` factor against RSS traces from several real `low_mem` runs.
As written it is one data point.

> Read `MemoryHeadroomTest.aDiskSizedRuleWouldHaveRefusedAWorkingSession` before "improving" it. The
> first version sized the estimate from bytes-on-disk, and the device disproves that: `inference/` is
> 3.5 GB and chat works with 2.4 GB available, because ONNX external initializers are mmapped.

### 2g. Two levers if training memory is ever tight again

`maxSequenceLength` is 512 against a tool-call corpus whose rows are far shorter, and `batchSize` is 4.
Both are in Configuration → Dataset/Training. Not needed as of 2026-08-15 — `low_mem` was enough — but
they are the next levers for a bigger model or a smaller phone.

### 2h. `cpp/inference.cpp:288-294` returns a dangling pointer

`generateEmbedding` returns `output->GetTensorMutableData<float>()` from an `Ort::Value` destroyed at
scope exit, and `native-lib.cpp`'s error path `delete[]`s that same pointer (never `new[]`-allocated).
It happens to work today. **Pre-existing — it affects RAG now and would affect `classify()`.** Fix it
before relying on either path under memory pressure. (The generation path had the identical bug and
was fixed by `InferenceSessionCache::last_output`; this is the same fix in the embedding session.)

### 2i. Build a real catalog of small models, exported and published to the org

The catalog is currently one working entry and three placeholders (§2j below). The feature it is
meant to demonstrate — "pick a model, tap install, it runs" — needs a shelf with something on it.

Scope, roughly in order of payoff:

- **Pick 4–6 SLMs that are actually good on a phone** and span the interesting axes: a chat decoder
  (SmolLM2-360M / Qwen2.5-0.5B-Instruct), a tool-caller (FunctionGemma-270M, already done), a tiny
  one that trains fast enough for a live demo (SmolLM2-135M), and an encoder for the classification
  and RAG stories (all-MiniLM-L6-v2). Check each against `config/registry/architecture.py` first — an
  architecture with no `ArchitectureSpec` is an export project, not a catalog entry.
- **Export each with the flags the story needs** (`TRAIN=1` at minimum; `RAG=1` for the retrieval
  demo) and publish under the `mobiletransformers` org, then flip `published: true` and replace the
  estimated `approxSizeMb` with the real figure.
- **Prefer int4/int8 variants where the export supports them.** Every published package today ships an
  fp32 inference graph (`cpu-int4` is a naming debt, §4), and 3.5 GB is a lot to ask of a first-run
  user. A genuinely quantized variant would cut both the download and the memory ceiling.
- **Verify each end to end on a device before publishing**, not just that it exports. The 2026-08-15
  cycle is the argument: FunctionGemma exported cleanly and was unusable on a phone for four separate
  reasons.
- **Record the per-model numbers** the catalog claims — size, tokens/second, whether training fits —
  from a real run. `RuntimeCapabilities` and the chat stats line now report all three.

Two rules the entries must keep:

- **A catalog entry names an exported package** — a repo with `mobiletransformers_manifest.json` at
  its root — never a base HF model id. `fromPretrained` reads the manifest first to plan the download,
  so a plain model id fails on the first request in a way that looks like a broken app.
- **`published: false` is honest, and better than a 404.** Today's three placeholders
  (`SmolLM2-135M-Instruct`, `Qwen2-0.5B`, `all-MiniLM-L6-v2`) render with a disabled Install and an
  explanation. Flip the flag only once the repo exists and a device has loaded it.

### 2j. Re-export and re-publish FunctionGemma

The package on the Hub still carries the wrong `mobiletransformers_tokenizer_config.json`
(`vocab_size: 262146`, 12/12/12 heads/layers, `type: unknown`, `bos: null`). The device-side clamp
covers it, but the file is wrong at the source. `build/pkg-functiongemma` is already repaired locally,
including its manifest `sha256`/`fileSizes`.

### 2k. Docs, README and the showcase recording

The reference docs were brought up to date on 2026-08-15 (`docs/COOKBOOK.md` §3/§4/§6,
`docs/CONFIGURATION.md` → Memory profiles). What is left is the **front door**, which is what anyone
evaluating the project actually reads:

- **`README.md` still describes the six-tab app** that the drawer navigation replaced, and says
  nothing about what now demonstrably works on hardware. It should open with the one-paragraph claim
  (export → pull → chat → fine-tune → merge → tool-call, on a phone) and the measured evidence for it.
- **Showcase GIFs.** `docs/ortransformer-feature.gif` predates the rewrite; `base-model.gif` and
  `on-device-trained.gif` are the before/after pair and should be re-recorded against the current UI.
  The recording plan — 7 clips plus a hero GIF — is in
  `~/.claude-personal/plans/review-what-we-have-partitioned-seal.md` §4.3. Worth capturing now that
  the app is presentable: the model bar with a live status dot, streaming chat with the per-turn
  token/context line, a tool call rendered as a card with **Run**, and the loss curve moving during a
  real training run.
- **Record the numbers the GIFs imply** in prose beside them — tokens/second, steps/second, the loss
  drop — so a reader who cannot run it still learns what the hardware does.
- `docs/ANDROID_SDK.md` and `docs/PUBLIC_API.md` gained fields this cycle
  (`RuntimeCapabilities.toolCalling` / `.trainingParameterCount`, `ToolCallResult.NoCall`,
  `GenerationResult.promptTokenCount` / `.contextLimit`, `TrainingScheduleConfig.initialDelayMinutes`).
  Sweep both for anything the 08-15 additions left unmentioned.

### 2l. Two ideas worth taking from `Edge-Intelligence-Lab/MobileFineTuner`

Its one distinctive feature is **on-device evaluation** — before/after perplexity as a number that
moves when you train, the most convincing proof that fine-tuning did anything; host-side machinery
exists in `research/evaluation/` and `docs/mobile_evaluation.md`. Second, a **live resource panel**:
`runtime/MemoryProbe`, `scheduler/ThermalGuard` and the CSV trace `TrainingWorker` writes are all
present and surfaced nowhere.

### 2m. `agent-dataset` intent mapping

`ANDROID_INTENT_BY_ACTION` covers 5 of the corpus's 7 actions. Both flashlight actions are
`CameraManager` torch calls with no public intent, so they are deliberately unmapped: trainable and
validatable, never bindable. Extending it is an app decision.

---

## 3. Not engineering

- **The licence — the only real v1.0 blocker.** CC-BY-NC-4.0 contradicts the consumable-AAR goal. It
  is a rights-holders decision (both authors in `CITATION.cff`); the four sites are listed in
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
  `base_layer_name` into checkpoint space for the **MARS** mapping only, so generalising it would
  change MARS's `peft_mapping` keys for encoders. The other (`module_name.replace(...)` with the
  result discarded) is **dead, and load-bearing by being dead**: the loop below it works in raw space,
  so assigning it would break `relative_path`. Both want a deliberate look with the MARS tests in
  hand, not a sweep.
- **The stage-path guard carries 2 allow-listed sites**, both legitimate producers that write a layout
  down once (`export/pipeline.py`, `ModelPackageInstaller.kt`).
- **`export-rocm` is a declared-but-empty group** — ROCm wheels need a dedicated AMD index.
- **Android ORT-training AAR** — `third_party/onnxruntime/manifest.json`'s Android fields are still
  null (`ndk_version`/`abis`/`aar_sha256`); that leg was never built.
- **Broadening the top-level `__all__`** — still #32's call.
- **macOS cannot run the training side** without rebuilding the `cp312-linux_x86_64` wheel for
  `macosx_arm64`. Details in `IMPLEMENTATION_ORDER.md`'s permanent section.
- **No dark theme by default.** `AppThemedContent(isDarkMode = …)` now works and both schemes exist,
  but the default stays light — switching it for everyone is a change nobody asked for.
- **Scheduled runs cannot promise a wall-clock start.** `initialDelayMinutes` is a floor, not an
  appointment: WorkManager batches deferrable work and Doze can hold it. An exact start needs
  `AlarmManager.setExactAndAllowWhileIdle` + `SCHEDULE_EXACT_ALARM`, which Play restricts to alarm
  clocks and calendar reminders. Accepted, and stated in the UI rather than worked around.
- **Reclaimable disk (untracked, gitignored):** `build/` ~2.5 G, `.venv-genai-spike/` 5.2 G — the
  latter is recreated by `spikes/genai_external_swap/build_tiny_genai_model.sh`.

---

## 5. Traps that have each cost a cycle — check these before debugging

- **A green host suite proves nothing about the device.** See the banner in "Where the project stands".
- **`make android-build` does not source `.env`.** The APK builds fine and cannot pull private repos.
- **Material 3 fills omitted colour roles from its baseline palette, which is purple** — not from
  `surface`. Any role you do not set (`surfaceContainer*`, `surfaceTint`, `inverse*`, `scrim`) arrives
  lilac; `TopAppBar` and `ModalDrawerSheet` paint from `surfaceContainer`/`surfaceContainerLow`. All
  four schemes now set them explicitly.
- **`strokeWidth = 1f` in a Compose `Canvas` is one PIXEL, not one dp**, and a line drawn at exactly
  `y = height` is half-clipped by the canvas bounds. Both were true of the training charts and made
  the axes look absent.
- **A `launch` that throws does not deliver to whoever `join()`s it.** It goes to the scope's parent,
  and `LLMRepository`'s parent is a bare `Job()` — i.e. the thread's default handler, i.e. process
  death. Both the generation and training paths now capture into a `last*SessionFailure` field and
  re-raise on the caller's coroutine.
- **Release native sessions on a resource check (`!= null`), never a state check.** The training path
  released the inference session only `if (llmState == ReadyGenerate)`, and four of the seven states
  can hold a live session.
- **Exported `training_config.json` carries no `deviceOptions` section**, so `parseTrainingArguments`'
  fallback is the real setting for every training run — which is how `high_perf` reached training.
- **An out-of-memory death is a SIGKILL, not an exception.** Look for `lmkd: Reclaim` and
  `exited due to signal 9` in logcat; there will be no stack trace and no `FATAL EXCEPTION`.

---

## Cycle protocol

Follow `IMPLEMENTATION_ORDER.md` "How to execute a plan". For this repo specifically:

1. **Reset the profile before `make check`** — `uv sync --frozen --group dev --python 3.10`. More
   "broken repo" reports have come from a leftover export/training venv than from any real defect.
   `scripts/device_package.sh` leaves the tree on the training profile.
2. **Flip a `Done` box only when the self-check holds**, with the evidence (device, OS, ABI, date)
   inline.
3. **Assert across the seam, and check the assertion can FAIL.** This keeps paying. On 08-15 the
   exporter fix was verified by replaying the old field routing against FunctionGemma's real config
   and confirming it produced 262146 — the bug reproduced before the fix was trusted.
4. **When you empty a ratchet, re-point it or delete it** — never leave it scanning the void.
   `test_gate_ratchet` catches this; on 08-15 it caught the `tokenizer_export.py` ruff entry.
5. **When you correct an earlier claim, correct it where the claim lives**, not only in a new entry.
6. **Walk the app on hardware before claiming a cycle is done.** Every defect in the 08-15 cycle came
   from that walk and none from the suite.
