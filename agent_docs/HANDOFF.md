# Handoff — what is still open

**Branch:** `restructure` · **Reset:** 2026-08-14 · **Showcase-app cycle appended 2026-08-15 (see §0)**

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

**36 / 37 plans done · 108 / 111 self-check boxes.** The one open plan is **#32, which is
licence-gated rather than work**: a relicense decision only the two rights-holders can make, a version
tag, and the release gate that waits on both. **There is no open engineering item and no known
outstanding defect.**

**#37 closed 2026-08-14** (§1a) when its differentiation demo passed on hardware, after the merge
transpose defect that had been silently corrupting every on-device merge was found and fixed. The
remaining work is verification breadth, not features — see §2.

**Host gates (measured 2026-08-14 after the merge-transpose fix — do not copy forward, re-measure):**

```
Python 558 passed / 11 skipped · C++ 39 · Kotlin JVM 252 (library) + 6 (app) · guard 8 · parity OK
uv lock --check clean · make android-build / publish-local / consumer-app all pass
```

**Device gate — the full instrumented suite now runs end to end for the first time:**

```
23 tests / 0 failures / 3 assumption-skips / 3219 s
S21 FE SM-G990B · Android 15 · arm64-v8a · pristine TRAIN=1 SmolLM2-135M-Instruct
skips: encoder package not installed, FEDERATION_ENABLED=false (both by design)
```

Read the JVM counts out of `build/test-results/**/*.xml`, never off the console: Gradle reports
`UP-TO-DATE` as success, so a suite that never executed looks identical to one that passed. Use
`--rerun-tasks`.

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
| SmolLM2-135M `TRAIN=1 RAG=1` (decoder) | **23 tests, 0 failures, 3 skipped, 3219 s (2026-08-14, post-transpose-fix)**. Supersedes the earlier "15 / 15, 798.0 s", which predates both the transpose fix and the three `PostMergeNumericsTest` cases — that run measured a system where every merged weight was stored transposed. |
| all-MiniLM-L6-v2 `TASK=text-classification TRAIN=1` (encoder) | 19 tests, 0 failures, 16 skipped — **not re-run since the transpose fix.** The encoder legs train but the recorded run predates the fix; re-measure before relying on it. |

**The device currently holds the train-capable SmolLM2-135M-Instruct DECODER package** (pushed
2026-08-14). **Push a fresh one before a suite run** — that precondition still stands, and the recorded
15/15 above predates both the transpose fix and the three mutating `PostMergeNumericsTest` cases, so
re-measure rather than copying it forward.

What *no longer* applies: the package used to degrade **during** a run, so a suite of 23 tests came
back with 3 failures that were all one earlier test contaminating later ones. `PristinePackageRule`
now captures and restores the mutable artifacts around every class that trains or merges, so a suite
that starts clean stays clean and also **ends** clean. See "The device-suite mutation hazard" in
`IMPLEMENTATION_ORDER.md` — it had bitten once before and the per-class fix did not hold.

**You do not need a full re-export to re-push.** `build/pkg` holds the exported decoder; steps 2–4 of
`scripts/device_package.sh` (reshape → `adb push` → `chmod -R 777` → stage `mt_genai_spike`) are pure
file movement and take ~15 s, against ~30 min and two profile switches for a full
`make device-package`. The profile switching is also how gotcha 13's `onnxruntime` collision gets
reached, so prefer the short path unless you are testing the export itself.

---

## 0. Showcase-app usability cycle — 2026-08-15 (READ FIRST if you are continuing it)

A large showcase-app cycle landed on 2026-08-15. It is **green and installed on hardware**; what
follows is only the part that was deliberately left. Everything below §0 predates this cycle — where
the two disagree, §0 wins (in particular, §1b's "what is left on the app" list is now largely closed).

**Gates as measured 2026-08-15, after the cycle:**

```
Python 560 passed / 11 skipped · guard 9 · Kotlin JVM 300 (library + app, --rerun-tasks)
make android-build passes · APK installed on S21 FE SM-G990B, launches, no FATAL in logcat
```

### 0a. Two defects the user hit, both fixed — do not re-introduce

- **Loading an installed package failed.** The Models screen loaded `baseModelId ?: sanitizedRepoId`,
  and the manifest's `baseModelId` names the model a package was exported **from**
  (`google/functiongemma-270m-it`), not the repo it was pulled **from**
  (`mobiletransformers/functiongemma-270m-it`). It sanitized to a different, absent cache directory
  and reported an installed package as missing. Fixed by recording the installing repo id:
  `packages/InstallRecord.kt` is written **inside the staging tree before publish**, so it lands with
  the same rename as the package; `CacheIndex.InstalledPackage.repoId` reads it, falling back to
  un-sanitizing the directory name for legacy/`adb push`ed trees. **`repoId` is the load key;
  `baseModelId` is provenance and must never be used as one.**
- **A download was indistinguishable from a hang.** `PackageDownloader` fired progress only after a
  whole file completed, and a package's weights are one or two files of 1–4 GB — so a working pull
  showed "0 / N files" for ten minutes. `OkHttpClient()` also carried OkHttp's default **10-second
  read timeout**, which is the most likely cause of the FunctionGemma pull that appeared to stall.
  Now: byte-level progress with rate and ETA (`manifest.fileSizes` gives the denominator before the
  first GET), `PackageDownloader.defaultClient()` with 60s read / no call timeout, `ensureActive()`
  **inside** the read loop so a 3 GB transfer cancels promptly, and a Cancel button. The `.partial`
  survives cancellation and Range-resume picks it up.

### 0b. What else landed

Shell: drawer navigation (`app/Navigation.kt` — `Destination.availability(ModelState)` is pure and
JVM-tested) replacing the six-tab `ScrollableTabRow`; sub-tabs *within* destinations only; `AppTheme.FRI`
(the red, light-only, as the user chose); a top-anchored snackbar fed by `AppSnackbar`; a persistent
`views/ModelBar.kt` showing repo id, engine, capability chips and live download progress on every screen.

Features: a bundled model catalog (`assets/model_catalog.json` + `app/ModelCatalog.kt`); Configuration
split into five tabs with dropdowns over every closed set — including a **Device** tab, which had been
on every public config and editable from nowhere, and a **PEFT** picker wired to the never-called
`applyPeft`; a hand-rolled Compose loss chart (`views/LossChart.kt`) plus a readable event log
(`TrainingEvent.describe()` used to be `toString()`); a scheduled-run queue panel via the new
`TrainingScheduler.observeChunks` (`observe` had no caller, so scheduling reported only a UUID).

SDK additions, each a facade gap the app found: `packages/PackageTask.kt` (reads the `task`/`id2label`
the exporter already wrote to `inference/optimum_config.json` — nothing on the device had ever read
it), `RuntimeCapabilities.task`/`isClassifier`/`supportsClassification`, `TrainingScheduler.observeChunks`
+ `ScheduledChunk`, `Tasks.TASKS` (the trainer's own dispatch, so a picker cannot drift from it),
`MobileTransformerModel.classify` + `internal/runtime/ClassifierSession.kt`.

**Tool calling now works against FunctionGemma at all.** It does not emit JSON — it emits
`<start_function_call>call:name{key:<escape>value<escape>}<end_function_call>` — so
`FunctionCallValidator`'s JSON parse rejected every well-formed call it made, and no further
fine-tuning could have changed that. `agent/ToolCallParser.kt` splits parsing from validation
(`ToolCallParser.Json` / `.FunctionGemma` / `.forModel(id)`), and `agent/ToolPromptBuilder.kt` renders
the allowlist as a declaration the model can actually read — `generateToolCall` previously sent the
bare instruction with **no tool declaration at all**. The boundary is unchanged: a parser only chooses
*which candidate* to check, and `ToolCallParserTest` asserts an undeclared action is still refused
whichever dialect it arrives in.

### 0c. What is left — in priority order

1. **`ClassifyScreen` + `ClassifyViewModel`, and re-add `Destination.Classify`.**
   `MobileTransformerModel.classify` is implemented and the drawer already computes
   `supportsClassification`; the destination was removed from `app/Navigation.kt` rather than shipped
   as a stub. Re-add the enum entry, its `Availability.Hidden`-unless-`supportsClassification` branch
   (the removed branch is in git), the icon in `MainActivity`, and a screen showing per-label
   probability bars — ideally before/after a training run, which is the encoder story's payoff.
2. **`classify()` has never run on a device.** The scoring maths is JVM-tested
   (`ClassifierScoringTest`, 7 cases) and the forward pass is not. `ClassifierSession` deliberately
   bypasses `ORTRetriever.createEmbeddingModel` (which would resolve the embedding stage, open the
   embedder's tokenizer, build a vector store, and `DimensionRegistry.requireSupported` would reject
   a 2-label head) and instead opens `createEmbeddingSession` against **`inference/`** with
   `embeddingDim = numLabels`. The premise is that `inference::generateEmbedding` returns the raw
   first output tensor with no pooling — read it in `cpp/inference.cpp:235-295` and confirm before
   trusting the path. Needs `TASK=text-classification` package on device (`make device-package`).
3. **`id2label` export is written but unproven.** `export/pipeline.py::_read_id2label` copies the HF
   config's labels into `inference/optimum_config.json` for `SEQUENCE_CLASSIFICATION` only. It has
   **no test** — add one to `tests/export/`, and re-export an encoder package to confirm the file
   carries the names. Without them `classify()` fails closed by design (a `LABEL_3` is not an answer).
4. **Training foreground-notification progress.** `TrainingWorker.foregroundInfo` takes a `progress`
   parameter and is only ever called with `null`, so the ongoing notification never advances. Call
   `setForeground` on each optimizer step with `setProgress(totalSteps, currentStep, false)` and text
   naming the step and loss. The runtime permission is now requested
   (`app/NotificationPermission.kt`) — it never was before, so on API 33+ **every** training
   notification had been silently dropped by the system.
5. **Route downloads through `PackageDownloadWorker`.** It is complete, still has no caller, and is
   what would make a pull survive leaving the app. Promote it to a foreground worker with progress +
   Cancel and have the Models screen observe its `WorkInfo` (`KEY_PHASE`/`KEY_BYTES_*` are already
   emitted). Note it hardcodes `NetworkType.UNMETERED`, which would silently never start on mobile
   data — expose that as a "Wi-Fi only" switch.
6. **Docs and the showcase recording.** `README.md` and `docs/COOKBOOK.md` still describe the six-tab
   app. The recording plan (7 clips + a hero GIF to replace `docs/ortransformer-feature.gif`) is in
   `~/.claude-personal/plans/review-what-we-have-partitioned-seal.md` §4.3.
7. **Two ideas worth taking from `Edge-Intelligence-Lab/MobileFineTuner`** (researched this cycle; its
   app is otherwise thinner than ours). Its one distinctive feature is **on-device evaluation** —
   before/after perplexity as a number that moves when you train, which is the most convincing proof
   that fine-tuning did anything; host-side machinery exists in `research/evaluation/` and
   `docs/mobile_evaluation.md`. Second, a **live resource panel**: `runtime/MemoryProbe`,
   `scheduler/ThermalGuard` and the CSV trace `TrainingWorker` writes are all present and surfaced
   nowhere.

### 0d. Two things flagged, not fixed — decide deliberately

- **`cpp/inference.cpp:288-294` returns a dangling pointer.** `generateEmbedding` returns
  `output->GetTensorMutableData<float>()` from an `Ort::Value` destroyed at scope exit, and
  `native-lib.cpp`'s error path `delete[]`s that same pointer (which was never `new[]`-allocated).
  It happens to work today. **Pre-existing — it affects RAG now and would affect `classify()`** —
  and it was left alone because it is a C++ lifetime change outside the cycle's scope. Fix it before
  relying on either path under memory pressure.
- **`FriLightColors` is not internally consistent.** Primary is the project red `#e03229`, but
  `primaryContainer` is a leftover green (`#E8F5F0`) and `tertiary` a leftover brown (`#7B5A3C`) from
  an earlier theme, so filled chips and containers read as off. The user explicitly chose "switch back
  to FRI, leave the palette as-is" this cycle; a red-family recolor is a three-line change whenever
  they want it. There is also still no dark scheme — `AppThemedContent(isDarkMode = …)` is an unused
  parameter.

### 0e. Catalog entries need confirming — the one place this cycle guessed

`assets/model_catalog.json` ships four entries and **only `mobiletransformers/functiongemma-270m-it`
is marked `published: true`** — the one the user confirmed downloads. The other three
(`SmolLM2-135M-Instruct`, `Qwen2-0.5B`, `all-MiniLM-L6-v2`, all under the `mobiletransformers` org) are
`published: false`, so they render with a disabled Install and an explanation rather than 404-ing on
tap. Their `approxSizeMb` figures are estimates. **Export and push them, then flip the flag and
correct the sizes** — or delete the entries. The public HF API returns nothing for
`author=mobiletransformers`, so the org's repos could not be enumerated to check.

Catalog entries must name **exported packages** (a repo with `mobiletransformers_manifest.json` at its
root), never a base HF model id — `fromPretrained` reads the manifest first to plan the download, so a
plain model id fails on the first request in a way that looks like a broken app.

---

## 1. Code — what is actually left

### 1a. #37's differentiation demo — **CLOSED 2026-08-14. The last open engineering box in the project.**

The chain is **per-user action set → on-device fine-tune → validated tool call → dry-run intent**, and
it now runs end to end on hardware.

> ### ✅ `ToolCallDeviceTest` PASSES — S21 FE `SM-G990B` / Android 15 / arm64-v8a, 2026-08-14
>
> 2 tests / 0 failures / 754 s, against a freshly pushed pristine `TRAIN=1` SmolLM2-135M-Instruct
> package:
>
> ```
> steps=108 lossDrop=99.5%
> instruction='wake me at 07:30'
> raw='{"actionName": "set_alarm", "parameters": {"time": "07:30"}}<|im_end|>' -> Accepted
> ```
>
> The second test confirms an action the app never declared is **still refused** after fine-tuning —
> training the model to emit calls did not widen the reachable action set.
>
> This was only a fair test after the merge transpose defect was fixed. Two earlier runs on the same
> package with the same 99.5% loss drop emitted token 198 (newline) 48 times: **the model had learned
> the task all along and the merge was destroying the result on the way out.** The `<|im_end|>` in the
> output is the EOS fix landing — the completion terminates instead of running to `maxNewTokens`.
>
> Full history — the transpose root cause, the uninitialized LoRA scale found alongside it, the two
> wrong hypotheses, and the sound elimination that located it — is in `IMPLEMENTATION_ORDER.md` →
> "The #37 demo run". Read that before touching merge code.
>
> *(This block has been corrected twice in place. It previously read "compiles but has NOT been
> executed on hardware", then "ran 2026-08-14 and FAILED … has NOT been re-run since the fix". Both
> are now false.)*

Rebuild the demo data with either or both sources:

```bash
mobiletransformers agent-dataset --source google/mobile-actions --output build/agent
mobiletransformers agent-dataset --source generated --allowlist build/agent/action_schema.json --output build/user
```

Notes that survive the close, for anyone extending the demo:

- **`gradAccumSteps = 1` is pinned in the test deliberately.** At the default 4 a bounded run applies
  no optimizer update at all and trains nothing. Do not "tidy" it.
- **The two dataset sources answer different halves.** The 5,794-row `google/mobile-actions` corpus
  teaches the JSON tool-call *format* — the generic, data-hungry part. The generated per-user set is a
  short pass on top that teaches *this user's* actions, and is what the personalization differentiator
  rests on. The passing run uses only the second; the first is the lever if a larger action set is
  added and format fidelity drops.
- **A demo that "passes" via the refusal path would show nothing** — the validator rejecting garbage is
  already covered by JVM tests. The assertion is on an *accepted* call reaching `IntentBinder.dryRun`,
  and it stays that way.
- **`maxSteps` is an upper bound, not a target**: training stops at the end of the epoch. 216 rows at
  batch 2 gives the 108 steps above.

> **SUPERSEDED 2026-08-15 — FunctionGemma is now exportable train-capable, and the pin was never the
> reason it was not.** This paragraph used to read: *"on-device training stays blocked by
> `ort-training-local`'s `transformers==4.46.2` … floating that pin has broken `get_peft_model`
> before."* Both halves turned out to be wrong.
>
> The experiment ran in a throwaway sibling group with only `transformers` raised to 4.57.6 and every
> other pin held. Under it **`get_peft_model` works** and the training export succeeds. **The sibling
> has since been retired: `ort-training-local` itself now carries `transformers>=4.50,<4.58`,** after a
> third control (the BERT encoder, `all-MiniLM-L6-v2` text-classification) also came back identical —
> 73,728 trainable parameters, unchanged. There is one training profile again, and it can build Gemma-3.
> `torch`, `peft`, `numpy` and `onnx` remain exactly pinned: those are the real ABI couplings, and
> `tests/unit/test_dependency_profiles.py` now guards that distinction instead of the retired fork. The real blocker was an input-set mismatch: `Gemma3TextOnnxConfig` declares
> `[input_ids, attention_mask]` where `LlamaOnnxConfig` declares
> `[input_ids, attention_mask, position_ids]`, and Optimum passes dummy inputs **positionally**, so the
> four-parameter decoder wrapper received `labels` in the `position_ids` slot and died inside
> `torch.jit` with *"missing 1 required positional argument: 'labels'"*. Fixed by
> `OnnxDecoderNoPositionIdsTrainerWrapper` + an `ArchitectureSpec.trainer_wrapper_class` override, with
> a fail-closed signature/inputs cross-check so this class of bug names itself in future.
>
> **The control is what makes this trustworthy:** SmolLM2-135M through the same bumped group exports
> fine at parity delta 0.0111 nats — the identical figure its 2026-08-10 export recorded under 4.46.2.
> Raising the pin moved nothing for a model that already worked.
>
> **Published 2026-08-15:** `mobiletransformers/functiongemma-270m-it` (private, 102 files, 3.87 GB)
> from `google/functiongemma-270m-it` — 18 layers, 72 LoRA tensors, rank 8, 368,640 trainable of
> 268,098,176. The `sha256` of downloaded files verifies against the published manifest.
>
> **NOT proven: on-device Gemma-3 training.** This is a train-capable *package*; no training step has
> run on a phone for this architecture. That is the next device leg, and the demo in §1a above still
> ran on a SmolLM2 decoder.

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

**Three of those screens were structurally dead on a freshly pulled package** — found and fixed
2026-08-15, which is exactly the class of problem "no screen has been walked on hardware" hides:

- **Train** reads `<cacheDir>/<repo>/train/<trainFile>.jsonl`, packages deliberately ship no training
  data, and the app had no way to create one — so Start could only ever fail for a real user. Every
  instrumented test writes its own copy; a person had no equivalent. Now: a bundled 48-row tool-call
  set generated from the *same allowlist the Tool calls screen declares* (so every completion is one
  `FunctionCallValidator` accepts), plus an "Install sample dataset" action.
- **RAG** — `MobileTransformerModel.ingest` was called **nowhere in the app**, so the vector store was
  always empty and the Chat RAG toggle always returned zero sources. Now there is an ingest control and
  a bundled document.
- **The `rag` download group was never requested** (`ModelsViewModel` only ever asked for Inference +
  Training), so no embedding encoder was ever downloaded — meaning even a correct ingest call had
  nothing to embed with. Now a checkbox on the Models screen, where the ~91 MB cost is visible.

**What is left on the app** — **mostly superseded by §0, which is authoritative.** Corrected in place
2026-08-15:

- ~~**It has never been run on the device.**~~ — **The app was installed and launched on the S21 FE
  `SM-G990B` on 2026-08-15** (`adb install` clean, `MainActivity` resumed, no `FATAL EXCEPTION` in
  logcat). Walking every screen against a real package is still the highest-value manual leg; that is
  what surfaced the two §0a defects in the first place.
- ~~**No Hub token field.**~~ — **CLOSED.** `ModelHolder.hubConfig()` plumbs `BuildConfig.HF_TOKEN`
  into `fromPretrained`, and the Models screen states whether the build carries credentials (the
  boolean, never the token). Build with `HF_TOKEN=… ./gradlew :MobileTransformersApp:assembleDebug`.
- The Tool calls screen shows `Rejected` against a base model that has not been fine-tuned on the
  allowlist — that is the expected outcome, not a screen defect. **But note §0b: against FunctionGemma
  it was rejecting for an unrelated reason** (a non-JSON dialect), which no amount of fine-tuning
  would have fixed. Both the parser and the missing tool declaration are now handled.
- The engine picker renders `availableEngines` but switching engines requires a reload from the Models
  tab (the engine is fixed at load). A single-tap switch would need the facade to re-open a session.
  **The engine is at least selectable at pull time now** — `ModelsViewModel` hardcoded
  `InferenceEngine.NATIVE`, so GenAI could be *reported* as available and never actually chosen.
- ~~The old `ui/theme/` survived the rewrite untouched.~~ — the app now uses `AppTheme.FRI` (the red).
  The palette's own inconsistencies are recorded in §0d.

### 1c. Smaller code items

- **`agent-dataset` intent mapping** — `ANDROID_INTENT_BY_ACTION` covers 5 of the corpus's 7 actions.
  Both flashlight actions are `CameraManager` torch calls with no public intent, so they are
  deliberately unmapped: trainable and validatable, never bindable. Extending it is an app decision.

*(The #31 markdown link-check closed 2026-08-14: `test_docs.py` now sweeps **every tracked markdown
file** — `agent_docs/` included, which is where the worst reference rot was — and `ci.yml`'s fast job
runs it as a named step.)*

---

## 2. Device / manual test legs still open

- ~~**`ToolCallDeviceTest` has never been run**~~ → ~~**RUN 2026-08-14, fails at generation**~~ —
  **PASSES 2026-08-14 post-fix.** See §1a. #37 is closed.
- ~~**`PostMergeNumericsTest` has never appeared in a recorded suite run**~~ — **CLOSED 2026-08-14.**
  `mergedWeightsChangeTheComputationAndStayNumericallySane` **PASSES in 87.07 s** against a pristine
  `TRAIN=1` SmolLM2-135M-Instruct package on the S21 FE `SM-G990B` / Android 15 / arm64-v8a, run
  before any training suite touched the package. It has since been joined by
  `aLargeAdapterDeltaSurvivesTheMergeIntoTheInferenceGraph`, which *does* assert the delta is the
  correct one (1.295 nats memorised vs 5.529 unrelated, 4.651 pristine baseline) — that is the test
  the old one was missing, and the one that proves the fix.
- ~~**Verify whether the offline Python merger has the same defect**~~ — **CHECKED 2026-08-14: it does
  not, because it never writes `.bin` files at all.** `merge_validators.py` keeps merged results in
  memory for validation only. **Exported packages were therefore never corrupted; only device merges
  were.** The `write_raw_tensor_atomic` comment claiming offline/device byte-identity was describing a
  comparison nobody had made, and has been corrected.
- ~~**`ObservedInit.transposed` is never assigned anywhere**~~ — **CLOSED 2026-08-14.** The dead field
  is deleted (with a comment saying why it must not come back) and replaced by
  `derive_transpose_policy()`, which *observes* orientation from the adapter and weight shapes, plus
  `resolve_package_transpose_policy()`, which settles the package from its non-square layers because a
  square layer cannot decide its own. Fails closed if non-square layers disagree. 3 falsifiable tests.
  **The device side still does not trust the field** and observes orientation itself at write time —
  deliberately, because every package already on a device declares `no_transpose` wrongly, and
  honouring a stale declaration would re-break them.
- ~~**Strengthen the assertions that let the transpose hide for months**~~ — **DONE 2026-08-14.**
  `TrainMergeGenerateTest` now also rejects blank output and degenerate single-character repetition
  (the two shapes a corrupted merge actually produces); `PostMergeNumericsTest`'s arbitrary-token
  `PROBE` is now real tokenized English. See Operational knowledge → "Magnitude-based checks cannot
  detect a permutation" for why every numeric probe was blind.
- #9: on-device atomic-overwrite-under-kill remains untested.
- **#21's on-device Hub pull — now RUNNABLE for the first time, and never was before.**
  `android.permission.INTERNET` was declared **nowhere** in the tree (verified against the merged
  manifests Gradle produced, not just the sources). The whole download stack — resolver, planner,
  streaming downloader with Range-resume and sha256 verify, WorkManager worker, installer — is complete
  and JVM-tested against MockWebServer, and could not have run on any device: the first real GET throws
  `SecurityException`. MockWebServer is localhost on the JVM, where no permission model applies, so no
  existing test could see it. Fixed, and pinned by
  `test_guards.py::test_the_library_manifest_declares_the_permissions_its_own_code_needs` (verified to
  fail first). **The leg itself is still unrun**: `make device-hub-test REPO=<org>/<name>` drives the
  new `HubPullDeviceTest`.
- **Two staging trees used to survive a pull.** `HubDownloader` never deleted `.download/<repo>`, and
  the installer `copyRecursively`'d into `.staging/` — so a pull cost ~3x the package in transient disk
  and left ~1x behind permanently. For the 3.87 GB FunctionGemma package that is the difference between
  needing ~11.6 GB free and ~3.9 GB. Both fixed (`consumeSource` + `finally`-delete), JVM-tested,
  falsifiability-checked.
- #21: Kotlin `VariantSelector` parity. #22: a real authenticated adapter upload with a checkpoint
  factor read.
- ~~**The rewritten showcase app has never been run on hardware**~~ — **installed and launched
  2026-08-15** (S21 FE `SM-G990B`); see §1b. Still the highest-value manual leg: walking every screen
  against a real package exercises the whole public facade in one pass, and it is what found the two
  defects in §0a. **Start with the Models screen** — install `mobiletransformers/functiongemma-270m-it`
  from the Catalog tab, watch the byte/rate/ETA progress, then tap Load on the Installed row (the
  §0a regression).
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
