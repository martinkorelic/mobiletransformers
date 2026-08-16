# Handoff — what is still open

**Branch:** `restructure` · **Rewritten:** 2026-08-16 (end of the 0.2.0 showcase-prep cycle)

This file is **only what is left to do**. Completed work is not recorded here — it lives in git, in the
docs, and in the per-plan self-checks. Rewritten each cycle.

> ## Read these first
>
> 1. **`IMPLEMENTATION_ORDER.md` → "Operational knowledge (permanent)"** — environment and profiles,
>    the ORT engine separation, the device workflow, the recurring failure shapes. Everything a cold
>    agent needs *regardless of cycle* lives there, not here.
> 2. **§5 "Traps"** at the bottom of this file. Each cost a cycle.
> 3. **Do NOT read `agent_docs/audits/` as a to-do list.** They are 2026-08-07 snapshots with no
>    closure annotations. This file is the authoritative list of what is open.

---

## Where the project stands

The 0.2.0 showcase-prep cycle landed the native-size fix, the model-card work, the PEFT surfacing, a
Retrieval screen, tool-call permissions, and a real defect in every package ever exported. It did
**not** land the identifier purge, the docs rebuild, the version bump, or the Hub publish.

**Host gates (measured 2026-08-16 — do not copy forward, re-measure):**

```
Kotlin JVM 406 = 379 library + 27 app   ·  C++ 46/46  ·  Python 599 passed / 11 skipped
```

```bash
uv sync --frozen --group dev --python 3.10 && make check    # ALWAYS reset the profile first
make test-cpp
cd android/MobileTransformers && JAVA_HOME=/opt/android-studio/jbr ./gradlew --rerun-tasks \
  :MobileTransformers:testDebugUnitTest :MobileTransformersApp:testDebugUnitTest
```

Read the JVM counts out of `build/test-results/**/*.xml`, never off the console: Gradle reports
`UP-TO-DATE` as success, so a suite that never executed looks identical to one that passed.

> **All three gates are green as of the rewrite.** One test needed fixing to get there:
> `test_handoff_map.py::test_the_derivation_agrees_with_a_real_exported_package` took `maps[0]` — the
> alphabetically first export on disk — and orientation is only observable from a **non-square**
> adapted weight. The new `all-MiniLM-L6-v2` package adapts `query`/`value` at 384x384, so it sorted
> first and made the suite fail on a package that simply cannot answer the question. It now scans for
> a package that can, and skips honestly when none does.

---

## 1. The model shelf — the blocking item for the showcase

`scripts/publish_catalog.sh` (+ `make publish-catalog`) exports, gate-checks and publishes the catalog.
`ONLY=<key>`, `PUSH=0`, `KEEP=1` are supported. Every entry must ship **both** an inference and a
training stage — a shelf entry that cannot be fine-tuned demonstrates half the framework.

| key | repo | exported? | total | inference group | features |
| --- | --- | --- | --- | --- | --- |
| `smollm2` | `mobiletransformers/SmolLM2-135M-Instruct` | ✅ | 935 MB | 663 MB | core, inference, train, rag |
| `qwen25` | `mobiletransformers/Qwen2.5-0.5B-Instruct` | ✅ | 3212 MB | 2554 MB | core, inference, train, rag |
| `minilm` | `mobiletransformers/all-MiniLM-L6-v2` | ✅ | 214 MB | 94 MB | core, inference, train, rag |
| `distilbert` | `mobiletransformers/distilbert-sst2-english` | ❌ **fails** | — | — | inference only |

### 1.1 DistilBERT's training export fails — a real, precisely-diagnosed bug

```
export failed: OnnxSequenceClassificationTrainerWrapper's forward signature does not match
OnnxConfigWithLoss's inputs, and Optimum passes them positionally.
  wrapper forward : ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
  config inputs   : ['input_ids', 'attention_mask', 'labels']
```

**DistilBERT has no `token_type_ids`.** BERT and RoBERTa do, so `OnnxSequenceClassificationTrainerWrapper`
(`export/training_export.py:227`) declares it and works for them; `DistilBertOnnxConfig` declares two
inputs, Optimum binds positionally, and `labels` would land on `token_type_ids`.

This is **exactly the Gemma-3 `position_ids` bug in encoder form**, and the fail-closed
signature/inputs cross-check added for that one caught it before `torch.jit` — which is the machinery
working as designed.

**The fix is the same shape:** add a `token_type_ids`-free sequence-classification wrapper beside the
others in `export/training_export.py`, and point the registry row at it —
`config/registry/architecture.py:330` `DistilBertForSequenceClassification` gains
`trainer_wrapper_class="mobiletransformers.export.training_export.<NewWrapper>"`. The precedent is
`Gemma3ForCausalLM` → `OnnxDecoderNoPositionIdsTrainerWrapper`.

Then: `ONLY=distilbert PUSH=0 scripts/publish_catalog.sh`.

### 1.2 Nothing has been pushed — the token cannot see the repos

`HF_TOKEN` in `.env` is **fine-grained and scoped to `functiongemma-270m-it` only**. `model_info` on
the other four returns `RepositoryNotFoundError`, which the Hub returns identically for "does not
exist" and "no access", so this cannot be resolved from here.

**Owner action:** widen the token to the `mobiletransformers` org (write), or add the four repos to
its scope. Then `scripts/publish_catalog.sh` (PUSH defaults to 1). Repos stay private; flipping them
public is a deliberate second step.

### 1.3 Then, and only then, update the app catalog

`MobileTransformersApp/src/main/assets/model_catalog.json` still lists `Qwen2-0.5B` (wrong id — it is
`Qwen2.5-0.5B-Instruct` now), has no DistilBERT entry, carries estimated `approxSizeMb`, and has every
entry at `published: false`. Use the **measured** inference-group figures in the table above, add
`peft` per entry (the field exists and renders as a chip), and flip `published: true` **only after a
device has loaded that package** — the 2026-08-15 cycle is the argument.

Note for the description text: the MiniLM package is exported as `text-classification` (that is what
makes an encoder trainable at all — `FEATURE_EXTRACTION` is `trainable=False`), so its head is
randomly initialised and its labels are `LABEL_0/1`. `supportsClassification` is therefore false and
Classify stays hidden for it. That is correct and self-consistent; say so, or it reads as a bug.

---

## 2. Code — what is left

### 2a. The plan-identifier purge — attempted, reverted, NOT done

~750 sites (351 Kotlin, 323 Python, 54 C++, 28 in `docs/`) still carry `#NN`, `agent_docs/…`,
`IMPLEMENTATION_ORDER`, `Migration Map S1`, `Gate 0.1`, `Tier 2`.

**A scripted pass was tried on 2026-08-16 and had to be fully reverted.** Two failures worth knowing
before anyone tries again:

1. The purge stripped identifiers *before* the possessive/determiner rules could match, producing
   `Distinct from's engine-level`, `(, 01_code_plans/01)`, `/**: TrainingScheduler`.
2. The repair written for that was worse: two tidy-up rules — `\s*\(\)` → `""` and `" {2,}"` → `" "` —
   ran on **every line of every file**, not just comments. That deleted every empty call parenthesis
   in the tree (`emptyMap()` → `emptyMap`, `fun search()` → `fun search`) and collapsed all
   indentation, which is syntax in Python.

Recovery worked only because both passes were pure line-wise functions: replaying them against `HEAD`
identified which lines came from there (256 files restored exactly, the rest by `difflib` alignment on
stripped content), the missing `()` came back by looping on the Kotlin compiler's own
`Function invocation 'x()' expected` diagnostics, and the ~500 collapsed-indent lines that were *new*
relative to HEAD were re-indented from brace depth. Verified back to baseline.

**The lesson: a whole-tree regex rewrite of prose cannot be validated by the test suite,** because
comments do not affect behaviour. Every gate stayed green on mangled prose; only the *accidental*
damage to code was caught. If retried: scope the regex to comment tokens only, never whole lines; go
file-by-file reading each diff; and `git stash` first, which would have made recovery a one-liner.

Realistically this is a per-file editing job, not a scripted one.

Once done, **add a guard** in `tests/unit/test_guards.py` (no `agent_docs` / `IMPLEMENTATION_ORDER` /
`#<1-2 digits>` outside string literals under `src/`, `android/*/src/main`, `scripts/`) and **verify it
fails before the cleanup**.

### 2b. Untrack `agent_docs/` (owner asked for this)

`git rm -r --cached agent_docs` + an `agent_docs/` line in `.gitignore`. The directory stays on disk.
Consequences: `tests/unit/test_docs.py` enumerates via `git ls-files` so its link check drops it
automatically (confirm rather than assume); `tests/unit/test_release_plumbing.py`'s `forbidden` tuple
names `"agent_docs/"` and should lose that element.

### 2c. Finish the de-jargon pass on the app

Done: the standalone Tool calls screen is gone (it was a dry-run duplicate of Chat rendering
`willExecute = false` and `IntentBinder` internals); its allowlist now lives in
`Configuration ▸ Actions`. Chat's "Setup" panel is gone and its "Settings" link opens
`Configuration ▸ Generation`.

Left:
- **Models screen: three tabs → two.** "Pull by id" should become an "Advanced: pull any package"
  expander at the bottom of Catalog. Its 4-sentence manifest/download-group paragraph becomes one
  sentence plus a "Why?" expander.
- **Train / Federated / Configuration / About**: lead with the control, move the justification into a
  `Details` expander or delete it. Named offenders: the engine paragraph ("Naming GenAI and being
  given Native would be a wrong answer…"), the RAG download-group paragraph, the federated round
  description.
- Delete `#NN` from user-visible strings and from each screen's KDoc (overlaps §2a).

### 2d. Version 0.2.0

Three sites, pinned against each other by `tests/unit/test_version_sites.py`:
`pyproject.toml` `version`, `android/MobileTransformers/gradle.properties` `version`, and
`MobileTransformersApp/build.gradle.kts` `versionName` (currently `"1.0"`, which matches nothing).
Plus a `## [0.2.0]` CHANGELOG section and a `docs/RELEASE_CHECKLIST.md` step for `make publish-catalog`.

### 2e. Docs rebuild

- **`README.md`** still describes the six-tab app the drawer replaced. Rewrite around the one-paragraph
  claim (export → pull → chat → retrieve → classify → fine-tune → merge → tool-call, on a phone) plus
  the measured evidence and the catalog table. Drop the `agent_docs/` note.
- **`docs/SHOWCASE.md`** (new) — a tour of the sample app, one section per capability, naming the
  package each needs and what you should see. This is where the prose stripped out of the UI goes.
- **`docs/CATALOG.md`** (new) — published packages: repo, base model, size, PEFT method, features, and
  measured tokens/second from a real device run.
- **Sweep `docs/ANDROID_SDK.md` + `docs/PUBLIC_API.md`** for everything they do not mention yet:
  `RuntimeCapabilities.toolCalling` / `.trainingParameterCount` / `.peftMethods` / `.graphPrecision` /
  `.isEncoderOnly`, `PackageTask.inferenceGraphPrecision`, `ActionSpec.requiredPermissions`,
  `IntendedAction.requiredPermissions`, `ToolCallResult.NoCall`, `GenerationResult.promptTokenCount` /
  `.contextLimit`, `TrainingScheduleConfig.initialDelayMinutes`, `classify()`, `retrieve()`.
- **`docs/RAG.md`** — cover retrieval as its own operation, not only as grounding.
- **`docs/EXPORT.md`** — document `make publish-catalog` and the per-task flag rules (why `--genai` is
  decoder-only; why encoders need an explicit `--task text-classification` to be trainable at all).
- **`docs/ARCHITECTURE.md`** — add the "Native dependencies" section it has never had (see §3).

### 2f. Portability: the fetch path (§3 is the analysis; this is the work)

- `third_party/android/manifest.json` — one entry per artifact with filename, destination, size,
  **sha256** and build provenance. This also fills `third_party/onnxruntime/manifest.json`'s
  `android.aar_sha256` / `android.so_sha256`, which are `null` today.
- `scripts/fetch_native_deps.sh` — download one archive, verify every sha256, unpack into `jniLibs/`,
  `aarLibs/`, `cpp/includes/`. Refuse rather than half-populate on mismatch. `URL` overridable.
- `make doctor` — one preflight report naming every missing prerequisite (uv, Python 3.10/3.12,
  JAVA_HOME, ANDROID_HOME, jniLibs, aarLibs, includes, training wheel, `.env` HF_TOKEN, adb/device)
  with the command that fixes each. `scripts/android_build_aar.sh`'s error message should point here
  instead of at `agent_docs/`.
- `.env.example`, committed, with `HF_TOKEN=` and a note on which operations need it.
- `JAVA_HOME` — `/opt/android-studio/jbr` is hardcoded as a fallback in the Makefile and four scripts.
  Probe `java` on PATH first; make the failure name what to set.
- A guard asserting no tracked file carries an absolute `/home/`, `/Users/` or `/opt/android-studio`
  path outside a documented allow-list.

### 2g. Still open from before, unchanged

- **`MemoryHeadroom` calibration** — it exists, is tested, is logged, and is deliberately not shown to
  a user: it warned about a model that fits. Calibrate `TRAINING_OVERHEAD` against RSS traces from
  several real `low_mem` runs before putting it in front of anyone. Read
  `MemoryHeadroomTest.aDiskSizedRuleWouldHaveRefusedAWorkingSession` before "improving" it.
- **Two levers if training memory is ever tight** — `maxSequenceLength` (512, against a tool-call
  corpus of far shorter rows) and `batchSize` (4). Not needed as of 2026-08-15.
- **On-device evaluation** (before/after perplexity) and a **live resource panel** — `runtime/MemoryProbe`,
  `scheduler/ThermalGuard` and the CSV trace `TrainingWorker` writes are all present and surfaced
  nowhere. Host-side machinery in `research/evaluation/` and `docs/mobile_evaluation.md`.
- **`agent-dataset` intent mapping** covers 5 of 7 corpus actions. Both flashlight actions are
  `CameraManager` torch calls with no public intent, deliberately unmapped.
- **Re-export and re-publish FunctionGemma.** The package on the Hub still carries the wrong
  `mobiletransformers_tokenizer_config.json` (`vocab_size: 262146`, 12/12/12 heads/layers,
  `type: unknown`, `bos: null`). The device-side clamp covers it; the file is wrong at the source.
  `build/pkg-functiongemma` is repaired locally. **It also predates the orphaned-external-data fix, so
  a re-export drops it from 3875 MB to roughly 2100 MB** — which is the single biggest improvement
  available to the flagship package.

---

## 3. Portability — a fresh clone cannot build the Android SDK

No absolute personal paths, device serials or credential literals exist in tracked code (`git grep`
for `/home/`, `/Users/`, the serial and `~/.claude` returns only `JAVA_HOME` fallbacks). The problem is
the opposite: what is *not* in the repo is undiscoverable.

| artifact | size on disk | tracked? | documented? |
| --- | --- | --- | --- |
| `MobileTransformers/src/main/jniLibs/arm64-v8a/*` (8 files) | **116 MB** (was 1.2 GB) | no | **no** |
| `MobileTransformers/src/main/aarLibs/onnxruntime-genai.aar` | 40 MB | no | **no** |
| `MobileTransformers/src/main/cpp/includes/{google,protobuf}` | 24 MB | no | **no** |
| `third_party/wheels/onnxruntime_training-…-cp312-linux_x86_64.whl` | 662 MB | no | instructions **unusable** |
| `.env` (`HF_TOKEN`), `ANDROID_HOME` / `local.properties` | — | no | partially |

`scripts/android_build_aar.sh:26` prints *"See docs/ARCHITECTURE.md and agent_docs/ for the
provisioning story"* — `ARCHITECTURE.md` never mentions `jniLibs`, and `agent_docs/` is being
untracked. `third_party/wheels/README.md` says to `cp ../on_device_llm_finetune/dist/…`, a sibling
repo that exists on one machine.

**Bundles are built and waiting to be uploaded** (`build/dist/`, gitignored):

| bundle | size | sha256 |
| --- | --- | --- |
| `mobiletransformers-natives-0.2.0-arm64-v8a.tar.zst` | 63 MB | `e4b00892d719fabe6dfcc7b52d5886e404997ed89e45337caf28bf0e47450fa1` |
| `…-arm64-v8a-debug-symbols.tar.zst` | 260 MB | `2383869b5015b0c3d3db83b7572542cf175598889f6d83f5d46eca84d0cd23d6` |

> **The debug-symbols archive is the ONLY copy of the unstripped originals.** They cannot be
> regenerated without a full ORT source build. Do not delete `build/dist/` before it is uploaded.

Owner action: host both (a GitHub Release keeps the URL stable), then §2f wires the fetch script to it.

---

## 4. Device / manual legs still open

Nothing in this cycle was run on hardware. `agent_docs/MANUAL_DEVICE_CHECKS.md` holds the walk-through
written on 2026-08-16; it needs new sections for this cycle's work.

- **The whole 0.2.0 app is unexercised on a device.** New this cycle: Retrieval screen, Classify
  screen, the removed Tool calls screen, Chat's Settings link, `Configuration ▸ Actions`, the PEFT
  badge, and tool calls actually firing.
- **`SET_ALARM` now fires.** `com.android.alarm.permission.SET_ALARM` was declared nowhere, which is
  why an accepted tool call died with `SecurityException` at `startActivity`. It is an install-time
  permission, so declaring it *is* granting it — confirm an alarm really appears in the clock app.
- **`classify()` has never run on a device.** Blocked on §1.1 — there is no working classifier package.
- **Retrieval on an encoder-only package** (`all-MiniLM-L6-v2` alone): the drawer should show Models /
  Retrieval / Train / Federated and hide Chat, Tool calls and Classify.
- **Re-run the recorded instrumented suite.** Both rows below predate every 2026-08-15 change.

| package on device | result |
| --- | --- |
| SmolLM2-135M `TRAIN=1 RAG=1` (decoder) | 23 tests, 0 failures, 3 skipped, 3219 s (2026-08-14) — **stale** |
| all-MiniLM-L6-v2 `TASK=text-classification TRAIN=1` (encoder) | 19 tests, 0 failures, 16 skipped — predates the merge-transpose fix |

- **`make device-hub-test REPO=<org>/<name>`** has still never run. It only became runnable when
  `android.permission.INTERNET` was added.
- **Scheduled-training start delay** under Doze; **atomic-overwrite-under-kill**; **multi-hour / FGS
  quota**; **Kotlin `VariantSelector` parity**; **a real authenticated adapter upload**.
- **Showcase GIFs** — `docs/ortransformer-feature.gif` predates the rewrite; `base-model.gif` and
  `on-device-trained.gif` are the before/after pair. Worth re-recording now the app is presentable.

---

## 5. Traps that have each cost a cycle — check these before debugging

- **A green host suite proves nothing about the device.** Four defects once sat between a user and
  every feature of the app over a fully green suite.
- **A green suite proves nothing about *comments*.** New this cycle, and the reason §2a failed: a
  regex pass mangled prose across 300 files and every gate stayed green.
- **Gradle prints `BUILD SUCCESSFUL` when ninja relinked nothing.** `--rerun-tasks` re-runs the Gradle
  task, not the CMake link. After changing anything in `jniLibs/`, delete
  `build/intermediates/cxx/Debug/*/obj/arm64-v8a/libmobiletransformers.so` and check the output's
  timestamp before believing a result.
- **AGP strips native libraries at packaging**, and to the same bytes as `llvm-strip --strip-all`
  (59,872,736 vs 59,872,456 for `libonnxruntime.so`). Unstripped prebuilts cost repo and clone size,
  never APK size. AGP strips *harder* than a plain `--strip-debug`, which yields 145 MB.
- **`make android-build` does not source `.env`.** The APK builds fine and silently cannot pull private
  repos. Build with `set -a && . ../../.env && set +a` first.
- **Gson bypasses Kotlin constructors**, so a field absent from JSON is `null` even where Kotlin
  declares it non-null with a default — and the NPE surfaces far from any mention of JSON.
  `FunctionCallValidator.fromSchema` normalises all four collection fields for this reason.
- **Material 3 fills omitted colour roles from its baseline palette, which is purple.**
- **`strokeWidth = 1f` in a Compose `Canvas` is one PIXEL, not one dp.**
- **A `launch` that throws does not deliver to whoever `join()`s it.**
- **Release native sessions on a resource check (`!= null`), never a state check.**
- **Exported `training_config.json` carries no `deviceOptions` section**, so `parseTrainingArguments`'
  fallback is the real setting for every training run.
- **An out-of-memory death is a SIGKILL, not an exception.** Look for `lmkd: Reclaim` and
  `exited due to signal 9`; there will be no stack trace.

---

## 6. Not engineering

- **The licence — the only real v1.0 blocker.** CC-BY-NC-4.0 contradicts the consumable-AAR goal. It is
  a rights-holders decision (both authors in `CITATION.cff`); **the second author has NOT agreed**. Do
  not touch licence files, do not add SPDX headers, do not set the `pyproject.toml` license expression.
  `tests/unit/test_release_plumbing.py` asserts the POM matches `LICENSE.md`, so a one-sided change
  fails the suite. Flag it, nothing more.
- **CI provisioning** — either vendor the native deps to a runner, or formally define "CI green" in
  `docs/RELEASE_CHECKLIST.md` as a recorded manual run. Today it is neither, which is the actual debt.
  All three workflows are `workflow_dispatch`-only; `device.yml` targets a `[self-hosted,
  android-device]` runner that does not exist.
- **The `v1.0.0` tag** (`git tag` is empty).

---

## 7. Standing debts — accepted positions, do not "fix" without deciding

- **Variant naming** — `cpu-int4` legitimately ships an fp32 inference graph, declared via the measured
  `inferenceGraphPrecision` (now readable on device as `RuntimeCapabilities.graphPrecision`). Renaming
  is a wire-contract change, deliberately not done. Making `--quant` actually quantize the optimum path
  is a separate project with a real payoff on download size.
- **`inference/builder.py`** carries ~10 upstream-derived TODOs and its own `load_config_from_file`
  copy. Vendored GenAI builder; treat as upstream. Allow-listed in the architecture-literal guard, and
  excluded from any identifier purge.
- **`peft/mapping.py` has two decoder-shaped prefix sites left, deliberately untouched.** One converts
  `base_layer_name` into checkpoint space for the **MARS** mapping only; the other is dead, and
  load-bearing by being dead. Both want a deliberate look with the MARS tests in hand, not a sweep.
- **The stage-path guard carries 2 allow-listed sites**, both legitimate producers.
- **`export-rocm` is a declared-but-empty group** — ROCm wheels need a dedicated AMD index.
- **`jniLibs/x86_64/` was deleted this cycle.** It was incomplete (no `libonnxruntime.so`, no tokenizer
  archives) so `libmobiletransformers.so` could never be built for it, and the genai AAR supplies the
  x86_64 libraries anyway — removing it changed the APK by 244 bytes. Restoring x86_64 means building
  ORT-training and tokenizers-cpp for it first.
- **`libobjectbox-jni.so` was deleted from `jniLibs/`** — the ObjectBox AAR ships it. The duplicate only
  ever worked because the two files were byte-identical and AGP silently deduped them.
- **macOS/Windows cannot run the training side** without rebuilding the `cp312-linux_x86_64` wheel.
- **No dark theme by default.** Both schemes exist and work; the default stays light.
- **Scheduled runs cannot promise a wall-clock start.** `initialDelayMinutes` is a floor, not an
  appointment. An exact start needs `SCHEDULE_EXACT_ALARM`, which Play restricts to alarm clocks and
  calendar reminders. Accepted, and stated in the UI rather than worked around.
- **No tool-call action needs a runtime permission, and that is not an oversight.** Intent-based
  actions delegate the sensitive work to the target app, which enforces its own permissions behind its
  own UI. The benign actions worth showcasing are all install-time. The generic runtime-request path
  exists (`PermissionGate` + `ChatViewModel.onPermissionResult`) for when one is added; the app's real
  runtime prompt is `POST_NOTIFICATIONS` when a training run starts.
- **Reclaimable disk (untracked, gitignored):** `build/` ~17 G — including `build/pkg-gemma3` (3.7 G,
  superseded by functiongemma) and the `*-ab` A/B duplicates (1.5 G) — and `.venv-genai-spike/` 5.2 G,
  recreated by `spikes/genai_external_swap/build_tiny_genai_model.sh`. **Do not delete `build/dist/`**
  (see §3) or `build/pkg-functiongemma` (a re-push needs it).

---

## Cycle protocol

1. **Reset the profile before `make check`** — `uv sync --frozen --group dev --python 3.10`. More
   "broken repo" reports have come from a leftover export/training venv than from any real defect.
   `scripts/device_package.sh` and `scripts/publish_catalog.sh` both leave the tree on the training
   profile.
2. **Checkpoint before any sweeping change.** `git stash` or a branch. §2a would have been a one-line
   recovery instead of a two-hour reconstruction.
3. **Assert across the seam, and check the assertion can FAIL.** This keeps paying — it is what caught
   §1.1 before `torch.jit` did, and what the new model-card and orphan-data tests were verified against.
4. **When you empty a ratchet, re-point it or delete it** — never leave it scanning the void.
5. **When you correct an earlier claim, correct it where the claim lives**, not only in a new entry.
6. **Walk the app on hardware before claiming a cycle is done.** Every defect in the 08-15 cycle came
   from that walk and none from the suite.
