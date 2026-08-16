# Manual device checks — to walk through together

**Device on hand:** S21 FE `SM-G990B` (`RFCT60FAW6H`), Android 15, arm64-v8a.
**Written:** 2026-08-16, after the §2 code-fix cycle.

Everything here needs a human at the phone: a UI observation, a deliberately corrupted package, a
process kill, or credentials. Each check says **what to run**, **what you should see**, and **what it
means if you don't** — the last part matters most, because several of these have a "passes by
accident" mode.

> **Build the APK with `.env` sourced.** `make android-build` does *not*, and the resulting APK
> silently cannot pull private repos (FunctionGemma is private).
>
> ```bash
> cd android/MobileTransformers && set -a && . ../../.env && set +a && \
>   JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformersApp:assembleDebug
> adb install -r MobileTransformersApp/build/outputs/apk/debug/MobileTransformersApp-debug.apk
> ```

---

## 1. Chat template now loads — the headline fix

Proven on the host (`ChatTemplateResolutionTest`) and measured against the real files: SmolLM2's
template **is accepted** by Pebble, FunctionGemma's (13,792 B of `namespace()`/`dictsort`/macros) is
**rejected**. Both outcomes are correct. This confirms it on hardware.

```bash
adb logcat -c
# load SmolLM2 in the app, send one chat message
adb logcat -d | grep -i ORTTokenizerNative
```

**Expect for SmolLM2:** `Chat template active (368 chars); prompts are turn-wrapped.`
**Expect for FunctionGemma:** `Chat template failed its probe render; continuing unwrapped.`

**What it means:** the SmolLM2 line is the fix working — before this, *every* package logged
`Chat template not found`. The FunctionGemma line is the probe doing its job; the tool-call path
frames its own turns and does not need it.

**Worth judging by eye:** SmolLM2's chat replies should be noticeably more coherent than you
remember, because the model is finally seeing the `<|im_start|>` format it was trained on.

---

## 2. GenAI inference — never exercised on hardware

The one 2026-08-15 fix never run on a device, and the item you flagged as untested.

1. Load **FunctionGemma**, open the engine picker.
   **Expect:** NATIVE only, plus a note naming `genai_config.json`.
2. Load **SmolLM2** (`build/pkg`), open the engine picker.
   **Expect:** NATIVE and GENAI both offered.
3. With SmolLM2 on GENAI, send a message.

```bash
adb logcat -d | grep -iE "supportedEngines|EngineUnavailable|genai"
```

**If GENAI is offered for FunctionGemma:** `ModelRuntimeFactory.enginesAvailableFor` is not reading
the package's `supportedEngines`; expect `supportedEngines=[native]` in the load line.

**Also worth checking now:** with the template fix, Native and GenAI should produce the **same first
token** for one prompt — both render through `ORTConversationState`, and keeping them in step is why
the `applyChatTemplate` flag is honoured in both generators. A divergence here is prompt
construction, not weights.

---

## 3. Vocab clamp — needs a deliberately broken package

**Why you have never seen the warning:** `build/pkg-functiongemma` already carries the *repaired*
`vocab_size: 262144`, so the clamp is a no-op on it. Only the copy on the Hub is still wrong. To see
the clamp fire you have to break a copy on purpose.

```bash
cp -r build/pkg-functiongemma /tmp/pkg-fg-broken
cd /tmp/pkg-fg-broken
f=shared/tokenizer/mobiletransformers_tokenizer_config.json
sed -i 's/"vocab_size": 262144/"vocab_size": 262146/' "$f"
sha256sum "$f"        # patch this into variants/cpu-int4/checksums.json, or install fails first
```

Push it, load it, generate.

**Expect in logcat:** `declared vocab_size 262146 exceeds the graph's logits width 262144`
(the format string is at `cpp/native-lib.cpp:656-658`).

**What it means:** the clamp protects every package already published and every one already on a
device. If generation *crashes* instead of logging, the clamp is not being applied — that is the bug
it exists to prevent (`Gather` fails on an id with no embedding row).

**Clean up:** `rm -rf /tmp/pkg-fg-broken`, and re-push the good package.

---

## 4. `classify()` — never run on a device, now has a screen

New this cycle: **Classify** appears in the drawer for a classifier that names its labels. The
package is already on disk, so this is ~15 s of file movement, not a re-export.

```bash
# build/pkg-encoder already exists: 204 MB, selectedTask "text-classification", inference/ + train/
# steps 2-4 of scripts/device_package.sh are pure file movement against it
```

1. Push `build/pkg-encoder`, load it.
2. **Expect the drawer to change shape:** Classify appears; **Chat and Tool calls disappear**. That
   inverse is deliberate — an encoder has no generative head.
3. Classify some text; expect per-label probability bars summing to ~100%.

**If Classify does not appear:** the package's `id2label` is empty. `supportsClassification` requires
labels, because a head that answers `LABEL_3` is a number in a costume. Check
`inference/optimum_config.json` carries `id2label` — the export path for it is now tested
(`tests/export/test_id2label_export.py`) but has still never been *observed* end to end.

**The riskiest part of this check** is the forward pass, not the maths (`ClassifierScoringTest` covers
the scoring). `ClassifierSession` bypasses `ORTRetriever.createEmbeddingModel` and opens
`createEmbeddingSession` against `inference/` with `embeddingDim = numLabels`, on the premise that
`inference::generateEmbedding` returns the raw first output tensor with no pooling. If the numbers
look like nonsense rather than a distribution, that premise is what to question.

---

## 5. Training notification now advances

Fixed this cycle: `foregroundInfo` always accepted a `progress` and the only caller passed `null`.

1. Start a training run with a step target set.
2. Watch the ongoing notification.

**Expect:** a moving progress bar and text like `Step 42/200 · loss 1.8471`, updating on whole-percent
changes.
**Expect on a resumed chunk:** the bar picks up where the last chunk stopped — it is a whole-run
fraction, not a per-chunk one. Restarting at 0% each chunk means `currentStep`/`totalSteps` are being
read as chunk-local.

---

## 6. Embedding lifetime fix — exercised by RAG

`generateEmbedding` returned a pointer into an `Ort::Value` destroyed at scope exit, and the JNI error
path `delete[]`'d it. Fixed by moving the tensor onto `EmbeddingSessionCache::last_output`, mirroring
the generation path. **There is no host test** — the C++ suite is deliberately ORT-free and cannot
link a session — so the device is the only check.

1. Ingest a document (RAG), then run a grounded query.
2. Ingest something large enough to be several hundred chunks.

**Expect:** no crash, and sane retrieval. Also fixed here: three `GetLongArrayElements` calls that were
never released, leaking on every embedding call — i.e. once per chunk. A long ingest is the workload
that would have shown it.

**If it crashes:** `adb logcat | grep -E "signal 11|SIGSEGV"`. Under memory pressure is where the old
code was most likely to fail.

---

## 7. Legs still open — unchanged by this cycle

| leg | command | note |
| --- | --- | --- |
| Full instrumented suite | `make device-package MODEL=HuggingFaceTB/SmolLM2-135M-Instruct TRAIN=1 RAG=1` then `make device-test` | ~54 min (23 tests / 3219 s on 2026-08-14). Predates every 08-15 **and** 08-16 change. |
| Encoder suite | `make device-package MODEL=sentence-transformers/all-MiniLM-L6-v2 TASK=text-classification TRAIN=1 RAG=0` then `make device-test` | 19 tests / 16 skipped previously. |
| Hub pull (#21) | `make device-hub-test REPO=<org>/<name>` | **Never run.** `android.permission.INTERNET` is declared (library manifest line 36), which is what made it runnable at all. |
| Scheduled start delay | set a short `initialDelayMinutes`, force Doze | `initialDelayMinutes` is a floor, not an appointment — WorkManager batches deferrable work. Confirm it does not start *early*; a late start is accepted behaviour. |
| #9 atomic overwrite under kill | `adb shell am force-stop` mid-merge | Needs a small harness. Assert the target `.bin` is old-or-new but never torn, and its `.sha256` sidecar matches. |
| #21 `VariantSelector` parity | — | Same constraints must select the same variant as the Python `hub/variant_select.py`. |
| #22 adapter upload | — | Needs Hub credentials and a trained checkpoint; read the checkpoint factor back. |

---

## 8. Background downloads — now wired, and the highest-risk change to test

Downloads now run through `PackageDownloadWorker`. `ModelsViewModel` calls
`ModelHolder.loadInBackground`, which enqueues the worker, observes `DownloadJob` until terminal, then
loads through the same `fromPretrained` path as before. **This is the change most worth testing on
hardware**, because WorkManager scheduling cannot be meaningfully exercised on the JVM.

**8a — a pull survives leaving the app.** This is the whole point.
1. Models → enter a repo id not yet installed → Load.
2. Once bytes are moving, press **Home**. Wait a minute. Return.

**Expect:** progress advanced while you were away, and the model loads when it completes.
**Before this change** the pull died with the Activity.

**8b — the Wi-Fi-only switch.** New control on the Models screen, default **on**.
1. Turn Wi-Fi off, leave mobile data on, keep the switch on, start a pull.
   **Expect:** the model bar reads **"waiting for Wi-Fi"** and nothing downloads. That is correct, not
   a hang — the worker sits in `ENQUEUED` until its constraint is met. Turn Wi-Fi on; it should start.
2. Repeat with the switch **off** on mobile data. **Expect:** it downloads.

```bash
adb shell dumpsys jobscheduler | grep -A5 mobiletransformers-download
```

**Why this switch exists:** `requireUnmetered` defaults to true, so without a visible control a pull
on mobile data waits forever and is indistinguishable from a broken app.

**8c — Cancel actually cancels.** Start a pull, press Cancel.
**Expect:** it stops, and does **not** resume in the background. Cancel now stops the *worker*, not
just the observing coroutine — cancelling only the latter would detach the UI from a download that
kept running, which is the failure mode background work introduces. `.partial` files are kept, so
pulling again resumes.

**8d — variant selection.** A silent bug fixed on the way: the worker never passed `abis` or
`totalMemMb`, and `HubDownloader` falls back to `manifest.defaultVariant` when `abis` is empty. So a
background pull would have taken whatever the publisher listed first regardless of what the phone can
run — the exact behaviour #21 removed from `fromPretrained`, still present here because the worker had
no caller. Both paths now read the same `DeviceCapabilities`.

**Expect:** the variant installed by a background pull matches the one a foreground pull picks —
`arm64-v8a` on this device. A mismatch means the shared path is not being taken.
