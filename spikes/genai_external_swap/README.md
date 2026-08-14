# GenAI External-Data-Swap Spike (#10 · Gate 0.1)

Validates finding **F2**: ONNX Runtime GenAI can consume on-device-merged weights **with no graph rewrite
and no fork**, just by reading the same external-data folder — proving GenAI can be a *selectable engine*
over the unified package (or, on failure, that we keep the manual loop for v1).

The decisive mechanism is the stable **`OgaCreateModel(<dir>)`** + external-data file resolution. We
deliberately avoid `OgaCreateModelWithInitializers` (fork-only) and the `model_input`/`SetModelInput`
rewrite path.

## Artifacts

| File | What |
| --- | --- |
| `check_symbols.sh` | Gate 0.1 #6/#8 — asserts `OgaCreateModelWithInitializers` ABSENT + `OgaCreateModel` PRESENT in the linked Android `.so`. |
| `desktop_spike.py` | Gate 0.1 #2/#3 — base-vs-swapped logits differ on a fresh `og.Model(dir)`. |
| `measure_rss.py` | Gate 0.1 #7 — RSS sampler (mmap-vs-copy). |
| `build_tiny_genai_model.sh` | Builds a tiny real GenAI model (SmolLM2-135M int4) in a standalone venv for the smokes. |
| `android/.../cpp/genai_spike.cpp` | JNI: `OgaCreateModel` → one token → logits fingerprint + RSS. |
| `android/.../GenAISpike.kt` + `androidTest/.../GenAISpikeTest.kt` | Device leg — external-data resolution + swap smoke. |

## Results so far (host, this machine — Linux + arm64 device SM-G990B)

- **Symbol check: PASS.** `OgaCreateModelWithInitializers` absent (fork-only confirmed); `OgaCreateModel`
  present; 23 `OgaGenerator*` symbols. Run: `./check_symbols.sh`.
- **Build + link: PASS.** `genai_spike.cpp` compiles against the real 0.14 AAR headers and
  `libmobiletransformers.so` links against the real `libonnxruntime-genai.so` on arm64; the instrumented
  `androidTest` APK assembles. The JNI symbol `Java_..._GenAISpike_runOneToken` is exported.
  - Setup done: the real `onnxruntime-genai-android-0.14.0.aar` is installed at
    `aarLibs/onnxruntime-genai.aar` (was a 1.3 MB stub) and the real `.so` copied into `jniLibs/{arm64-v8a,
    x86_64}` (was a stale 3 MB build missing the 0.14 generator symbols). Vendored fork header
    `cpp/onnxruntime-genai/ort_genai_c.h` replaced with the AAR's clean upstream header. A `packaging {
    jniLibs { pickFirsts } }` dedupe resolves the AAR-vs-jniLibs `.so` collision. The dead
    `ORTGenAITokenizer.kt` (old genai Java API) was reduced to a compiling stub (DECOMPOSE(#11)),
    and **deleted outright 2026-08-14** together with its unused `LLMRepository` field.

## How to run the swap smokes

### 1. Build a tiny GenAI model (host, one-time)
```bash
./spikes/genai_external_swap/build_tiny_genai_model.sh
# -> build/genai_spike_model/ with model.onnx + model.onnx.data + genai_config.json + tokenizer
```

### 2. Desktop swap smoke (Gate 0.1 #2/#3)
```bash
source .venv-genai-spike/bin/activate   # the venv the build script created
python spikes/genai_external_swap/desktop_spike.py --dir build/genai_spike_model
# PASS = base vs swapped logits differ on a fresh og.Model()
```

### 3. Device swap smoke (Gate 0.1 #2/#3/#5) — the key Android unknown
Push the model into the **test app's** external files dir, then run the instrumented test on the device:
```bash
# the test app package is com.martinkorelic.mobiletransformers.test
DEST=/sdcard/Android/data/com.martinkorelic.mobiletransformers.test/files/mt_genai_spike/inference
adb shell mkdir -p "$DEST"
adb push build/genai_spike_model/. "$DEST"

cd android/MobileTransformers
JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformers:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.martinkorelic.mobiletransformers.GenAISpikeTest
```
The test **skips** (with the expected path) if no model is pushed, so it never hard-fails. When the model is
present it asserts: `OgaCreateModel` resolves the relative external data (a token generates) **and** the
logits fingerprint changes after one external weight is overwritten (swap observed).

Watch `adb logcat -s GenAISpike` for the per-run `token / fp / rss pre|loaded|tok` line.

## Gate 0.1 checklist status (verified on this machine + device SM-G990B, arm64)

| # | Criterion | Status |
| --- | --- | --- |
| 6 | `OgaCreateModelWithInitializers` fork-only & not required | ✅ **PASS** (symbol check, device `.so`) |
| 2 | External weight overwrite changes GenAI output on fresh `OgaCreateModel` | ✅ **PASS** — desktop `|ΔL|=39.6`; **device** token 28→6156, fp 1.518e8→9.82e7 |
| 3 | Trainable externals not constant-folded (swap observable) | ✅ **PASS** — implied by #2 (desktop + device) |
| 5 | GenAI Android resolves relative external data in the package dir | ✅ **PASS (device)** — `OgaCreateModel` loaded + generated |
| 7 | Memory: mmap vs copy | ✅ measured — desktop 199 MB blob → RSS +144 MB; device load +102 MB (mmap/lazy, not 2×) |
| 4 | GenAI peak RSS within threshold of Native (device) | ⏳ needs a File #9 package to run Native side too |
| 1 | Same package correct under BOTH engines (device) | ⏳ needs a real File #9 package (per-tensor externals) |

### The device blocker — and how it was RESOLVED (ORT engine separation)

**Diagnosis.** The genai `0.14.0` AAR bundles **only `libonnxruntime-genai.so`, no `libonnxruntime.so`** —
GenAI `dlopen`s whatever `libonnxruntime.so` the app ships. This app ships the **source-built ORT-*training***
`libonnxruntime.so` (ORT **1.23**), but **genai 0.14 requires stock ORT ≥ 1.26** (`onnxruntime-genai`'s pip
metadata: `onnxruntime>=1.26.0`; desktop worked with 1.27.0). So `OgaCreateModel` aborted (SIGABRT) on model
load. The external-data mechanism itself was never the problem (proven on desktop). The real issue was
**two different ORTs** — training 1.23 (with the training C++ API the Native engine needs) vs stock 1.27
(genai-paired inference) — that must coexist in one process but share SONAME `libonnxruntime.so`.

**Resolution — implemented and verified on device.** Give GenAI its own stock ORT under a distinct name so
the two never collide (reproducible via `setup_ort_separation.sh`):

| Consumer | Library | Notes |
| --- | --- | --- |
| Native / training engine (`libmobiletransformers.so`) | `jniLibs/<abi>/libonnxruntime.so` | the source-built ORT-training 1.23 (unchanged; linked as `-lonnxruntime`) |
| GenAI engine | `jniLibs/<abi>/libort_gen.so` | stock ORT **1.27**; SONAME **raw-patched** `libonnxruntime.so`→`libort_gen.so` (no `patchelf` — it corrupts `verneed`; a length-preserving in-place byte edit keeps all offsets) |
| GenAI dispatcher | `jniLibs/<abi>/libonnxruntime-genai.so` | `dlopen` target **raw-patched** `libonnxruntime.so`→`libort_gen.so` (and the `.so.1` fallback) |

Why it's safe: each ORT **exports only ~3 symbols** (hidden visibility — `OrtGetApiBase` + two EP appenders),
and genai resolves ORT via `dlsym` on **its own `dlopen` handle**, so there is **no symbol interposition**
between the two ORTs. The distinct SONAME is essential — with the same soname the linker dedups and hands
genai the already-loaded training lib back (this exact failure was observed). The genai AAR is **not** a
Gradle dependency (its Java classes are unused on the C-API/JNI path), so only the patched `.so` ships.

**Verified on device (SM-G990B, arm64):** `GenAISpikeTest` passes — `libmobiletransformers.so` loads with the
**training** ORT 1.23 present, GenAI `OgaCreateModel` loads the model via the **stock** ORT 1.27
(`libort_gen.so`), generates a token (relative external data resolved), and overwriting one external weight
changes the output on a fresh model. **Both engines' ORTs coexist in one process.** This closes the F2 /
Gate 0.1 GenAI-side question and hands #11 a working engine-coexistence design.

### Running the device test
1. Build a model (`build_tiny_genai_model.sh` → `build/genai_spike_model`) and set up separation
   (`setup_ort_separation.sh`).
2. `installDebugAndroidTest -Pandroid.injected.build.abi=arm64-v8a`.
3. Stage the model where the app process reads it — the app's **internal** files dir is most reliable:
   ```bash
   adb shell mkdir -p /data/local/tmp/mt_genai_spike/inference
   adb push build/genai_spike_model/. /data/local/tmp/mt_genai_spike/inference/
   PKG=com.martinkorelic.mobiletransformers.test
   adb shell run-as $PKG mkdir -p files/mt_genai_spike/inference
   for f in genai_config.json model.onnx model.onnx.data tokenizer.json tokenizer_config.json chat_template.jinja; do
     adb shell "run-as $PKG sh -c 'cp /data/local/tmp/mt_genai_spike/inference/$f files/mt_genai_spike/inference/$f'"; done
   ```
   (The test tries `filesDir`, `/data/local/tmp`, then the external files dir, and *skips* if none has a
   `genai_config.json`. The internal `filesDir` copy survives `install -r`.)
4. `adb shell am instrument -w -e class com.martinkorelic.mobiletransformers.GenAISpikeTest \
   com.martinkorelic.mobiletransformers.test/androidx.test.runner.AndroidJUnitRunner` · watch
   `adb logcat -s GenAISpike`. Cleanup: `adb shell rm -rf /data/local/tmp/mt_genai_spike`.

**Note on #1/#4 (cross-engine):** the builder model is single-blob GenAI format — enough to prove the
GenAI-side items. Full cross-engine equivalence (#1/#4) also needs a real File #9 package (per-tensor
externals + `weight_handoff_map.json` consumable by BOTH `ORTGeneratorNative` and GenAI). `desktop_spike.py`
and `GenAISpikeTest.kt` already handle both layouts.
