# Mac Work Plan — moving the restructure to macOS

Guide for continuing on **macOS**. As of 2026-07-15 **all of #1–#29 are code-complete** with host tests
green (Python `make check` 215 passed, Android 125 SDK JVM tests, arm64 `assembleDebug` links); the only
open legs are **device acceptance runs**, now one command away (`make device-package` → `make device-test`).
So Mac work is: (a) run those device legs (needs a connected Android device + `adb` on the Mac), and
(b) the not-yet-started plans — release **#30/#32**, remaining docs **#31**, and Tier-3 **#33/#34/#36/#37**.
Read `agent_docs/HANDOFF.md` for full state.

---

## 0. Export checklist (the NOT-git-tracked files to carry over)

`git clone`/push brings ALL source (Python, Kotlin, C++ incl. the `cpp/onnxruntime*/` headers, docs,
schemas, build files). These paths are **git-ignored** and must be moved out-of-band (Drive/zip/scp).
All under `android/MobileTransformersApp/MobileTransformers/src/main/` unless noted.

- [ ] `jniLibs/arm64-v8a/`  — Android arm64 device libs (~1.2 GB w/ x86_64): `libonnxruntime.so`,
      `libort_gen.so`, `libonnxruntime-genai.so`(+`-jni.so`), `libtokenizers_c.a`, `libprotobuf-lite.a`,
      `libobjectbox-jni.so`, `libonnxruntime4j_jni.so`  → **move** (device binaries, host-agnostic)
- [ ] `jniLibs/x86_64/`  — same for the emulator ABI  → **optional** (upstream-incomplete; real device = arm64 only)
- [ ] `aarLibs/onnxruntime-genai.aar`  (~40 MB)  → **move**
- [ ] `cpp/includes/`  (~24 MB, protobuf headers)  → **move**
- [ ] `third_party/wheels/onnxruntime_training-…-linux_x86_64.whl`  (~632 MB)  → **DO NOT move**
      (linux/cp312; won't load on macOS — rebuild for macOS-arm64 only if you need the training side, §4)

Skip `*.old.bak`/`*.stub.bak`. Don't move recreatables (`.venv*`, `build/`, `cache_dir/`, `.gradle`,
`.cxx`, `onnx_models/`, `dist/`). One-liner to list them + zip the movable set:
```bash
git status --porcelain --ignored | grep '^!!' | grep -E 'jniLibs|aarLibs|cpp/includes|wheels/.*\.whl'
cd android/MobileTransformersApp/MobileTransformers/src/main && \
  zip -r ~/mtf-native.zip jniLibs/arm64-v8a aarLibs/onnxruntime-genai.aar cpp/includes -x '*.bak'
```
**Commit + push the repo first** — the working tree is the only copy of the source. §1 has the details.

---

## 1. What you actually need to move (git-tracked vs. not)

**Git-tracked → arrives with `git clone`/push, nothing to do:** all Python (`src/mobiletransformers/`,
`tests/`, `spikes/`), **all** Android Kotlin **and C++ source** — including the ORT/GenAI C++ headers
under `cpp/onnxruntime/` and `cpp/onnxruntime-genai/` (these ARE tracked) — `agent_docs/`, `schemas/`,
`docs/`, `pyproject.toml`/`uv.lock`, `Makefile`, `scripts/`, `third_party/wheels/README.md`,
`third_party/onnxruntime/{manifest.json,BUILD.md}`. **Commit + push first** — the working tree is the only
copy of this session's work.

**NOT git-tracked → must move out-of-band** (Drive/zip/scp), all under
`android/MobileTransformersApp/MobileTransformers/src/main/`:

| Path | Size | What it is | Move to Mac? |
| --- | --- | --- | --- |
| `jniLibs/arm64-v8a/` | (bulk of ~1.2 GB) | Android **arm64 device** libs: `libonnxruntime.so` (source-built ORT-training 1.23), `libort_gen.so` (stock ORT 1.27, soname-patched), `libonnxruntime-genai.so`+`-jni.so` (patched genai 0.14), `libtokenizers_c.a`, `libprotobuf-lite.a`, `libobjectbox-jni.so`, `libonnxruntime4j_jni.so` | ✅ yes — device binaries, host-OS-agnostic |
| `jniLibs/x86_64/` | (part of ~1.2 GB) | same set for the emulator ABI | ⚠️ optional — upstream-**incomplete** (arm64 is the good one); a real device only needs arm64-v8a |
| `aarLibs/onnxruntime-genai.aar` | ~40 MB | genai 0.14 AAR (CMake link input + clean upstream headers source) | ✅ yes |
| `cpp/includes/` | ~24 MB | protobuf headers (`google/` + `protobuf/`) — CMake native-build inputs | ✅ yes |
| `third_party/wheels/onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl` | ~632 MB | source-built ORT-training wheel | ❌ **NO** — `linux_x86_64` cp312 C-extension, **cannot load on macOS** (see §4) |

Skip the `*.old.bak` / `*.stub.bak` files in `jniLibs`/`aarLibs` (dev cruft). **Do not move** the
recreatable dirs: `.venv*`, `.venv-genai-spike/`, `build/`, `cache_dir/`, `.gradle`, `.cxx`,
`onnx_models/`, `dist/`.

Enumerate/verify the out-of-band set on this box before you copy:
```bash
git status --porcelain --ignored | grep '^!!' | grep -E 'jniLibs|aarLibs|cpp/includes|wheels/.*\.whl'
# zip the movable ones (excludes the linux wheel + backups):
cd android/MobileTransformersApp/MobileTransformers/src/main
zip -r ~/mtf-native.zip jniLibs/arm64-v8a aarLibs/onnxruntime-genai.aar cpp/includes -x '*.bak'
```
On the Mac, unzip into the same relative paths. (Or re-provision from the sibling `../ORTTransformer`
checkout, the same source these came from here.)

**Gradle deps are NOT a move:** the W5 additions (OkHttp, WorkManager, kotlinx-coroutines, mockwebserver,
androidx-test-runner) resolve from Maven Central over the network on first build — nothing to carry.

---

## 2. Prerequisites

### Track A — Python only
- macOS + [Homebrew](https://brew.sh); `uv` (`brew install uv`). `uv` fetches Python 3.10 (core/dev, mypy
  target) + 3.12 (export) on demand from `uv.lock`.
  ```bash
  uv sync --frozen --group dev                       # core + dev; `make check` works
  uv sync --extra export --python 3.12               # optimum inference export (host)
  uv sync --extra train  --python 3.12               # torch/peft/safetensors (CPU) — #22, #35 local
  uv sync --group genai-smoke --python 3.12          # onnxruntime-genai desktop (mmap/dual-engine spikes)
  ```
  None of these pull the Linux ORT-training wheel, so its absence is a non-issue for the host gate.

### Track B — Full Android (recommended)
Pinned toolchain (from `android/MobileTransformersApp`): **AGP 8.5.1, Kotlin 1.9.0, Gradle 8.7,
compileSdk 34, minSdk 24, CMake 3.22.1, NDK r26.x, ObjectBox 4.3.0**.
- **JDK 17** to run Gradle (`brew install --cask temurin@17`, or Android Studio's JBR); set `JAVA_HOME`.
- **Android SDK** (Android Studio or cmdline-tools): `android-34`, `build-tools;34.x`, `platform-tools`
  (gives `adb`), `cmake;3.22.1`, and the NDK AGP resolves. Set `ANDROID_HOME`/`ANDROID_SDK_ROOT`.
- Drop the moved native deps (§1) into their paths — `aarLibs/` + `jniLibs/arm64-v8a` + `cpp/includes/`
  are needed for `compileDebugKotlin` (types) and the arm64 native build in `assembleDebug`.
- Verify: `JAVA_HOME=<jdk17> ./gradlew :MobileTransformers:testDebugUnitTest \
  :MobileTransformers:compileDebugAndroidTestKotlin :MobileTransformers:assembleDebug \
  -Pandroid.injected.build.abi=arm64-v8a`.

---

## 3. Running the device legs on Mac (the main remaining work for #1–#29)

Everything for #1–#29 is written; a connected device + `adb` turns the boxes green. The inference+GenAI
package builds entirely on the Mac (export profile); the train-capable package needs the ORT-training
wheel (§4).

```bash
# inference+GenAI package (Mac-native) -> reshape -> adb push -> run instrumented suites:
make device-package MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
make device-test        # FacadeLoadGenerateTest, DualEngineParityTest, ConversationResetTest, RagDeviceTest*
```
Instrumented classes `assumeTrue`-skip without a pushed package, so a device-less Mac still runs green.
`*RagDeviceTest`/`TrainMergeGenerateTest` self-skip unless the package carries `embedding/`/`train/`.

---

## 4. The one hard macOS limitation — ORT-training

`third_party/wheels/onnxruntime_training-…-linux_x86_64.whl` **will not install/load on macOS**. It gates
the **training side**: `_build_training_stage` (#15), `tests/integration/test_training_stage_smoke.py`
(#3/#15), `gen_artifacts`/`optimum_hf_export`, and `make device-package TRAIN=1`. Consequences on Mac:
- **Works without it:** all core/dev + export + genai-smoke Python, every Android host build/test, the
  inference+GenAI device package + its device legs (#9-load / #11 / #17 / #23 / #24), #22 PEFT
  materialization (needs only the `train` extra = torch CPU, not ORT-training), #35 federated (torch CPU).
- **Blocked until you rebuild the wheel for macOS-arm64:** the real train→merge→generate legs (#18/#19),
  the train-capable package, and #34's real training run. Rebuild via
  `scripts/build_ort_training_*.sh` + `third_party/onnxruntime/BUILD.md` (source build of onnxruntime-training
  for `macosx_arm64` cp312), then point the `ort-training-local` group at the new wheel in `pyproject.toml`.
  Big, but a one-time cost if training-on-Mac is required. Otherwise keep the training legs on this Linux box.

---

## 5. Remaining plans (not-yet-started) — all Mac-doable

#1–#29 are code-complete; these are the genuinely-open plans for v1.0+ and Tier-3.

| # | Plan | Impl + test on Mac | Device leg only | Track |
| --- | --- | --- | --- | --- |
| 30 | AAR / Maven publication | `assembleRelease` + `publishToMavenLocal` + external consumer-app build — **all no-device** | — (host-only) | B (+NDK) |
| 31 | Docs set (finish) | `ARCHITECTURE.md`/`ANDROID_SDK.md` (contracts now locked by #23/#24) + CI link-check | — | A |
| 32 | Versioning / license / release | SPDX headers, version-site sync, CHANGELOG, license expression | full release gate (CI+AAR+device) | A |
| 33 | Encoder support | arch-registry entry + export config + host inference-export smoke | train step + Android smoke | A/B |
| 34 | Training scheduler | `LinearLRScheduler.stateDict/loadFromState` (pure) + WorkManager worker + resume + JVM tests | Doze/thermal multi-chunk (+ real train → §4) | B |
| 36 | Federated Android | JNI export/import + byte-identical codec golden vs #35 + privacy gate + JVM tests | real client/gateway run | B |
| 37 | FunctionGemma | Gemma-3 arch entry + inference-graph export config + tool-call allowlist/validation + intents | train→tool-call→intent demo | A/B |

**Flagship Mac plans:** #30 (AAR+consumer build, host-only), #31 docs, #34 scheduler state (pure logic +
WorkManager, mirrors the #21 worker pattern), #36 codec golden (reuses #35's pinned serialization).

---

## 6. Verify commands (Mac)
```bash
make check                                              # Python: lint + typecheck + parity + tests
uv run --extra export --python 3.12 mobiletransformers export --model <id> --output build/pkg --genai --dry-run
# Track B (from android/MobileTransformersApp):
JAVA_HOME=<jdk17> ./gradlew :MobileTransformers:testDebugUnitTest \
  :MobileTransformers:compileDebugAndroidTestKotlin :app:compileDebugKotlin \
  :MobileTransformers:assembleDebug -Pandroid.injected.build.abi=arm64-v8a
# Device (connected + adb): make device-package MODEL=<id> && make device-test
# Desktop GenAI/mmap invariants (genai-smoke): python -m spikes.mmap.base_blob_mmap_spike --dir build/pkg/variants/cpu-int4/inference
```

Out of reach on Mac without the wheel rebuild (§4): real ORT training-artifact gen / a train step, and the
train-capable package. The #12 device RSS table (Gate 0.2) and any emulator/device instrumentation are
device legs by nature. **Nothing is committed by the assistant — you commit.**
