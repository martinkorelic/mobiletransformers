# Mac Work Plan — implement-ahead without a device

Guide for continuing on **macOS without an Android device**. The core insight: for almost every
remaining plan the device is only the **final acceptance leg** (does it actually run/measure on
hardware). The *implementation*, static checks (compile + type), and *unit/integration tests* (with
fixtures/mocks) can nearly all be done ahead of time. Read `agent_docs/HANDOFF.md` for full state.

## TL;DR — how much is reachable

- **With the full Android toolchain set up (Track B): ~80% of the remaining implementation** across
  Tiers 1–3 can be written + compiled/type-checked + unit/integration-tested on the Mac. The device
  becomes a *verification* step you run later, not a blocker to writing the code.
- **Python-only (Track A):** still substantial — the entire federated plan (#35), the Python halves of
  #33/#37, #22 materialization, #31 docs, #32 license/version, and all core/export Python.
- **Genuinely NOT reachable on Mac:** running real ML — ORT **training-artifact generation / a real
  train step** (Linux/cp312 wheel; no macOS ORT-training) and **real on-device generate** — plus the two
  measurement spikes that *are* device experiments by nature (#10 GenAI RSS/symbol, #12 mmap RSS).

## The verification model (what "done on Mac" means)

Every plan's `Tests & acceptance` is already split **unit → integration → manual(device)**. On the Mac we
complete the first two legs for each plan and leave a clearly-marked device stub for the third:
- **Unit (Mac):** `pytest` (Python) / `testDebugUnitTest` (Kotlin JVM) / host C++ tests. Compile +
  `mypy`/`ruff` + `compileDebugKotlin` are the static gate.
- **Integration (Mac):** fixture-in → asserted-out, mocking the JNI / model / network boundary
  (`InMemoryVectorStore`, injected `downloader`/`uploader`, injected detection, synthetic caches).
- **Manual (later, device):** the real run — write it as a documented, skipped/`workflow_dispatch` stub.

## Prerequisites

### Track A — Python only (no Android)
- macOS + [Homebrew](https://brew.sh).
- `uv` (`brew install uv` or the astral installer) — drives all Python envs from `uv.lock`.
- Python 3.10 (core/dev, mypy target) and 3.12 (export/train profiles); `uv` fetches these on demand.
- Setup:
  ```bash
  uv sync --frozen --group dev                 # core + dev; `make check` works
  uv sync --extra export --group dev --python 3.12   # optimum export (host)
  uv sync --extra train  --group dev --python 3.12   # torch/peft/safetensors (CPU) for #22, #35 local training
  ```
  None of these pull the (Linux-only) ORT-training wheel, so its absence is a non-issue.

### Track B — Full Android (recommended, unlocks the Kotlin/C++ plans)
Pinned toolchain (from `android/MobileTransformersApp`): **AGP 8.5.1, Kotlin 1.9.0, Gradle 8.7,
compileSdk 34, minSdk 24, CMake 3.22.1, ObjectBox 4.3.0**.
- **JDK 17** to *run* Gradle (`brew install --cask temurin@17`, or use Android Studio's JBR). Set
  `JAVA_HOME` to it. (Modules compile to JVM 1.8 bytecode, but AGP 8.5 needs JDK 17 to run.)
- **Android SDK**: install Android Studio (easiest) or command-line tools. Via SDK Manager get:
  platform **android-34**, **build-tools;34.x**, **platform-tools**, **cmake;3.22.1**, and the **NDK**
  AGP 8.5.1 resolves (r26.x — `ndkVersion` isn't pinned, so let AGP pick, or install `ndk;26.1.10909125`).
  Set `ANDROID_HOME` / `ANDROID_SDK_ROOT`.
- **The vendored native deps** (git-ignored — carry them over, see next section): `aarLibs/` (needed for
  `compileDebugKotlin` — types), `jniLibs/` + `cpp/includes/google/` (needed for the NDK/CMake native
  build in `assembleDebug`/`assembleRelease`).
- Optional: **Robolectric** only if we add unit tests that touch Android framework classes (current tests
  deliberately avoid it — plain JUnit4 on the JVM).
- Verify the setup: `./gradlew :MobileTransformers:testDebugUnitTest :MobileTransformers:assembleDebug`.

### Moving artifacts to the Mac (Drive / zip / scp)
| Artifact | Move it? | Why |
| --- | --- | --- |
| `android/.../src/main/aarLibs/` | ✅ yes | Android `.aar`s — host-OS-agnostic; needed for `compileDebugKotlin`. |
| `android/.../src/main/jniLibs/` | ✅ yes | Android-ABI `.so`s — consumed by the build, never run on host. |
| `android/.../src/main/cpp/includes/google/` | ✅ yes | protobuf headers — needed for the CMake native build. |
| `third_party/wheels/onnxruntime_training-*.whl` | ❌ no | `linux_x86_64` cp312 C-extension — cannot load on macOS. |
| Android SDK / NDK / JDK | ❌ install fresh | Use the macOS builds via SDK Manager / brew, don't copy Linux ones. |

Zip the three ✅ dirs, drop them on Drive, unzip into the same paths on the Mac. (Or clone
`../ORTTransformer` on the Mac and re-copy from there — same source they were provisioned from here.)
**Commit + push the repo first** — the tree is the only copy of this session's work.

## Coverage map (remaining plans)

Legend: **Impl+test on Mac** = write code + static compile/type + unit/integration (mock/fixture) tests.
**Device/Linux leg** = the acceptance-only step deferred.

| # | Plan | Impl + test on Mac | Device/Linux leg only | Track |
| --- | --- | --- | --- | --- |
| 10 | GenAI swap spike | swap harness + host symbol-probe scaffold (low) | RSS + observable-output Gate 0.1 | B |
| 11 | Engine abstraction | `ModelRuntime`/`InferenceEngine` enum (+Py mirror+parity)/impls/selector/fallback + JVM tests w/ mocked JNI; callback-parity test w/ mocks | real cross-engine parity lock | B |
| 12 | mmap experiments | experiment harness code (low) | RSS numbers Gate 0.2 | B |
| 17 | Facade foundation | `fromPretrained`/`ModelSession`/exceptions/selector + JVM tests (mock repos) | load→generate 1 token | B |
| 18 | Training lifecycle | `TrainingJob`/event adapter/checkpoint format/session lock/cancellation + JVM tests | real train run | B |
| 19 | HF Kotlin facade | applyPeft/train/merge/generate/retrieve + config-mapping table + JVM tests | train→merge→generate | B |
| 23 | Native load hardening | map-driven fail-closed load (C++/Kotlin) compiles; conversation-reset fix + reset test | load-and-generate on device | B (+NDK) |
| 24 | Sampling/streaming | `SamplingMethod` enum+mirror+mapping+defaults + JVM tests | cross-engine callback parity | B |
| 26 | RAG ingestion | chunking + document-loader registry + ingest via `InMemoryVectorStore` + JVM tests | real embedding step (host-ORT or device) | B |
| 27 | RAG grounded gen | `RagConfig` + retrieve→assemble→prompt logic + JVM tests | full grounded generate | B |
| 30 | AAR/Maven | `assembleRelease` + `publishToMavenLocal` + external consumer app build — **all no-device** | — (host-only) | B (+NDK) |
| 31 | Docs (remaining) | `MODEL_FORMAT.md`, `CONFIGURATION.md` now (locked); others as contracts lock | pages for device contracts | A |
| 32 | Versioning/license | SPDX headers, version-site sync, CHANGELOG, license expression | full release gate (CI+AAR+device) | A |
| 33 | Encoder support | arch-registry entry + export config + host inference-export smoke | train step + Android smoke + MARS-transfer verify | A/B |
| 34 | Training scheduler | `LinearLRScheduler.stateDict/loadFromState` (pure) + WorkManager worker + resume logic + JVM tests | Doze/thermal/energy multi-chunk | B |
| 35 | **Federated (Flower sim)** | `FederatedAdapterRecord` from codec + Python N-client Flower sim + Py↔Kotlin byte-parity golden — **all host** | — (Option A is a Python sim; device is Option B/#36) | A |
| 36 | Federated Android | JNI export/import + byte-identical codec golden + privacy-gate logic + JVM tests | real client/gateway run | B |
| 37 | FunctionGemma | Gemma-3 arch entry + inference-graph export config + tool-call allowlist/dry-run/validation + intents | train→tool-call→intent demo | A/B |

**Flagship Mac plans (highest coverage, lowest device dependence):** **#35** (entirely host Python),
**#11/#17/#18/#19/#24** (facade+engine — implement fully, mock the JNI), **#34** (scheduler state is pure
logic), **#30** (AAR+consumer build is host-only).

## Concrete first tasks on the Mac (ordered)

1. **#22 materialize_peft_weights** (`adapter/convert.py`) — read the A/B `.bin` externals from the cache
   (`AdapterPackage.tensors`, dtype/shape) → `numpy`→`torch`→`safetensors.save_file`; **needs only the
   `train` extra (CPU), not ORT-training** (the stub docstring's "ORT CheckpointState" is pessimistic —
   the cache already has raw `.bin` bytes). Test behind `importorskip("torch")`.
2. **#31 remaining docs** — `docs/MODEL_FORMAT.md` (#8/#9/#13/#14) + `docs/CONFIGURATION.md` (#6) are
   locked; write them (pure markdown).
3. **#35 federated (Python)** — the whole Option-A Flower simulation + codec-derived record + golden. Big,
   self-contained, no device. If local per-client training needs a runtime, use torch-CPU (Mac-ok), not
   ORT-training.
4. **(Track B) #11 + #17** — the engine enum/`ModelRuntime` + facade foundation, compiled + JVM-tested
   against mocked repositories/JNI. This unblocks #18/#19/#24 as pure-logic + mock work.

## Verify commands (Mac)
```bash
make check                                   # Python: lint + typecheck + parity + tests
uv run --extra train pytest tests/adapter -q # after #22 (torch CPU)
# Track B:
./gradlew :MobileTransformers:testDebugUnitTest :MobileTransformers:assembleDebug
```

## Out of reach on Mac (defer, write as skipped/device stubs)
- Real **ORT training** (artifact gen / a train step) and real **on-device generate** — Linux/device.
- **#10 / #12** — device RSS/symbol measurements (they *are* the experiment).
- Any emulator/device instrumentation — `device.yml` (`workflow_dispatch` + nightly) is their harness.

Nothing is committed by the assistant — **you** commit. Keep the "nothing committed" discipline.
