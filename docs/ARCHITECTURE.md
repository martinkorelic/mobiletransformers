# Architecture

MobileTransformers has two halves that meet at **one on-disk contract**:

- a **host (Python) exporter** that turns an HF model into a device-ready package, and
- an **Android SDK (Kotlin + C++)** that installs that package, trains against it, merges the result
  back into it, and generates from it.

Nothing crosses that boundary except files. There is no RPC, no shared process, and no code sharing —
which is why the file formats ([MODEL_FORMAT.md](MODEL_FORMAT.md),
[HUB_PACKAGE_FORMAT.md](HUB_PACKAGE_FORMAT.md)) are specified as carefully as they are.

```
   HOST (Python)                          HUB                    DEVICE (Android)
┌──────────────────────┐          ┌────────────────┐      ┌────────────────────────────┐
│ export.pipeline      │  push    │ model package  │ pull │ HubDownloader              │
│  ├ inference stage   ├─────────►│  manifest      ├─────►│  └ VariantSelector         │
│  ├ training stage    │          │  variants/     │      │ ModelPackageInstaller      │
│  └ embedding stage   │          │  shared/       │      │        │                   │
└──────────┬───────────┘          └────────────────┘      │        ▼                   │
           │                                              │ MobileTransformers         │
           │ weight_handoff_map.json                      │  .fromPretrained           │
           └──────────── the contract ────────────────────┤        │                   │
                                                          │        ▼                   │
                                                          │ MobileTransformerModel     │
                                                          │  train ─► merge ─► generate│
                                                          └────────────────────────────┘
```

## The three registries

Closed-set behaviour is resolved from **data**, never from string comparisons. Adding a model, PEFT
method or merger is a registry row, not a new `if` branch — and a CI guard (`make guard`) fails the
build if a dispatch literal reappears.

| Registry | Keyed by | Answers |
| --- | --- | --- |
| `config/registry/architecture.py` | `config.architectures[0]` | which Optimum `OnnxConfig`, which inference builder, target modules, attention-module name |
| `config/registry/peft.py` | `PEFTMethod` | which PEFT config class, the adapter component schema (the codec's tensor order), the merger variant, the adapter-mapping builder |
| `config/registry/merger.py` | `(PEFTMethod, quant_in, quant_out)` | the resolved `MergerVariant` + the merger ONNX filename |

The resolved `MergerVariant` is written into the package, so the device selects its merger session from
a typed tag rather than re-deriving one.

## Enums cross three languages

`config/constants.py` is the single source of truth. `codegen/enums.py` generates `schemas/enums.json`
and checks the Kotlin mirrors under `constants/`; the C++ mirror
(`cpp/constants/merger_variant.h`) is covered by the host googletest suite. `make parity` fails on any
drift, so a wire value cannot change in one language only.

## Host: the export pipeline

`export_package` resolves a side-effect-free `ExportPlan` (what `--dry-run` prints), then builds the
selected **stages** into a staging tree and hands the reshape + manifest to `assemble_package`:

| Stage | Profile | Produces |
| --- | --- | --- |
| `inference` | `export` (optimum-onnx, py3.12) | normalized `model.onnx` (+ `model.onnx_data`), `generation_config.json`, `genai_config.json`, `optimum_config.json`, tokenizer, an empty handoff map |
| `training` | `ort-training-local` (py3.12) | training/eval/optimizer graphs, `checkpoint/`, `trainable_parameters.json`, and the **real** handoff map (the trainable split) |
| `embedding` | `export` | the RAG embedding subtree |

The two profiles are declared **conflicting** in `pyproject.toml` and cannot co-install, so a full
package is produced by two runs into the same `--output`; re-assembly preserves the stages already
present. Stage selection is automatic by request + importable dependencies, overridable with `--stages`,
and a skipped stage is logged rather than silently dropped.

## Device: engines

`ModelRuntime` is the engine interface. Two implementations read the **same** `inference/` directory:

| Engine | Backing | Notes |
| --- | --- | --- |
| `NATIVE` | ONNX Runtime **training** build, `cpp/` | the guaranteed floor; owns training, merging and the KV-cached generation loop |
| `GENAI` | `onnxruntime-genai` | opt-in; requires `genai_config.json` in the package |

`ModelRuntimeFactory.selectEngine` is pure and reads `ORTGenerationConfig.engine`; `.create` performs
the availability probe and falls back to Native transparently. Callers never branch on engine, and both
engines drive identical callback sequences.

**The two runtimes coexist by soname separation.** GenAI needs stock ORT ≥1.26 while the native trainer
is built against ORT-training 1.23, and the GenAI AAR ships no ORT of its own — it `dlopen`s the app's.
Shipping a distinct-soname `libort_gen.so` and repointing GenAI's `dlopen` at it lets both load in one
process. Each ORT exports only a handful of symbols (hidden visibility) and GenAI resolves via `dlsym`
on its own handle, so there is no interposition; the distinct soname is essential, since a shared one
makes the linker dedupe them.

## Device: train → merge → generate

1. **Train.** `ORTTrainerNative` runs the ORT training loop against `train/`. Checkpoints are
   `train/checkpoint` + `training_state.json`; cancellation is cooperative (`cancelRequested` checked
   at the step/epoch loop tops) so the existing save path still runs.
2. **Merge.** `weight_merger.cpp` reads the handoff map, runs the resolved merger graph per trainable
   layer, and writes each merged tensor's **raw bytes** over the exact `<name>.bin` the inference graph
   references — atomically, with a refreshed `.sha256` sidecar. A partial merge fails the whole
   operation rather than reporting success.
3. **Generate.** `HandoffPrecondition` gates the load (map present, every `.bin` present, checksums
   match) *before* the session is created; `session_cache.h` then loads those raw bytes as external
   initializers using the map's per-role dtype/shape. Any failure aborts session creation instead of
   quietly falling back to the frozen base weights.

The invariant throughout: **a wrong model is worse than no model.** Every gate on this path fails
closed, because a silently-unmerged model still generates fluent text and looks healthy.

## Native dependencies

**A `git clone` cannot build the Android SDK.** Roughly 180 MB of prebuilt binaries and vendored
headers are gitignored — they are the only thing the clone does not bring, and everything else
(`build/`, `.venv*`, `.gradle`, `.cxx`) is recreatable.

| what | where | size |
| --- | --- | --- |
| ONNX Runtime, GenAI, tokenizers, protobuf | `MobileTransformers/src/main/jniLibs/arm64-v8a/` (8 files) | 116 MB |
| the GenAI Android AAR | `MobileTransformers/src/main/aarLibs/onnxruntime-genai.aar` | 40 MB |
| protobuf headers/sources | `MobileTransformers/src/main/cpp/includes/{google,protobuf}` | 24 MB |
| the source-built ORT-training wheel | `third_party/wheels/onnxruntime_training-…-cp312-linux_x86_64.whl` | 632 MB |

```bash
make doctor                                  # what is missing, and the command that fixes each
make fetch-native-deps                       # the Android natives
TRAINING=1 scripts/fetch_native_deps.sh      # those plus the ORT-training wheel (632 MB)
SYMBOLS=1  scripts/fetch_native_deps.sh      # plus the unstripped debug symbols (260 MB)
```

[`third_party/android/manifest.json`](https://github.com/martinkorelic/mobiletransformers/blob/main/third_party/android/manifest.json) is the source of truth:
one entry per artifact with its destination, size, **sha256** and provenance.
`scripts/fetch_native_deps.sh` reads it, verifies the archive hash, unpacks, then verifies every file
individually. It refuses rather than half-populating — a partly-filled `jniLibs/` fails the link
naming a missing *symbol*, not a missing file.

The training wheel is fetched separately (`TRAINING=1`) because only the export host needs it. It is
`cp312` + `linux_x86_64`, so **macOS and Windows cannot run the training side** without rebuilding it
(`third_party/onnxruntime/BUILD.md`). Everything else — every host gate, every Android build,
inference export, PEFT materialization, federated — works anywhere, and `make check` does **not**
need the wheel: the Makefile runs `uv run --frozen` precisely so a missing local wheel cannot break
an unrelated target.

### Why two ONNX Runtimes ship side by side

GenAI 0.14 needs stock ORT ≥ 1.26; the Native/training path needs the source-built ORT-training 1.23.
Both would otherwise carry the soname `libonnxruntime.so`, the linker dedups them, and GenAI silently
gets the training build — observed as a SIGABRT.

The resolution is a distinct name, not a version bump: stock ORT 1.27 ships as `libort_gen.so` with a
**raw-patched SONAME** (not `patchelf`, which corrupts `verneed`), and the genai `.so`'s `dlopen`
target is raw-patched to match. Training ORT keeps `libonnxruntime.so`. This is safe because each ORT
exports only a handful of symbols under hidden visibility and GenAI resolves through `dlsym` on its
own handle, so there is no interposition. Reproducible via
`spikes/genai_external_swap/setup_ort_separation.sh`.

### arm64-v8a only

`jniLibs/x86_64/` never had `libonnxruntime.so` or the tokenizer archives — absent here *and*
upstream — so `libmobiletransformers.so` has never existed for that ABI. It was dropped from
`abiFilters` rather than advertising an ABI that dies at `System.loadLibrary`. Restoring it means
building ORT-training and tokenizers-cpp for x86_64 first. **There is no x86_64 emulator path.**

The shipped binaries are stripped. AGP strips native libraries at packaging anyway, and to the same
bytes as `llvm-strip --strip-all`, so unstripped prebuilts cost clone size and never APK size. The
unstripped originals exist only in the debug-symbols bundle and cannot be regenerated without a full
ORT source build.

## Concurrency

One native session at a time. `LLMRepository` holds a `Mutex` across session create/teardown for the
training, generation and retrieval paths — the ORT handles are not safe to swap concurrently. The lock
is *not* held across a full training run or generation loop, so a long job never blocks release.

## Testing

| Layer | Harness | Runs |
| --- | --- | --- |
| Python | pytest (`make check`) | every PR |
| Kotlin | JUnit + Robolectric (`make test-jvm`) | every PR |
| C++ | googletest, ORT-free headers (`make test-cpp`) | every PR |
| Guards | `make guard` — secret reads, dispatch literals | every PR |
| Android assemble | Gradle + NDK | self-skips without the vendored native libs |
| Device | `androidTest` | manual: `make device-package` → `make device-test` |

Everything that can be tested without a device is; the instrumented classes all `assumeTrue` on a
pushed package, so they skip rather than fail when one is absent.
