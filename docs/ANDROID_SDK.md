# Android SDK

The `mobiletransformers-android` AAR: load an exported package on a device, generate, fine-tune it with
LoRA, merge the result back into the inference graph, and query a local RAG store — all on device, with
no server round trip.

This page is the **consumer's** view. For what a package contains see
[MODEL_FORMAT.md](MODEL_FORMAT.md); for how one is produced see [EXPORT.md](EXPORT.md); for the exact
Kotlin signatures see [PUBLIC_API.md](PUBLIC_API.md).

## Requirements

| | |
| --- | --- |
| `minSdk` | 24 |
| `compileSdk` | 34 |
| ABI | **`arm64-v8a` only** |
| Storage | the package, uncompressed — a 135M int4 package is ~1.3 GB with the training stage |

**arm64-v8a is the only shipped ABI in v1.** The AAR carries a >1 GB ONNX Runtime `.so` plus the GenAI
runtime, and the x86_64 builds of ONNX Runtime and tokenizers-cpp do not exist in this project. The
build refuses to publish an ABI whose native inputs are missing rather than shipping a variant that
fails at `System.loadLibrary`, so **the library does not run on an x86_64 emulator** — development
needs a physical arm64 device.

## Install

The AAR is not on Maven Central yet. Publish it locally and consume it from there:

```bash
make publish-local      # -> ~/.m2/repository/com/martinkorelic/mobiletransformers/
```

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        mavenLocal()
        google()
        mavenCentral()
    }
}
```

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("com.martinkorelic.mobiletransformers:mobiletransformers-android:<version>")
}

android {
    defaultConfig {
        ndk { abiFilters += listOf("arm64-v8a") }
    }
}
```

A worked example lives in [`examples/consumer-app/`](../examples/consumer-app/), which is built against
mavenLocal by `make consumer-app` — the proof that the published artifact is actually consumable from
outside this repo.

> **Licence.** The project is currently **CC-BY-NC-4.0**, which does not permit commercial use. This
> is a known blocker for distributing the AAR; see [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Loading a model

`fromPretrained` is the single entry point. It resolves an installed package, or pulls and atomically
installs one if it is absent, then loads the features you ask for.

```kotlin
val model = MobileTransformers.fromPretrained(
    context = context,
    repoId  = "HuggingFaceTB/SmolLM2-135M-Instruct",
)

val result = model.generate("The capital of France is", GenerationConfig(maxNewTokens = 32))
println(result.text)

model.close()   // releases the native session; not optional
```

`close()` frees native handles. It is idempotent, but skipping it leaks a session — the library holds
**one native session at a time** and a leaked one blocks the next `train`/`generate`.

### Features

A package ships stages; you declare which you need, and a stage that is not installed fails closed
rather than degrading:

```kotlin
val model = MobileTransformers.fromPretrained(
    context, repoId,
    features = setOf(ModelFeature.Inference, ModelFeature.Training),
)
```

| feature | needs | gives you |
| --- | --- | --- |
| `Inference` | `inference/` | `generate` |
| `Training` | `train/` | `train`, `merge`, `trainingJob` |
| `Rag` | `embedding/` | `ingest`, `retrieve`, `generateWithRag` |
| `GenAI` | `inference/genai_config.json` | selects the GenAI engine (see below) |

Asking for a missing feature raises `FeatureNotInstalledException`, naming what *is* installed.

### Asking the package what it can do

`model.capabilities` is a `RuntimeCapabilities`, computed from the artifacts actually installed. It is
the honest answer to "what can I offer the user", and the sample app's whole navigation derives from
it — a UI built on it cannot claim a capability the package lacks, or withhold one it has.

| property | type | means |
| --- | --- | --- |
| `supportsTraining` / `supportsMerge` / `supportsRag` / `supportsEmbedding` | `Boolean` | the corresponding stage is installed |
| `supportsClassification` | `Boolean` | it is a classifier **and** its labels are named. See below |
| `isClassifier` | `Boolean` | the graph is a classification graph, labels or not |
| `isEncoderOnly` | `Boolean` | no generative head at all — `generate` will not work |
| `task` | `PackageTask` | the exported task, incl. `inferenceGraphPrecision` and `labelCount` |
| `graphPrecision` | `String?` | the **measured** precision of the inference graph |
| `peftMethods` / `primaryPeftMethod` | `Set<String>` / `String?` | what the package was exported with (`lora`, `lora-xs`, `mars`) |
| `trainingParameterCount` | `Long` | trainable parameters, from the training config |
| `toolCalling` / `supportsToolCalling` | `ToolCallSupport` / `Boolean` | whether the package declares a tool-call format |
| `engine` / `availableEngines` | `InferenceEngine` | resolved engine, and what could be selected |
| `supportsScheduledTraining` | `Boolean` | WorkManager-backed scheduling is usable |

`supportsClassification` is deliberately **not** `isClassifier`. A classification graph whose labels
are unknown runs fine and answers `LABEL_3`, which is a number in a costume — so
`supportsClassification` additionally requires `task.labelCount > 0`. Gate a Classify UI on it.

`graphPrecision` reports what the graph *is*, not what the variant is called. A variant named
`cpu-int4` legitimately ships an fp32 inference graph; this is the field that tells you so.

## Classification

```kotlin
if (model.capabilities.supportsClassification) {
    val result = model.classify("this sentence is grammatical", topK = 5)
    println("${result.best?.label} ${result.best?.score}")
}
```

Asking a decoder to classify **throws** rather than reading generation logits as class scores, which
would be a confident wrong answer instead of an error. `topK` bounds `result.top`; the full ranking is
always in `result.scores`.

Prefer showing the distribution over the single top label: a classifier that is 34%/33%/33% has told
you nothing, and a top label hides that completely.

## Engines

Two inference engines run over the **same** `inference/` directory:

```kotlin
MobileTransformers.fromPretrained(context, repoId, engine = InferenceEngine.NATIVE)  // default
MobileTransformers.fromPretrained(context, repoId, engine = InferenceEngine.GENAI)
```

- **`NATIVE`** — the project's own C++ decode loop over ONNX Runtime. Always available; the floor.
- **`GENAI`** — onnxruntime-genai. Requires `genai_config.json` in the package.

**Naming an engine is binding.** If you pass `InferenceEngine.GENAI` and it cannot load, you get an
`EngineUnavailableException` — the library does **not** quietly hand you Native instead. Silent
substitution is how a cross-engine parity test once compared Native with Native and passed. Pass
`engine = null` to opt into automatic selection, where falling back to Native *is* the intended
behaviour.

Check what you actually got with `model.engine`.

## Training and merging

```kotlin
model.train(
    DatasetConfig(trainFile = "my_data", task = "cola", maxSequenceLength = 64),
    TrainConfig(maxSteps = 100, batchSize = 2),
)
model.merge()                                   // folds the adapter into the inference weights
val after = model.generate(prompt, GenerationConfig(maxNewTokens = 32, loadMerged = true))
```

Two things worth knowing before you build on this:

1. **The caller supplies the data and names the preprocessor.** Packages ship model artifacts, not
   training sets. `DatasetConfig.trainFile` resolves to `<cacheDir>/<repo>/train/<trainFile>.jsonl`
   and `task` selects the parser that reads it.
2. **`merge()` rewrites the package in place.** The per-tensor `.bin` files under `inference/` are
   overwritten with merged weights. This is deliberate — it is what makes the merged model loadable by
   the ordinary path — but it means a package is no longer pristine afterwards. Re-install it if you
   need the original weights back.

On-device training can only re-run within the PEFT topology the package was exported with; asking for a
different method raises `PeftMismatchException`.

## RAG

```kotlin
model.ingest(path = "/sdcard/…/notes.md", config = RagConfig())
val hits   = model.retrieve("query", RagConfig())          // search alone, nothing generated
val answer = model.generateWithRag("query", RagConfig(), GenerationConfig(maxNewTokens = 64))
```

`retrieve` is a first-class operation, not only a step inside `generateWithRag`. It needs
`supportsEmbedding` and nothing else, so it works on a **pure encoder** package that cannot generate at
all. It is also the only way to judge retrieval on its own: inside a grounded answer, bad retrieval and
a model ignoring good retrieval are indistinguishable.

`generateWithRag` returns a `GroundedResult` whose `prompt` is the text that was actually assembled —
without it a bad grounded answer is undebuggable.

Backed by ObjectBox HNSW with cosine distance. The encoder's output dimension must be one the on-device
store can index (64/128/256/384/512/768/1024/1536) — the exporter fails closed rather than shipping an
unusable `embedding/` stage. See [RAG.md](RAG.md).

## Generation results

`GenerationResult` carries more than `text`:

| field | means |
| --- | --- |
| `text` | the generated continuation |
| `promptTokenCount` | how many tokens the prompt consumed |
| `contextLimit` | the model's context window |

The pair is what lets a UI say "you have used 400 of 2048 tokens" *before* generation truncates
something. Read them rather than estimating from character counts.

## Tool calls

A package that declares a tool-call format (`capabilities.supportsToolCalling`) can emit a structured
call instead of prose. The result is a `ToolCallResult`:

- `ToolCallResult.NoCall` — the model answered normally. **This is the common case**, and it is a
  distinct type rather than a null so a caller cannot forget to handle it.
- a parsed call — validated against your `ActionSpec` allowlist before anything runs.

`ActionSpec.requiredPermissions` and `IntendedAction.requiredPermissions` declare what an action needs.
Today every showcased action is install-time, because intent-based actions delegate the sensitive work
to the target app, which enforces its own permissions behind its own UI. The runtime-request path
exists for when an action needs one.

Nothing executes without an explicit accept. That is the design, not a sample-app convention.

## Errors

Everything the library raises descends from `MobileTransformersException`, and every failure path is
fail-closed — there are no silent fallbacks that leave you with a working-looking object doing something
other than what you asked.

| exception | means |
| --- | --- |
| `ModelNotInstalledException` | package absent and no pull configured |
| `MissingArtifactException` | a required file is missing from the installed package |
| `FeatureNotInstalledException` | you asked for a stage the package does not carry |
| `EngineUnavailableException` | the engine you **named** could not load |
| `PeftMismatchException` | requested PEFT differs from the exported topology |

## Memory

A 135M int4 package peaks around **800 MB RSS** during generation on a Galaxy S21 FE, on either engine.
Budget for the package on disk *and* that resident peak.

An opt-in `mmap` path zero-copies the trainable tensors instead of reading them into allocator buffers.
It is **off by default** and currently covers only the trainable split (~8% of weight bytes), which
measured a ~6% peak reduction — real, but below the 15% the memory gate targets. Treat it as an
optimisation, not a memory strategy.
