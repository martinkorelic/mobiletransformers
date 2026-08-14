# Cookbook

Copy-pasteable Kotlin for each thing the SDK does. Every snippet uses **only the public facade** —
`MobileTransformers`, `MobileTransformerModel`, and the `config/` types. No `ORT*`, `*Native` or
`*Repository` type appears here, and none should appear in your app either.

Each recipe mirrors a screen in the sample app (`MobileTransformersApp`), so the documentation and the
worked example cannot describe different APIs. The screen is named under each heading.

> **Device reality.** arm64-v8a only; there is no x86_64 emulator build. Most calls below do nothing
> until a package is installed, so start with the first recipe.

---

## 1. Pull a model from the Hub, with progress

*Sample app: Models screen.*

`fromPretrained` installs the package when it is missing, then loads it. Download progress is reported
through `DownloadProgress`; `fraction` is `null` until the download plan is known, which is the honest
state for "total not yet resolved".

```kotlin
val model = MobileTransformers.fromPretrained(
    context = context,
    repoId = "HuggingFaceTB/SmolLM2-135M-Instruct",
    features = setOf(ModelFeature.Inference),
    onDownloadProgress = { progress ->
        Log.i("pull", "${progress.filesDone}/${progress.filesTotal} ${progress.path}")
    },
)
```

Ask for a feature only if you need it. Requesting one the package does not ship fails closed with
`FeatureNotInstalledException` at construction rather than at first use.

### What is already installed?

```kotlin
MobileTransformers.installed(context).forEach { pkg ->
    Log.i("cache", "${pkg.sanitizedRepoId} ${pkg.sizeBytes / (1024 * 1024)} MB ${pkg.variantIds}")
}
```

---

## 2. Generate, with streaming

*Sample app: Chat screen.*

```kotlin
val result = model.generate(
    prompt = "The capital of France is",
    config = GenerationConfig(
        maxNewTokens = 64,
        sampling = SamplingConfig(method = SamplingMethod.GREEDY),
    ),
    callback = object : GenerateCallback {
        override fun onPartialResult(progress: GenerateProgress) {
            print(progress.token)          // token-by-token, in order
        }
    },
)
Log.i("gen", "${result.tokenCount} tokens at ${result.avgTokensPerSecond} tok/s")
```

---

## 3. Choose an engine

*Sample app: Chat screen, engine picker.*

`Native` is the guaranteed floor. `GenAI` is selectable only when the installed package ships
`inference/genai_config.json` **and** the device's GenAI probe succeeds — ask, rather than guessing:

```kotlin
if (InferenceEngine.GENAI in model.capabilities.availableEngines) {
    // reload with the other engine; the engine is fixed at load time
    val genai = MobileTransformers.fromPretrained(
        context = context,
        repoId = repoId,
        engine = InferenceEngine.GENAI,
    )
}
```

Naming an engine you cannot have raises `EngineUnavailableException` instead of quietly giving you
Native. Silent substitution is what made an earlier engine-parity test compare Native with Native and
pass.

---

## 4. Fine-tune on device, then merge

*Sample app: Train screen.*

Use `trainingJob()` when you want status, events, cancellation or resume; `train()` is the one-shot
convenience.

```kotlin
val job = model.trainingJob()

launch { job.status.collect { status -> Log.i("train", status.toString()) } }

job.start(
    dataset = DatasetConfig(trainFile = "my_data", task = "mobile_actions", maxSequenceLength = 160),
    config = TrainConfig(
        maxSteps = 120,
        batchSize = 2,
        learningRate = 5e-4f,
        // The optimizer steps on `globalStep % gradientAccumulationSteps == 0`. At the default of 4
        // a short bounded run can finish, report success, and apply no update at all.
        gradientAccumulationSteps = 1,
        mergeAtEnd = true,
    ),
)
```

Two things worth knowing before you size a run:

* **`maxSteps` is an upper bound.** Training also stops at the end of the epoch, so `rows / batchSize`
  wins when it is smaller — a run asking for 120 steps over 108 rows at batch 2 takes 54.
* **Cancelling is resumable.** `job.cancel(saveCheckpoint = true)` sets a cooperative flag; the loop
  breaks at the next step boundary and writes a checkpoint, and `job.canResume` then reads `true`.

### Train while charging

```kotlin
TrainingScheduler.schedule(
    context = context,
    repoId = model.repoId,
    dataset = DatasetConfig(trainFile = "my_data", task = "cola"),
    training = TrainConfig(maxSteps = 500),
    config = TrainingScheduleConfig(),
)
```

Each chunk re-enters the WorkManager queue, so unplugging pauses the run rather than failing it.

---

## 5. Ground answers in your own documents

*Sample app: Chat screen, RAG toggle.*

```kotlin
model.ingest(path = "/sdcard/Download/notes.md", config = RagConfig())

val grounded = model.generateWithRag(
    query = "what did I write about batching?",
    rag = RagConfig(topK = 5, minScore = 0.2),
    generation = GenerationConfig(maxNewTokens = 200),
    promptStrategy = PromptAssembler.DEFAULT,
)

Log.i("rag", grounded.text)
grounded.matches.forEach { Log.i("rag", "${it.score} ${it.text}") }
Log.i("rag", "asked: ${grounded.prompt}")   // the assembled prompt, for when the answer is wrong
```

Leave the embedding identity unset unless you mean to override the package: it is read from
`embedding/rag_config.json`, written by the exporter from the encoder it actually shipped.

---

## 6. Tool calls: instruction → validated call → dry-run intent

*Sample app: Tool calls screen.*

Your app declares the actions. **A model selects an action; it cannot name an intent** — the intent
string comes from your `ActionSpec` — so the reachable set of intents is fixed when you write this list.

```kotlin
val validator = FunctionCallValidator(
    listOf(
        ActionSpec(
            actionName = "set_alarm",
            parameters = mapOf("time" to "string"),
            allowedIntent = "android.intent.action.SET_ALARM",
            validationRules = mapOf("time" to "HH:mm"),
        ),
    ),
)

when (val result = model.generateToolCall("wake me at 07:30", validator)) {
    is ToolCallResult.Accepted -> {
        val intended = result.dryRun()
        Log.i("tool", "${intended.intent.action} willExecute=${intended.willExecute}")  // false
    }
    is ToolCallResult.Rejected -> Log.i("tool", "refused: ${result.reason}")
}
```

`Rejected` is a value, not an exception: refusing untrusted output is the expected path. Nothing here
executes — `IntentBinder` holds no `Context` and has no `startActivity` call site, so firing the intent
is your deliberate act with your own `Context`.

Build the training set from the same declaration so the corpus and the boundary are provably one value:

```bash
mobiletransformers agent-dataset --source generated \
  --allowlist build/agent/action_schema.json --output build/user
```

> **Status, 2026-08-14.** The on-device gate for this recipe (`ToolCallDeviceTest`) **fails**: after a
> converging fine-tune the model emits a repeated newline rather than a call. The cause is under
> investigation and is recorded against #37 — it is not convergence, dataset size or prompt format,
> all of which were measured and excluded. Expect `Rejected` until it is fixed.

---

## 7. One federated round

*Sample app: Federated screen.*

Federation is **off by default** (`BuildConfig.FEDERATION_ENABLED`). The round returns bytes and accepts
bytes; the transport is yours, which is what lets the whole loop run against a local `federated serve`.

```kotlin
val result = model.federatedRound(
    config = FederatedConfig(
        gatewayUrl = "https://gateway.example",
        clientAuthToken = token,
        consent = FederatedConsent(
            granted = true,
            policyVersion = "1.0",
            grantedAtEpochMs = System.currentTimeMillis(),
        ),
    ),
    globalRecord = previousAggregate,   // null for round 0
    roundNumber = 1,
    localTraining = { round -> model.train(dataset, TrainConfig(maxSteps = 20)) },
)

upload(result.update)                    // result.payloadBytes is what federation costs per round
```

Consent, TLS and auth are checked before any tensor is read, and the refusal names the missing
protection. Only adapter factors and aggregate metrics ever leave the device — never your examples.

---

## 8. Close it

```kotlin
model.close()
```

A model owns native sessions. Load once and share the handle; loading the same package twice opens two
sessions over one set of weights.
