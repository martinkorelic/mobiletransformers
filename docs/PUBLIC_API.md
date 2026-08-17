# Public API

The SemVer-governed public surface (F5) has three peers: the **Python library API**, the **CLI**, and the
**Kotlin facade**. The Python side is exactly the surface declared in `mobiletransformers.__all__`
(guarded by a parity test against `src/mobiletransformers/public_api.txt`).

All three surfaces are documented below. See also [EXPORT.md](EXPORT.md), [MODEL_FORMAT.md](MODEL_FORMAT.md),
[HUB_PACKAGE_FORMAT.md](HUB_PACKAGE_FORMAT.md), [ARCHITECTURE.md](ARCHITECTURE.md) and [RAG.md](RAG.md).

## Python (`import mobiletransformers`)

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `__version__` | str | Package version. |
| `resolve` | func | Config resolution with precedence CLI > env > YAML > package default. |
| `get_settings` | func | Load secrets/settings (secrets live only in `Settings`, never in YAML). |
| `Settings` | class | Typed settings/secrets container. |
| `get_logger` | func | Library logger (`NullHandler`; no `print` in library code). |
| `configure_logging` | func | Opt-in logging configuration for applications. |
| `MobileTransformersError` | exc | Base of the exception hierarchy. |
| `ConfigValidationError` | exc | Invalid configuration. |
| `ExportError` | exc | Export/packaging failure. |
| `ManifestError` | exc | Manifest parse/validation failure. |
| `HandoffError` | exc | Weight-handoff-map failure. |
| `MergeError` | exc | Merge failure. |
| `HubError` | exc | Hub pull/push failure. |
| `UnsupportedModelError` | exc | Unsupported architecture/model. |

The exception names mirror the Kotlin hierarchy. This list is authoritative — it is regenerated/guarded
against `public_api.txt`, so any addition is a deliberate SemVer change.

## CLI (`mobiletransformers <command>`)

| Command | Purpose |
| --- | --- |
| `export` | HF model → device-ready package (`--dry-run`, `--config`, `--validate` supported). |
| `validate` | Validate a written package (`--package`) and/or a config YAML (`--config`). |
| `package-model` | Re-hash an existing package and re-emit its manifest + checksums (`--package`). |
| `push` | Validate + publish a package to the Hub. |
| `pull` | Download a package (manifest-first, sha256-verified). |
| `install-package` | Materialize a pulled package into the SDK cache layout. |
| `support-matrix` | Generate `model_support_matrix.json` (+ `--docs`, `--md`). |
| `push-adapter` | Publish a trained adapter (PEFT Mode 1 / native Mode 2). |
| `federated` | `federated simulate` — FedAvg simulation over codec-ordered adapter records. |
| `agent-dataset` | Build the tool-call training set + action schema (import a corpus, or synthesise per-user). |

Run `mobiletransformers <command> --help` for flags. `make help` lists the wrapper targets
(`export-model`, `package-model`, `android-build`, …).

## Kotlin facade (`com.martinkorelic.mobiletransformers`)

Obtained from `MobileTransformers.fromPretrained(context, repoId, …)`, which pulls and installs the
package when it is not already in the cache.

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `MobileTransformers.fromPretrained` | entry point | resolve → (pull) → load; returns a `MobileTransformerModel`. |
| `MobileTransformerModel` | handle | `train`/`trainingJob`/`merge`/`generate`/`retrieve`/`ingest`/`generateWithRag`/`classify`/`applyPeft`/`pushAdapter`/`close`. |
| `TrainingJob` | lifecycle | `status`/`events` flows, cooperative `cancel`, `checkpoint()`/`canResume`. |
| `RuntimeCapabilities`, `EngineCapabilities` | capability | installed features, resolved engine, merged-weight support. Also `supportsClassification`, `isEncoderOnly`, `graphPrecision`, `peftMethods`, `trainingParameterCount`, `toolCalling`. |
| `PackageTask` | capability | the exported task; carries `inferenceGraphPrecision` (the **measured** precision, which a variant name may not match) and `labelCount`. |
| `InferenceEngine` | enum | `NATIVE` (the floor) \| `GENAI`. |
| `TrainConfig`, `GenerationConfig`, `RagConfig`, `DatasetConfig`, `PeftConfig`, `HubConfig`, `DeviceConfig` | config | public configs; mapped to the internal `ORT*Config` types. |
| `TrainingScheduleConfig` | config | WorkManager-backed scheduling. `initialDelayMinutes` is a floor, not an appointment — an exact start needs `SCHEDULE_EXACT_ALARM`, which Play restricts. |
| `TrainingResult`, `TrainingSummary`, `GenerationResult`, `MergeResult`, `RetrievalResult`, `GroundedResult`, `IngestResult`, `PushResult` | results | plain data; no `ORT*`/`*Native` type appears on this surface. `GenerationResult` also carries `promptTokenCount` and `contextLimit`; `GroundedResult` carries the assembled `prompt`. |
| `ClassificationResult` | results | `scores` (full ranking), `top` (bounded by `topK`), `best`. |
| `ToolCallResult`, `ToolCallSupport` | tool calls | `ToolCallResult.NoCall` is the common case and a distinct type, so a caller cannot forget to handle "the model just answered". |
| `ActionSpec`, `IntendedAction` | tool calls | the allowlist a parsed call is validated against, and the bound action. Both declare `requiredPermissions`. |
| `TrainCallback`, `GenerateCallback`, `RetrieveCallback` | callbacks | streaming progress. |
| `MobileTransformersException` | errors | base of the hierarchy (`ModelNotInstalledException`, `MissingArtifactException`, `PeftMismatchException`, `FeatureNotInstalledException`, `EngineUnavailableException`, `NotImplementedFeatureException`). Deliberately `open`, not `sealed`: subclasses live in sibling packages (e.g. `hub.AdapterUploadDisabledException`). |
| `constants/*` | enums | wire-value mirrors of the Python enums, parity-checked by `make parity`. |

Internal packages (`repository`, `internal.*`, `ORT*`/`*Native`) are **not** public and may change.

## Stability

The three surfaces above are the public contract; internal modules (`export.pipeline`, `hub.*`,
`artifacts.*`, `support.*`, `adapter.*`) may change between releases. The full surface is finalized and
version-locked at the v1.0 release.
