# Public API

The SemVer-governed public surface (F5) has three peers: the **Python library API**, the **CLI**, and the
**Kotlin facade**. The Python side is exactly the surface declared in `mobiletransformers.__all__`
(owned by `00_code_plans/10`, guarded by a parity test against `src/mobiletransformers/public_api.txt`).

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
| `agent-dataset` | Build the #37 tool-call training set + action schema (import a corpus, or synthesise per-user). |

Run `mobiletransformers <command> --help` for flags. `make help` lists the wrapper targets
(`export-model`, `package-model`, `android-build`, …).

## Kotlin facade (`com.martinkorelic.mobiletransformers`)

Obtained from `MobileTransformers.fromPretrained(context, repoId, …)`, which pulls and installs the
package when it is not already in the cache.

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `MobileTransformers.fromPretrained` | entry point | resolve → (pull) → load; returns a `MobileTransformerModel`. |
| `MobileTransformerModel` | handle | `train`/`trainingJob`/`merge`/`generate`/`retrieve`/`ingest`/`generateWithRag`/`applyPeft`/`pushAdapter`/`close`. |
| `TrainingJob` | lifecycle | `status`/`events` flows, cooperative `cancel`, `checkpoint()`/`canResume`. |
| `RuntimeCapabilities`, `EngineCapabilities` | capability | installed features, resolved engine, merged-weight support. |
| `InferenceEngine` | enum | `NATIVE` (the floor) \| `GENAI`. |
| `TrainConfig`, `GenerationConfig`, `RagConfig`, `DatasetConfig`, `PeftConfig`, `HubConfig`, `DeviceConfig` | config | public configs; mapped to the internal `ORT*Config` types. |
| `TrainingResult`, `TrainingSummary`, `GenerationResult`, `MergeResult`, `RetrievalResult`, `GroundedResult`, `IngestResult`, `PushResult` | results | plain data; no `ORT*`/`*Native` type appears on this surface. |
| `TrainCallback`, `GenerateCallback`, `RetrieveCallback` | callbacks | streaming progress. |
| `MobileTransformersException` | errors | base of the hierarchy (`ModelNotInstalledException`, `MissingArtifactException`, `PeftMismatchException`, `FeatureNotInstalledException`, `EngineUnavailableException`, `NotImplementedFeatureException`). Deliberately `open`, not `sealed`: subclasses live in sibling packages (e.g. `hub.AdapterUploadDisabledException`). |
| `constants/*` | enums | wire-value mirrors of the Python enums, parity-checked by `make parity`. |

Internal packages (`repository`, `internal.*`, `ORT*`/`*Native`) are **not** public and may change.

## Stability

The three surfaces above are the public contract; internal modules (`export.pipeline`, `hub.*`,
`artifacts.*`, `support.*`, `adapter.*`) may change between releases. The full surface is finalized and
version-locked at #32.
