# Configuration

MobileTransformers configuration is a **typed, closed-set** contract. Every closed string set is an enum
mirrored 1:1 between Python and Kotlin; every cross-boundary config is a Pydantic v2 model with a generated
JSON schema; and every extension point (PEFT method, architecture, merger) is a registry entry. This page is
sourced from the config contract owner (`src/mobiletransformers/config/`, #6).

## Enum vocabulary (Python ↔ Kotlin mirrors)

The enums in `config/constants.py` are the single Python source of truth for every closed string set. Each
has a hand-written Kotlin `enum class` mirror, and `python -m mobiletransformers.codegen.enums --check` is the
CI parity gate that fails on drift. The **wire value** (right column) is the on-disk / JSON string.

| Enum | Values (wire) |
| --- | --- |
| `SamplingMethod` | `greedy`, `top_k`, `top_p` |
| `SchedulerType` | `linear`, `cosine` |
| `ExecutionProvider` | `cpu`, `xnnpack`, `nnapi` |
| `CoreConfigId` | `opt1`, `opt2`, `opt3` |
| `MemoryConfigId` | `low_mem`, `high_perf` |
| `SearchType` | `semantic`, `text` |
| `QuantizationType` | `QInt8`, `QUInt8`, `int4` |
| `PEFTMethod` | `lora`, `lora-xs`, `mars`, `all`, `nolora` |
| `TaskType` | `text-generation`, `feature-extraction` |
| `HandoffMode` | `external_initializer` (v1), `model_input`, `adapter` |
| `MergerVariant` | `lora`, `lora_q`, `mars_q` (device-resolved, not a user choice) |

`ExportFrontend` (`optimum-onnx`, `torch.onnx`) is build-time/Python-only — Android never sees it, so it has
no Kotlin mirror and no parity obligation.

## Cross-boundary config models

The Pydantic v2 models in `config/models.py` define the three on-disk configs the device reads
(`training_config.json`, `generation_config.json`, `rag_config.json`). They share a base with
`populate_by_name=True`, `extra="ignore"`, `use_enum_values=True`:

- **`extra="ignore"`** (not `forbid`) — readers tolerate unknown fields so additive minor schema bumps stay
  non-breaking.
- Field names are snake_case in Python; the **wire/JSON name is the camelCase `alias`** (e.g. `maxSequenceLength`).
- Cross-boundary models carry `schemaVersion` / `minReaderVersion` and fail closed on an unsupported major.

`GenerationConfig` (`generation_config.json`):

| Field (wire) | Default | Type |
| --- | --- | --- |
| `maxSequenceLength` | `128` | int |
| `sampling` | `SamplingConfig()` | `{ method, temperature, topK, topP, seed }` |
| `deviceOptions` | `DeviceOptions()` | `{ enableProfiling, coreConfigId, memoryConfigId, executionProvider }` |

`TrainingConfig` (`training_config.json`):

| Field (wire) | Default | Type |
| --- | --- | --- |
| `peftMethod` | `lora` | `PEFTMethod` |
| `rank` | `8` | int |
| `alpha` | `16` | int |
| `maxSteps` | `10` | int |
| `scheduler` | `LinearScheduler()` | discriminated on `schedulerType` (`linear` → start/end factor; `cosine` → minLearningRate/warmupSteps) |
| `quantization` | `QuantizationOptions()` | weight type + symmetry/subgraph flags |

`RagConfig` (`rag_config.json`):

| Field (wire) | Default | Type |
| --- | --- | --- |
| `searchType` | `semantic` | `SearchType` |
| `topK` | `5` | int |
| `embeddingDim` | `384` | int |

The set of models that emit a checked-in schema is `CROSS_BOUNDARY_MODELS`; `schemas/*.schema.json` are
regenerated from these by the codegen module and validated for drift in CI. Devices enforce the contract by
**typed fail-closed parsing**, not runtime schema validation.

## Registries — the public extension points

Adding support for a new PEFT method, model architecture, or merger is a **registry entry**, not new dispatch
code. The registries live in `config/registry/`:

- **`peft.py` — `PEFT_REGISTRY`** (`PEFTMethod` → `PEFTMethodSpec`). A spec declares the `config_class` (lazy
  dotted path), the `component_schema` (ordered `AdapterComponent`s — the **source of truth for tensor naming**
  consumed by `TrainableTensorCodec`), and the fp/quantized merger variants. To add a method: add a
  `PEFTMethod` enum member + one `PEFT_REGISTRY` row.
- **`architecture.py` — `ARCHITECTURE_REGISTRY`**. Maps a model architecture to its export config and the
  attention-module naming used by the weight-handoff name rewrite. To add an architecture: add a registry
  entry (no KV-cache-specific dispatch to touch).
- **`merger.py` — `MERGER_REGISTRY`** + `resolve_merger`/`build_merger_model`. Maps a resolved `MergerVariant`
  to the ONNX-graph merger builder. The variant is **derived** on device from adapter shape + quantization,
  not chosen by the user.

Lookups fail closed: an unknown method/architecture/merger raises a typed error naming the offender rather
than silently falling back.

## Precedence

Runtime settings resolve with the precedence **CLI > environment > YAML (`config/config.yml`) > model
default** (`config/settings.py` / `resolve()`). Secrets never live in `constants.py`; they belong in
`config/settings.py`.
