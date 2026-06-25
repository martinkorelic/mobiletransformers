# Typed Config Models, Enums & Method/Architecture/Merger Registries

**Priority (global #):** 6  |  **Prerequisites:** `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md` (#2, adds `pydantic>=2` to core), `00_code_plans/02_config_layering_settings_constants.md` (#4, `config/` package + `constants.py`)  |  **Blocks:** `00_code_plans/07_weight_handoff_map_and_tensor_codec.md` (#8), `01_code_plans/01_unified_merger_and_external_data_export.md` (#9), `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`, and every builder-consuming Tier-2/3 plan (`03_code_plans/*`, `04_code_plans/01`, `/05`)

> This plan **OWNS** four cross-cutting contracts: (1) the Pydantic config models, (2) the Python+Kotlin enum/constants vocabulary, (3) the **PEFT method registry** + **architecture registry**, and (4) the **merger registry** + unified `build_merger_model`. Every other plan *references* these and must not redefine them. The rule this plan enforces everywhere: **a closed set of choices is data (enum + registry entry), never an `if/elif` chain in business logic.** Adding a PEFT method, model architecture, or merger variant is a registry entry plus an enum member — no new branch.

---

## Purpose

The codebase dispatches behavior through ~83 hard-coded `if/elif` chains keyed on string literals, and carries no typed config layer. Three structural problems result:

1. **Dispatch drift & non-extensibility.** Adding a model/PEFT/merger means editing parallel `if/elif` chains in Python *and* C++ that must agree but have no shared source of truth:
   - Merger selection: `artifact/onnx_builder.py:628-641` (`if peft_method == "lora" … elif "mars" … else raise "Unsupported PEFT method."`).
   - Merger variants as four near-duplicate factories with `_2`-suffixed siblings: `create_lora_merger_model` (`artifact/merger.py:240`), `create_lora_merger_model_2` (`:351`), `create_mars_merger_model_2` (`:599`), `create_mars_merger_model` (`:824`).
   - C++ merger dispatch: `WeightMerger::get_merger_type` returns the literals `"mars_q"`/`"lora_q"`/`"lora"` (`weight_merger.cpp:476-499`); `run_merger_model` branches on them at `:526, :562, :613, :687, :707`; merger ONNX paths are hard-coded at `:448-464` (`/lora_merger_model.onnx`, `/lora_qmerger_model.onnx`, `/mars_qmerger_model.onnx`).
   - PEFT config + mapping dispatch: `trainer/builder.py:282-346` (`if train_method == "lora" … elif "lora-xs" … elif "mars" … elif "all" … elif "nolora"`, then `if train_method == "mars": create_mars_adapter_mapping … elif "lora": create_lora_mapping`, `:343-346`).
   - Architecture dispatch: training export `trainer/builder.py:260-272` (`# TODO: Add support for other architectures`, then `if architectures[0] == "LlamaForCausalLM" … elif Gemma/Gemma2/Gemma3 … elif Phi3 … elif Qwen2 … elif OPT … elif BertModel`); inference graph `inference/builder.py:3234-3271` (13 branches, several gated on `config.max_position_embeddings`).
   - Duplicated target-module tables: `TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING` (`peft_models/mars/utils.py:1`) ≡ `TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING` (`peft_models/ablation/utils.py:1`).

2. **No typed config / no cross-boundary contract.** Config is raw `yaml.safe_load` dicts with string-key access (`config.yml` → `trainer/builder.py`, `inference/builder.py`, `artifact/onnx_builder.py`). The export side writes `training_config.json` / `generation_config.json` / `rag_config.json` as free dicts (e.g. `trainer/builder.py:363-372`); Kotlin reads them with defensive `.optString(...)`/`.optInt(...)` fallbacks (`FileUtil.kt`). There is no schema enforced at the boundary, so a Python rename silently breaks Kotlin parsing at runtime.

3. **Magic strings & code-only config.** The same closed sets are re-spelled across languages: `methodMap = mapOf("greedy" to 0, "top_k" to 1, "top_p" to 2)` (`ORTGeneratorNative.kt:266`), `listOf("greedy", …)` / `listOf("linear", "cosine")` / `listOf("opt1", …)` (`ConfigurationScreen.kt`), `when (schedulerType) { "linear" -> … "cosine" -> … }` (`FileUtil.kt`). Behavior tunables live only in code and are invisible to operators (quantization `extra_options` at `trainer/builder.py:447-462`; reconstruction config at `peft_models/lora_xs/merger.py:34,42`; session options at `trainer/validator.py:402`; `large_model=True` at `artifact/onnx_builder.py:551`).

This plan replaces all of that with: one typed config layer (Pydantic v2), one mirrored enum vocabulary (Python ↔ Kotlin), and three data-driven registries (PEFT, architecture, merger) that are the single source of truth for every closed choice.

---

## Touched / new files

**Python — new (the owned contracts):**
- `config/models.py` — Pydantic v2 models for every config object (A1).
- `config/registry/__init__.py`, `config/registry/peft.py`, `config/registry/architecture.py`, `config/registry/merger.py` — the three registries (A3/A4/A5). (Co-locate under `config/` so they sit beside `settings.py`/`constants.py` from `02`; `peft_models/registry.py` is an acceptable alternative location if PEFT specs need to import `peft_models` config classes — pick one and reference it everywhere.)
- `schemas/*.schema.json` — generated JSON Schema per cross-boundary model (A1).
- `tests/unit/test_config_models.py`, `tests/unit/test_registries.py`, `tests/unit/test_enum_parity.py`.

**Python — edit (consume, do not redefine):**
- `config/constants.py` (from `02`) — add the enums (A2); replace the `SUPPORTED_PEFT_METHODS` tuple with the `PEFTMethod` enum.
- `trainer/builder.py` — `:260-272` arch dispatch → `architecture_registry`; `:282-346` PEFT config + mapping dispatch → `peft_registry`; `:363-372` JSON emit → `TrainingConfig.model_dump(by_alias=True)`; `:447-462` quantization `extra_options` → typed `QuantizationOptions`.
- `inference/builder.py` — `:3234-3271` arch dispatch → `architecture_registry`.
- `artifact/onnx_builder.py` — `:628-641` merger dispatch → `merger_registry` + `build_merger_model`; remove the `_2`-factory imports at `:31`; `:551` `large_model` → config field; `:205` hard-coded dequantize match documented as registry/handoff-driven.
- `artifact/merger.py` — collapse the four `create_*_merger_model{,_2}` factories (`:240, :351, :599, :824`) into one parameterized `build_merger_model` (A5).
- `trainer/utils.py` — `create_mars_adapter_mapping` (`:533-668`) / `create_lora_mapping` (`:670-703`) re-expressed as one mapping builder driven by the PEFT spec's component schema (A3).
- `peft_models/mars/utils.py` + `peft_models/ablation/utils.py` — collapse the duplicate target-module tables into the architecture registry (A4).
- `peft_models/lora_xs/merger.py` — `:34,42` `# TODO: Hardcoded` reconstruction config → `MergerSpec`/`QuantizationOptions` fields.
- `trainer/validator.py` — `:402` `# TODO: Customize` session options → typed `SessionOptions` model.

**Kotlin — new + edit (mirror the vocabulary):**
- NEW `android/ORTransformer/ORTransformersMobile/src/main/java/com/martinkorelic/ortmobile/constants/*.kt` — `enum class` mirrors of every Python enum (A2), values byte-identical to the `model_dump(by_alias=True)` strings.
- `ORTGenerationConfig.kt`, `ORTTrainingConfig.kt`, `ORTRagConfig.kt` — typed enum fields replace `String` fields where the set is closed (`SamplingMethod`, `SchedulerType`, `ExecutionProvider`, `CoreConfigId`, `MemoryConfigId`, `SearchType`).
- `FileUtil.kt` — `when (schedulerType)` / `.optString(...)` parsing validates against the enum + the checked-in JSON Schema (fail-closed on unknown value).
- `ORTGeneratorNative.kt:266` `methodMap` → `SamplingMethod.ordinalForNative()` (or an explicit enum→Int map owned by the enum).
- `app/.../ConfigurationScreen.kt` — `listOf("greedy", …)` etc. → `SamplingMethod.entries` (and siblings).

---

## Data contracts / interfaces

### A1 — Pydantic config models (`config/models.py`)

Pydantic v2, alias-driven so the on-disk JSON keeps the existing camelCase the Kotlin side already reads (no on-device migration needed). The scheduler union closes the `cosineLearningRate`-vs-`learningRate` mismatch documented in the config sweep.

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Union, Annotated
from config.constants import SamplingMethod, SchedulerType, ExecutionProvider, \
    CoreConfigId, MemoryConfigId, SearchType, QuantizationType

class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", use_enum_values=True)

class SamplingConfig(_Base):
    method: SamplingMethod = SamplingMethod.GREEDY
    temperature: float = 1.0
    top_k: int = Field(10, alias="topK")
    top_p: float = Field(0.9, alias="topP")
    seed: int = 42

class DeviceOptions(_Base):
    enable_profiling: bool = Field(False, alias="enableProfiling")
    core_config_id: CoreConfigId = Field(CoreConfigId.OPT1, alias="coreConfigId")
    memory_config_id: MemoryConfigId = Field(MemoryConfigId.HIGH_PERF, alias="memoryConfigId")
    execution_provider: ExecutionProvider = Field(ExecutionProvider.CPU, alias="executionProvider")

class LinearScheduler(_Base):
    scheduler_type: Literal[SchedulerType.LINEAR] = Field(SchedulerType.LINEAR, alias="schedulerType")
    learning_rate: float = Field(1e-4, alias="learningRate")
    start_factor: float = Field(1.0, alias="startFactor")
    end_factor: float = Field(0.333, alias="endFactor")

class CosineScheduler(_Base):
    scheduler_type: Literal[SchedulerType.COSINE] = Field(SchedulerType.COSINE, alias="schedulerType")
    learning_rate: float = Field(1e-4, alias="learningRate")
    min_learning_rate: float = Field(0.0, alias="minLearningRate")
    warmup_steps: int = Field(10, alias="warmupSteps")

SchedulerConfig = Annotated[Union[LinearScheduler, CosineScheduler],
                            Field(discriminator="scheduler_type")]

class GenerationConfig(_Base):
    max_sequence_length: int = Field(128, alias="maxSequenceLength")  # HF-aligned rename tracked in 03_code_plans/02
    sampling: SamplingConfig = SamplingConfig()
    device_options: DeviceOptions = Field(default_factory=DeviceOptions, alias="deviceOptions")

class QuantizationOptions(_Base):                       # lifts trainer/builder.py:447-462 into config
    weight_type: QuantizationType = Field(QuantizationType.QINT8, alias="weightType")
    activation_symmetric: bool = Field(False, alias="activationSymmetric")
    weight_symmetric: bool = Field(False, alias="weightSymmetric")
    enable_subgraph: bool = Field(False, alias="enableSubgraph")
    force_quantize_no_input_check: bool = Field(True, alias="forceQuantizeNoInputCheck")
    matmul_const_b_only: bool = Field(True, alias="matMulConstBOnly")
# TrainingConfig, RagConfig, TrainBuilder, InferenceBuilder, ArtifactBuilder follow the same pattern.
```

Cross-boundary rule: export writes `Model.model_dump(by_alias=True, mode="json")`; `schemas/<name>.schema.json` is produced by `Model.model_json_schema(by_alias=True)` and checked in; Kotlin/C++ validate the file they read against that schema before parsing. The Python and Kotlin field vocabularies are tested for parity (see Tests).

**Parity enforcement (F2).** The Pydantic models (`config/models.py`) + enums (`config/constants.py`) are the **single source of truth**; the checked-in `schemas/*.schema.json` and a golden `enums.json` are *generated* from them, never hand-edited. A CI `parity` job regenerates and diffs: `python -m mobiletransformers.codegen.enums --check` fails the build if the checked-in Kotlin/C++ mirrors have drifted from the Python source. This is the mechanism behind the enum-parity and grep guards below.

**Schema versioning (F1).** Each generated cross-boundary schema also carries a `schemaVersion` (`"MAJOR.MINOR"`) + `minReaderVersion` field block — the same contract the manifest (`#13`) and handoff map (`#8`) use — so a reader fails closed on a major it does not support and tolerates additive minor bumps (unknown fields preserved). One `check_compat()` helper, mirrored Python↔Kotlin.

### A2 — Enum vocabulary (`config/constants.py`, mirrored in Kotlin)

```python
from enum import Enum
class SamplingMethod(str, Enum): GREEDY="greedy"; TOP_K="top_k"; TOP_P="top_p"
class SchedulerType(str, Enum): LINEAR="linear"; COSINE="cosine"
class ExecutionProvider(str, Enum): CPU="cpu"; XNNPACK="xnnpack"; NNAPI="nnapi"
class CoreConfigId(str, Enum): OPT1="opt1"; OPT2="opt2"; OPT3="opt3"
class MemoryConfigId(str, Enum): LOW_MEM="low_mem"; HIGH_PERF="high_perf"
class SearchType(str, Enum): SEMANTIC="semantic"; TEXT="text"
class QuantizationType(str, Enum): QINT8="QInt8"; QUINT8="QUInt8"; INT4="int4"
class PEFTMethod(str, Enum): LORA="lora"; LORA_XS="lora-xs"; MARS="mars"; ALL="all"; NOLORA="nolora"
class TaskType(str, Enum): TEXT_GENERATION="text-generation"; FEATURE_EXTRACTION="feature-extraction"
class HandoffMode(str, Enum): EXTERNAL_INITIALIZER="external_initializer"; MODEL_INPUT="model_input"; ADAPTER="adapter"
class MergerVariant(str, Enum): LORA="lora"; LORA_Q="lora_q"; MARS_Q="mars_q"   # device-side resolved value
```

Kotlin mirror (one file per enum), e.g. `constants/SamplingMethod.kt`:
```kotlin
enum class SamplingMethod(val wire: String) {
    GREEDY("greedy"), TOP_K("top_k"), TOP_P("top_p");
    companion object { fun fromWire(s: String) = entries.firstOrNull { it.wire == s }
        ?: error("Unknown SamplingMethod: $s") }   // fail-closed, replaces .optString fallback
}
```
`MergerVariant` is the *resolved* tag the device computes from adapter shape + quantization (today's `"lora"/"lora_q"/"mars_q"`); it is **not** a user choice — it is derived (A5).

### A3 — PEFT method registry (`config/registry/peft.py`)

One declarative spec per method replaces both the config-setup branches (`trainer/builder.py:282-346`) and the bespoke adapter-mapping key hardcoding (`trainer/utils.py:533-703`). The `component_schema` is exactly what `00_code_plans/07`'s `TrainableTensorCodec.from_peft_mapping` consumes for tensor naming.

```python
from dataclasses import dataclass
from config.constants import PEFTMethod, MergerVariant

@dataclass(frozen=True)
class AdapterComponent:
    role: str                 # "shared_A" | "intermediate" | "adapter_A" | "adapter_B" | ...
    search_pattern: str       # how to locate the tensor in the PEFT-wrapped module

@dataclass(frozen=True)
class PEFTMethodSpec:
    method: PEFTMethod
    config_class: type        # LoraConfig | MarsConfig | LoraXsConfig | AblationConfig
    component_schema: tuple[AdapterComponent, ...]   # ORDER is the codec's source of truth
    merger_variant_fp: MergerVariant                 # variant when output is not quantized
    merger_variant_q: MergerVariant                  # variant when output is quantized
    builds_mapping: bool = True

PEFT_REGISTRY: dict[PEFTMethod, PEFTMethodSpec] = { ... }   # lora, lora-xs, mars, all, nolora

def get_peft_spec(method: PEFTMethod) -> PEFTMethodSpec: ...
def build_adapter_mapping(peft_model, spec: PEFTMethodSpec) -> dict:
    """Single mapping builder. Replaces create_mars_adapter_mapping / create_lora_mapping."""
```
The MARS architecture assumptions flagged in `peft_models/mars/model.py:53,60,75,138` (attention-module name `"self_attn"`, `kwargs['hidden_states']` location) become **spec/architecture-registry data** (per-architecture attention-module name + hidden-state accessor), not in-code assumptions.

### A4 — Architecture registry (`config/registry/architecture.py`)

```python
@dataclass(frozen=True)
class ArchitectureSpec:
    architecture: str                 # config.architectures[0], e.g. "Gemma3ForCausalLM"
    onnx_config_class: type           # training export (GemmaOnnxConfig, BertOnnxConfig, ...)
    inference_model_class: type | None  # inference graph builder class; None = not yet supported
    target_modules: tuple[str, ...]   # replaces the duplicated MARS/ABLATION target maps
    attention_module_name: str = "self_attn"
    variant_key: str | None = None    # e.g. "max_position_embeddings" for Phi3 4K vs 128K
    variant_values: dict[int, type] | None = None

ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec] = { ... }

def resolve_architecture(config) -> ArchitectureSpec:
    """Single lookup. Replaces trainer/builder.py:260-272 and inference/builder.py:3234-3271."""
```
Gemma-3 inference (the FunctionGemma gate, `04_code_plans/05`) and Bert inference/pooling (encoder, `04_code_plans/01`) are added by setting `inference_model_class` on a registry entry — no new `elif` at `inference/builder.py:3234`.

### A5 — Merger registry + unified builder (`config/registry/merger.py`, `artifact/merger.py`)

```python
@dataclass(frozen=True)
class MergerSpec:
    peft_method: PEFTMethod
    quant_in: bool
    quant_out: bool
    variant: MergerVariant            # resolved device-side tag ("lora"/"lora_q"/"mars_q")
    output_filename: str              # descriptive, e.g. "merger_mars_qin_qout.onnx" — NO "_2"

def resolve_merger(peft_method: PEFTMethod, quant_in: bool, quant_out: bool) -> MergerSpec: ...

def build_merger_model(spec: MergerSpec, output_path: str) -> None:
    """ONE parameterized ONNX-graph builder. Replaces:
       create_lora_merger_model{,_2}, create_mars_merger_model{,_2}
       (artifact/merger.py:240, :351, :599, :824)."""
```
Full unification (decision 3): the four factories collapse into `build_merger_model`, parameterized by `(peft_method, quant_in, quant_out)`. Output filenames are descriptive and emitted into the **handoff map** (`00_code_plans/07`), so the C++ side resolves the merger ONNX path + variant from data, not literals. The export-side dispatch (`onnx_builder.py:628-641`) becomes: for each `MergerSpec` required by the package's PEFT method, call `build_merger_model(spec, handoff_map.merger_path(spec))`.

C++ side (coordinated with `01_code_plans/01`, which owns the on-device merge filename contract): `WeightMerger::get_merger_type` (`weight_merger.cpp:476-499`) is replaced by a handoff-map/registry lookup that returns the resolved `MergerVariant`; `run_merger_model` (`:526-712`) selects the session by variant from a registry-built map instead of `if (merger_type == "lora")` chains; merger ONNX paths (`:448-464`) come from the handoff map. The hand-derivation stays correct because `MergerVariant` is computed from the same adapter-shape + quantization signals it uses today (`has_shared_A`, `has_quantized`), just expressed as registry data.

### More registries as consumers land (F3)

This plan **owns the registry pattern** — a closed set of choices is data (enum member + registry row), never an `if/elif` chain in business logic. Beyond PEFT/architecture/merger above, the same pattern is extended (one new registry per closed set) as each consumer plan lands, so a new entry is a registry row plus an enum member with no business-logic edit:

- `TASK_REGISTRY` — task types (`TaskType`), as the task-routing consumers arrive.
- `EXECUTION_PROVIDER_REGISTRY` — `cpu`/`xnnpack`/`nnapi`/`genai`/future-NPU (`ExecutionProvider`).
- `DOCUMENT_LOADER_REGISTRY` — `txt`/`md`/`jsonl` now, `pdf`/`html` later (RAG ingestion, `03_code_plans/*`).
- `EXPORT_FRONTEND_REGISTRY` — `optimum-onnx` / `torch.onnx` export frontends.

---

## Implementation steps

1. **Deps:** confirm `pydantic>=2` lands in the core group (`00_code_plans/03`). Add `config/registry/` package.
2. **Enums first** (`config/constants.py`): add all enums (A2); replace `SUPPORTED_PEFT_METHODS`. Generate the Kotlin mirrors; add `fromWire`/`entries`-based parsing.
3. **Pydantic models** (`config/models.py`): author A1; wire `model_dump(by_alias=True)` at the three JSON emit sites; generate + check in `schemas/*.schema.json`.
4. **PEFT registry** (A3): build `PEFT_REGISTRY` + `build_adapter_mapping`; replace `trainer/builder.py:282-346` config-setup and mapping dispatch; re-express `trainer/utils.py:533-703` as the single mapping builder; move MARS attention/hidden-state assumptions into architecture-registry data.
5. **Architecture registry** (A4): build `ARCHITECTURE_REGISTRY`; replace `trainer/builder.py:260-272` and `inference/builder.py:3234-3271`; collapse `mars/utils.py` ≡ `ablation/utils.py` target maps into it.
6. **Merger unification** (A5): collapse the four factories into `build_merger_model`; replace `onnx_builder.py:628-641` dispatch; emit descriptive merger filenames into the handoff map.
7. **C++ merger** (with `01_code_plans/01`/`07`): handoff-map/registry-driven `get_merger_type`/`run_merger_model`/session paths.
8. **Lift code-only config into models:** `QuantizationOptions` (`trainer/builder.py:447-462`), reconstruction config (`peft_models/lora_xs/merger.py:34,42`), session options (`trainer/validator.py:402`), `large_model` (`onnx_builder.py:551`).
9. **Kotlin typed configs:** swap closed-set `String` fields to enums in the three config data classes; fail-closed parsing in `FileUtil.kt` against the JSON Schema; `methodMap`/`ConfigurationScreen` use enum `entries`.
10. **CI guard:** a parity check (test or script) asserting the Python enum members == the Kotlin enum `wire` values, and that no business module re-introduces an `architectures[0] ==` / `peft_method ==` / `merger_type ==` literal branch (grep guard, like the secrets guard in `02`).

---

## Interactions

- **`00_code_plans/02`** — extends `constants.py` (enums) and adds the Pydantic config layer alongside `Settings` (secrets stay stdlib as decided in `02`).
- **`00_code_plans/03`** — must list `pydantic>=2` in the core dependency group.
- **`00_code_plans/07`** — `TrainableTensorCodec.from_peft_mapping` consumes A3's `component_schema`; `canonical_inference_name` consumes A4's per-architecture naming data instead of hardcoded `self_attn→attn`/`base_layer→MatMul`.
- **`01_code_plans/01`** — consumes A5's `build_merger_model` + merger registry; owns the merge→handoff on-disk filename contract and the C++ merge path; this plan provides the registry/spec, that plan wires the device write/lookup.
- **`01_code_plans/05`** — training-export arch dispatch uses A4.
- **`04_code_plans/01` / `/05`** — Bert and Gemma-3 inference become A4 registry entries.
- **`03_code_plans/02` / `/03` / `/05`** — consume the `SamplingMethod`/`SearchType` enums and the `GenerationConfig`/`RagConfig` Pydantic models.
- **`05_code_plans/04`** — documents the schemas + registries as public contracts; the compatibility matrix is generated from the registries.

---

## References

- `https://docs.pydantic.dev/latest/` — Pydantic v2 models, validation, and JSON-schema generation (`model_json_schema`/`model_dump(by_alias=True)`) used for the typed config layer + generated `schemas/*.schema.json`.

---

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `pytest tests/unit/test_config_models.py`: round-trip each model `model_validate(json) → model_dump(by_alias=True)` is byte-stable; the `SchedulerConfig` discriminated union selects `Linear` vs `Cosine` by `schedulerType`; `extra="forbid"` rejects unknown keys; legacy `config.yml` parses through the models unchanged.
- `pytest tests/unit/test_registries.py`: `resolve_architecture` covers every entry currently in `trainer/builder.py:260-272` + `inference/builder.py:3234-3271` (parametrized) and raises a clear error for unknown architectures; `get_peft_spec` covers `lora/lora-xs/mars/all/nolora`; `resolve_merger` returns the correct `MergerVariant` for each `(method, quant_in, quant_out)` and a descriptive (no-`_2`) filename; `build_adapter_mapping` reproduces the keys the old `create_*_adapter_mapping` produced (golden vs a checked-in fixture).
- `pytest tests/unit/test_enum_parity.py`: parse the Kotlin `constants/*.kt` `wire` values and assert the set equals the Python enum values for every enum (the cross-language parity guard, F2; equivalently `python -m mobiletransformers.codegen.enums --check`).
- Grep guard (CI): no `architectures[0] ==`, `train_method ==`/`peft_method ==`, or `merger_type ==` string-literal branch remains in `trainer/`, `inference/`, `artifact/`, or `weight_merger.cpp` business paths (registry lookups only).
- Kotlin: `SamplingMethod.fromWire("bogus")` throws (fail-closed); `FileUtil` rejects a JSON whose enum value is not in the schema; module compiles (`./gradlew :MobileTransformers:compileDebugKotlin`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- Merger-unification golden: `build_merger_model` output ONNX is functionally equivalent (graph structure / IO names within quant tolerance) to the legacy `create_*_merger_model{,_2}` outputs for each variant.
- Generated artifacts are regenerable: `Model.model_json_schema(by_alias=True)` reproduces the checked-in `schemas/*.schema.json` and the golden `enums.json` byte-for-byte (no diff on re-run).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- None for this plan (typed config + registries are host-side; the device/export consumers exercise them in their own plans).

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- One typed config layer (`config/models.py`, Pydantic v2, alias-driven) is the source of truth; the three JSON emit sites write `model_dump(by_alias=True, mode="json")`; `schemas/*.schema.json` (+ a `schemaVersion`/`minReaderVersion` block, F1) and a golden `enums.json` are generated and checked in.
- One mirrored enum vocabulary exists Python↔Kotlin; the CI `parity` job (F2) fails on drift.
- The three data-driven registries (PEFT, architecture, merger) are the single source of truth for every closed choice; adding a method/architecture/merger is a registry row + enum member with **no** new `if/elif` branch (grep guard passes), and the pattern extends to further registries (F3) as consumers land.
- The four `create_*_merger_model{,_2}` factories collapse into one `build_merger_model`; the C++ merger dispatch resolves variant/paths from the handoff map/registry, not literals.
