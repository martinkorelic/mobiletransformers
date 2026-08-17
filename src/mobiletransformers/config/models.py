"""Typed config models (Pydantic v2) — the single source of truth for cross-boundary JSON.

Alias-driven so the on-disk JSON keeps the camelCase the Kotlin side already reads (no on-device
migration). Export writes ``Model.model_dump(by_alias=True, mode="json")``; ``schemas/*.schema.json``
is generated from ``Model.model_json_schema(by_alias=True)`` and checked in as a **CI parity artifact
only** — the device does typed fail-closed parsing (Gson + enum ``fromWire`` + ``check_compat``), not
runtime JSON-Schema validation.

``extra="ignore"`` (not ``forbid``): the schema-versioning contract requires readers to tolerate
unknown fields so additive minor bumps are non-breaking. Unknown *enum values* still fail closed
(enum coercion raises).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mobiletransformers.config.constants import (
    CoreConfigId,
    ExecutionProvider,
    MemoryConfigId,
    PEFTMethod,
    QuantizationType,
    SamplingMethod,
    SchedulerType,
    SearchType,
)

#: Cross-boundary schema version (MAJOR.MINOR) + minimum reader version. Mirrors the manifest (#13)
#: and handoff-map (#8) contract: readers fail closed on an unsupported major, tolerate minor bumps.
SCHEMA_VERSION = "1.0"
MIN_READER_VERSION = "1.0"


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore", use_enum_values=True)


class _CrossBoundary(_Base):
    """Base for models written to disk and read by Kotlin/C++ — carries the version block."""

    schema_version: str = Field(SCHEMA_VERSION, alias="schemaVersion")
    min_reader_version: str = Field(MIN_READER_VERSION, alias="minReaderVersion")


# --- nested value objects ---------------------------------------------------------
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


#: Discriminated union — the on-disk ``schedulerType`` selects Linear vs Cosine.
SchedulerConfig = Annotated[LinearScheduler | CosineScheduler, Field(discriminator="scheduler_type")]


class QuantizationOptions(_Base):
    """Lifts the ad-hoc quantization ``extra_options`` dict (trainer/builder.py) into typed config."""

    weight_type: QuantizationType = Field(QuantizationType.QINT8, alias="weightType")
    activation_symmetric: bool = Field(False, alias="activationSymmetric")
    weight_symmetric: bool = Field(False, alias="weightSymmetric")
    enable_subgraph: bool = Field(False, alias="enableSubgraph")
    force_quantize_no_input_check: bool = Field(True, alias="forceQuantizeNoInputCheck")
    matmul_const_b_only: bool = Field(True, alias="matMulConstBOnly")


# --- cross-boundary top-level models ----------------------------------------------
class GenerationConfig(_CrossBoundary):
    max_sequence_length: int = Field(128, alias="maxSequenceLength")
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    device_options: DeviceOptions = Field(default_factory=DeviceOptions, alias="deviceOptions")


class TrainingConfig(_CrossBoundary):
    peft_method: PEFTMethod = Field(PEFTMethod.LORA, alias="peftMethod")
    rank: int = 8
    alpha: int = 16
    max_steps: int = Field(10, alias="maxSteps")
    scheduler: SchedulerConfig = Field(default_factory=lambda: LinearScheduler())
    quantization: QuantizationOptions = Field(default_factory=QuantizationOptions)


class RagConfig(_CrossBoundary):
    search_type: SearchType = Field(SearchType.SEMANTIC, alias="searchType")
    top_k: int = Field(5, alias="topK")
    embedding_dim: int = Field(384, alias="embeddingDim")


#: Every model that produces a checked-in cross-boundary schema. Consumed by the codegen module.
CROSS_BOUNDARY_MODELS: dict[str, type[BaseModel]] = {
    "GenerationConfig": GenerationConfig,
    "TrainingConfig": TrainingConfig,
    "RagConfig": RagConfig,
}

__all__ = [
    "SCHEMA_VERSION",
    "MIN_READER_VERSION",
    "SamplingConfig",
    "DeviceOptions",
    "LinearScheduler",
    "CosineScheduler",
    "SchedulerConfig",
    "QuantizationOptions",
    "GenerationConfig",
    "TrainingConfig",
    "RagConfig",
    "CROSS_BOUNDARY_MODELS",
]
