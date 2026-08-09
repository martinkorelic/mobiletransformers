"""Non-secret shared constants + the closed-set enum vocabulary.

Migrated from ``tools/parser_config.py`` (config section names + dataset map) plus the non-secret
experiment constants from the legacy root ``config.py``. The enums below are the single Python source
of truth for every closed string set; each is mirrored 1:1 by a hand-written Kotlin ``enum class``
(``android/.../constants/*.kt``), and ``python -m mobiletransformers.codegen.enums --check`` is the CI
parity gate that fails on drift.

Secrets never live here — they belong in ``mobiletransformers.config.settings``.
"""

from __future__ import annotations

from enum import Enum


# --- Enum vocabulary (mirrored Python <-> Kotlin; wire values are the on-disk/JSON strings) -------
class SamplingMethod(str, Enum):
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"


class SchedulerType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"


class ExecutionProvider(str, Enum):
    CPU = "cpu"
    XNNPACK = "xnnpack"
    NNAPI = "nnapi"


class CoreConfigId(str, Enum):
    OPT1 = "opt1"
    OPT2 = "opt2"
    OPT3 = "opt3"


class MemoryConfigId(str, Enum):
    LOW_MEM = "low_mem"
    HIGH_PERF = "high_perf"


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    TEXT = "text"


class IndexingMode(str, Enum):
    """RAG indexing strategy (#27). v1 supports ``precompute`` only; ``dynamic`` is a fail-closed stub."""

    PRECOMPUTE = "precompute"
    DYNAMIC = "dynamic"


class QuantizationType(str, Enum):
    QINT8 = "QInt8"
    QUINT8 = "QUInt8"
    INT4 = "int4"


class PEFTMethod(str, Enum):
    LORA = "lora"
    LORA_XS = "lora-xs"
    MARS = "mars"
    ALL = "all"
    NOLORA = "nolora"


class TaskType(str, Enum):
    #: Decoder LM. Autoregressive, KV-cached, labels shaped [batch, seq].
    TEXT_GENERATION = "text-generation"
    #: Encoder embedding output (the RAG embedder). Inference/export only — it has no head, so it
    #: cannot produce a training graph; SEQUENCE_CLASSIFICATION is the trainable encoder task.
    FEATURE_EXTRACTION = "feature-extraction"
    #: Encoder classification (#33). Single forward pass, no KV cache, labels shaped [batch].
    SEQUENCE_CLASSIFICATION = "text-classification"


class HandoffMode(str, Enum):
    EXTERNAL_INITIALIZER = "external_initializer"
    MODEL_INPUT = "model_input"
    ADAPTER = "adapter"


class MergerVariant(str, Enum):
    # Device-side *resolved* tag (derived from adapter shape + quantization), not a user choice.
    LORA = "lora"
    LORA_Q = "lora_q"
    MARS_Q = "mars_q"


class ExportFrontend(str, Enum):
    """Export front-door engine selected by ``EXPORT_FRONTEND_REGISTRY`` (F3).

    Build-time / Python-only. Android never sees this value, so it is deliberately NOT registered
    in :data:`ENUM_REGISTRY` (no Kotlin mirror, no cross-language parity obligation). ``optimum-onnx``
    is the durable inference exporter; ``torch.onnx`` is the manual graph path used by the
    training-graph fallback (``OnnxConfigWithLoss`` was removed in optimum 2.1 — see
    ``spikes/optimum_migration``).
    """

    OPTIMUM_ONNX = "optimum-onnx"
    TORCH_ONNX = "torch.onnx"


#: Every enum that must stay in Python<->Kotlin parity. Consumed by the codegen parity check.
#: ``ExportFrontend`` is intentionally excluded — it is a Python build-time concern, never a
#: cross-boundary (on-device) value, so it needs no Kotlin mirror.
ENUM_REGISTRY: dict[str, type[Enum]] = {
    "SamplingMethod": SamplingMethod,
    "SchedulerType": SchedulerType,
    "ExecutionProvider": ExecutionProvider,
    "CoreConfigId": CoreConfigId,
    "MemoryConfigId": MemoryConfigId,
    "SearchType": SearchType,
    "IndexingMode": IndexingMode,
    "QuantizationType": QuantizationType,
    "PEFTMethod": PEFTMethod,
    "TaskType": TaskType,
    "HandoffMode": HandoffMode,
    "MergerVariant": MergerVariant,
}

# --- Config section names (from tools/parser_config.py) ---------------------------
ARTIFACT_CONFIG = "ARTIFACT_BUILDER"
ARTIFACT_VALIDATOR_CONFIG = "ARTIFACT_VALIDATOR"
TRAIN_CONFIG = "TRAIN_BUILDER"
INFERENCE_CONFIG = "INFERENCE_BUILDER"
INFERENCE_ARTIFACT_CONFIG = "inference_config"
TEST_GENERATION_CONFIG = "test_generation_config"

# If a value is prefixed with "data", it is loaded from the local "data/" directory.
TASK_NAME_TO_DATASET = {
    "logiqa": "data/logiqa_train",
    "hellaswag": "Rowan/hellaswag",
    "arc": "allenai/ai2_arc",
    "boolq": "google/boolq",
}

# --- Experiment constants (from legacy root config.py; non-secret) -----------------
TASK_EPOCHS = {
    "boolq": 2,
    "logiqa": 3,
    "arc_e": 4,
    "winogrande": 4,
    "arc_c": 4,
    "hellaswag": 1,
    "mini_personalqa": 6,
}
BATCH_SIZE = 32
PER_DEVICE_BATCH_SIZE = 6
GRADIENT_ACCUMULATION = 2
EXPERIMENT_RANKS = [2, 8, 32]

# --- Canonical artifact filenames -------------------------------------------------
DEFAULT_TRAIN_MODEL = "quant_model.onnx"
DEFAULT_INFERENCE_MODEL = "quant_model.onnx"

#: Supported PEFT method wire names, derived from the enum (superseded the legacy tuple).
SUPPORTED_PEFT_METHODS = tuple(m.value for m in PEFTMethod)
