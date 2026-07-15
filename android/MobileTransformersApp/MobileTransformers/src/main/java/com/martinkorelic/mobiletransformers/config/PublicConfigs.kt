package com.martinkorelic.mobiletransformers.config

import com.martinkorelic.mobiletransformers.constants.CoreConfigId
import com.martinkorelic.mobiletransformers.constants.ExecutionProvider
import com.martinkorelic.mobiletransformers.constants.MemoryConfigId
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.constants.SearchType

/**
 * Public, HF-flavored typed configs for the SDK facade (#17). These are the single definition site of the
 * facade config shapes; #19 EXTENDS them (adds `applyPeft`/`pushAdapter`/callbacks) and #24 refines the
 * sampling surface — neither re-declares them. Internal mapping to the existing `ORT*Config` data classes
 * lives in `internal/config/ConfigMappers.kt`; defaults here match the `ORT*Config()` defaults 1:1 so the
 * round-trip is behavior-preserving.
 */

/** Device/execution-provider selection. Maps 1:1 to the internal `DeviceOptions`. */
data class DeviceConfig(
    val executionProvider: ExecutionProvider = ExecutionProvider.CPU,
    val coreConfigId: CoreConfigId = CoreConfigId.OPT1,
    val memoryConfigId: MemoryConfigId = MemoryConfigId.HIGH_PERF,
    val enableProfiling: Boolean = false,
)

/**
 * Sampling configuration. DECOMPOSE(#24): `03_code_plans/02` owns the final `SamplingConfig(method, …)`
 * parity/mapping; this is the foundation shape the facade needs now. Do not introduce a competing sealed
 * `Sampling` class.
 */
data class SamplingConfig(
    val method: SamplingMethod = SamplingMethod.GREEDY,
    val temperature: Float = 1f,
    val topK: Int = 10,
    val topP: Float = 0.9f,
    val seed: Int = 42,
)

/** Training configuration. The authoritative field-by-field mapping table is in `02_code_plans/01` (#19). */
data class TrainConfig(
    val epochs: Int = 1,
    val batchSize: Int = 4,
    val maxSteps: Int? = 10,
    val saveSteps: Int = 100,
    val gradientAccumulationSteps: Int = 4,
    val scheduler: SchedulerType = SchedulerType.LINEAR,
    val learningRate: Float = 1e-4f,
    val minLearningRate: Float = 0f,
    val warmupSteps: Int = 10,
    val mergeAtEnd: Boolean = true,
    val saveAtEnd: Boolean = true,
    val resumeFromState: Boolean = true,
    val device: DeviceConfig = DeviceConfig(),
)

/** Generation configuration. `maxNewTokens` is the public length field (maps to internal `maxSequenceLength`). */
data class GenerationConfig(
    val maxNewTokens: Int = 128,
    val sampling: SamplingConfig = SamplingConfig(),
    val systemPrompt: String? = null,
    val loadMerged: Boolean = false,
    val device: DeviceConfig = DeviceConfig(),
)

/** Retrieval configuration. Maps 1:1 to the internal `ORTRagConfig`. */
data class RagConfig(
    val topK: Int = 10,
    val searchType: SearchType = SearchType.SEMANTIC,
    val embeddingDimension: Int = 256,
    val chunkSize: Int = 512,
    val chunkOverlap: Int = 50,
    val maxTextLength: Int = 1024,
    val device: DeviceConfig = DeviceConfig(),
)

/**
 * PEFT selection (LoRA/MARS trainable-tensor selection). First pass is metadata-only — it folds into the
 * training config / handoff metadata; #19 wires `applyPeft`.
 */
data class PeftConfig(
    val method: String = "lora",
    val rank: Int = 8,
    val alpha: Int = 16,
    val targetModules: List<String> = emptyList(),
)

/** Local dataset description; mapped onto the existing `ORTDataCurator`/`DatasetOptions` loader. */
data class DatasetConfig(
    val trainFile: String = "arc_e",
    val maxSequenceLength: Int = 512,
    val datasetBatchSize: Int = 64,
    val maxDatasetLength: Int = 256,
)

/** Hub credentials for remote pulls (only needed by #21). */
data class HubConfig(
    val token: String? = null,
    val endpoint: String? = null,
)
