package com.martinkorelic.mobiletransformers.config

import com.martinkorelic.mobiletransformers.constants.CoreConfigId
import com.martinkorelic.mobiletransformers.constants.ExecutionProvider
import com.martinkorelic.mobiletransformers.constants.IndexingMode
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
 * Sampling configuration (#24 locked the HF-aligned names + native mapping). `method` maps to the native
 * sampler via [com.martinkorelic.mobiletransformers.constants.SamplingMethod.nativeOrdinal]; wire strings
 * are the shared #6 enum values. Do not introduce a competing sealed `Sampling` class.
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
    /**
     * Defaults to the **low-memory** profile, unlike [GenerationConfig]'s.
     *
     * `MemoryConfigId.HIGH_PERF` enables ORT's memory-pattern planner and CPU arena. On a training
     * session that means the whole backward activation plan is pre-allocated and the arena keeps its
     * peak for the life of the run: FunctionGemma-270M measured 2.35 GB RSS + 1.02 GB swap and was
     * killed by `lmkd` on a 5.5 GB phone, for a LoRA run whose weights are ~1.07 GB.
     *
     * Pass `DeviceConfig(memoryConfigId = MemoryConfigId.HIGH_PERF)` explicitly to trade it back.
     */
    val device: DeviceConfig = DeviceConfig(memoryConfigId = MemoryConfigId.LOW_MEM),
)

/** Generation configuration. `maxNewTokens` is the public length field (maps to internal `maxSequenceLength`). */
data class GenerationConfig(
    val maxNewTokens: Int = 128,
    val sampling: SamplingConfig = SamplingConfig(),
    val systemPrompt: String? = null,
    val loadMerged: Boolean = false,
    val device: DeviceConfig = DeviceConfig(),
    /**
 * Wrap the prompt in the package's chat template, when it ships one the device can render.
     *
 * Leave it true for chat. Set it false when **you have already framed the turns yourself** — the
 * two framings would otherwise nest. `generateToolCall` does exactly that, because
 * `ToolPromptBuilder` writes a complete turn structure of its own.
 */
    val applyChatTemplate: Boolean = true,
)

/**
 * Retrieval configuration (#25/#27). Maps to the internal `ORTRagConfig`. `similarityMetric` is fixed to
 * COSINE by the ObjectBox backing store and is exposed read-only (not a settable knob). `indexingMode`
 * `DYNAMIC` is a fail-closed stub in v1 (F7).
 */
data class RagConfig(
    val topK: Int = 10,
    val searchType: SearchType = SearchType.SEMANTIC,
    val minScore: Double = 0.0,
    val indexingMode: IndexingMode = IndexingMode.PRECOMPUTE,
    // Encoder identity: `null` (the default) means "whatever the installed package declares" — these
    // are read from `embedding/rag_config.json`, which the exporter writes from the encoder it
    // actually shipped. Hardcoded defaults here would silently point the retriever at a directory and
    // a vector width that need not exist in the package, and the mismatch surfaces only at first
    // ingest on device. Set them to deliberately override the package.
    val embeddingRepoId: String? = null,
    val embeddingModelFile: String? = null,
    val embeddingDimension: Int? = null,
    val chunkSize: Int = 512,
    val chunkOverlap: Int = 50,
    val maxTextLength: Int = 1024,
    val device: DeviceConfig = DeviceConfig(),
) {
    /** Read-only: the on-device vector store uses cosine similarity; not configurable. */
    val similarityMetric: String get() = "COSINE"
}

// PEFT selection now lives in config/PeftConfig.kt as a sealed class (#19 wires `applyPeft`).

/** Local dataset description; mapped onto the existing `ORTDataCurator`/`DatasetOptions` loader. */
data class DatasetConfig(
    val trainFile: String = "arc_e",
    /**
     * Which on-device preprocessor parses [trainFile] (`logiqa`, `boolq`, `mini_personalqa`,
     * `mini_recommendation`, `cola`, `cola_cls`, `mobile_actions`). `null` = whatever the installed
     * package declares.
     *
     * The task belongs with the data, and the data is supplied by the caller: model packages
     * deliberately do not ship training sets. Leaving this unset on a package that declares nothing
     * gives `DataUtil`'s fail-closed `Unsupported task: none`, which is the honest outcome — the
     * trainer cannot guess how to parse a file it has never seen.
     */
    val task: String? = null,
    val maxSequenceLength: Int = 512,
    val datasetBatchSize: Int = 64,
    val maxDatasetLength: Int = 256,
)

/** Hub credentials for remote pulls (only needed by #21). */
data class HubConfig(
    val token: String? = null,
    val endpoint: String? = null,
)
