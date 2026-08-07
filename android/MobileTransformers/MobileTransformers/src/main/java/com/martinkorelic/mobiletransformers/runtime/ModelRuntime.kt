package com.martinkorelic.mobiletransformers.runtime

import android.util.Log
import com.martinkorelic.mobiletransformers.ORTGenerationConfig
import com.martinkorelic.mobiletransformers.ORTGeneratorGenAI
import com.martinkorelic.mobiletransformers.ORTGeneratorNative
import com.martinkorelic.mobiletransformers.ORTTokenizerNative
import com.martinkorelic.mobiletransformers.constants.ExecutionProvider
import com.martinkorelic.mobiletransformers.repository.GenerationCallback

/**
 * The single inference-engine boundary (#11): one interface, two implementations — Native (guaranteed
 * floor) and GenAI (opt-in) — over the **same** `inference/` package produced by File #9. The engine is a
 * selection over one package, never a separate package/build. `generate` MUST drive the exact same
 * `GenerationCallback`/`InferenceProgress` sequence on both engines so the facade/UI never branch on engine.
 *
 * This is `ModelRuntime` (engine boundary); #17's whole-model facade contract is `ModelSession`.
 */
interface ModelRuntime {
    val capabilities: EngineCapabilities

    /** Open a session over `<cacheDir>/<repoName>/inference` for [config]. */
    suspend fun load(cacheDir: String, config: ORTGenerationConfig)

    /** Same return + callback sequence as `ORTGeneratorNative.generate`. */
    fun generate(
        promptText: String,
        generationArgs: ORTGenerationConfig,
        callback: GenerationCallback? = null,
    ): String

    fun release()
}

/** Engine-level capability record (distinct from #17's model-level `RuntimeCapabilities`). */
data class EngineCapabilities(
    val engine: InferenceEngine,
    val supportsStreaming: Boolean,
    val supportsLoadMergedWeights: Boolean,
    val maxContextLength: Int,
)

/**
 * Data-driven execution-provider registry (F3): the engine's ORT execution providers are rows, not an
 * `if/elif` over EP names. Each row carries an availability probe + an [InferenceEngine] affinity. Adding a
 * provider (e.g. an NPU EP) is a registry row + enum member — no business-logic edit. `ModelRuntimeFactory`
 * resolves the ordered provider list for a chosen engine from this registry.
 */
data class ExecutionProviderRow(
    val provider: String,
    val engine: InferenceEngine,
    val available: () -> Boolean,
    /** Lower = earlier in the EP append order for its engine. */
    val order: Int,
)

object EngineRegistry {
    /** The single source of EP→engine affinity + availability. `genai` is GenAI's provider row. */
    val EXECUTION_PROVIDER_REGISTRY: List<ExecutionProviderRow> = listOf(
        ExecutionProviderRow(ExecutionProvider.CPU.wire, InferenceEngine.NATIVE, { true }, 0),
        ExecutionProviderRow(ExecutionProvider.XNNPACK.wire, InferenceEngine.NATIVE, { true }, 1),
        ExecutionProviderRow(ExecutionProvider.NNAPI.wire, InferenceEngine.NATIVE, { true }, 2),
        ExecutionProviderRow("genai", InferenceEngine.GENAI, { GenAiSupport.available() }, 0),
    )

    /** Ordered, available EP names for [engine] (F3 — resolved from data, not branched on strings). */
    fun providersFor(engine: InferenceEngine): List<String> =
        EXECUTION_PROVIDER_REGISTRY
            .filter { it.engine == engine && it.available() }
            .sortedBy { it.order }
            .map { it.provider }
}

/**
 * `genaiAvailable()` = Gate 0.1 passed AND the GenAI stack is linked AND `OgaCreateModel` is present. The
 * native probe is injectable (default: JNI symbol/init check) so selection logic is JVM-testable.
 */
object GenAiSupport {
    /** Overridable for tests; production points at the native probe. */
    @Volatile
    var probe: () -> Boolean = { nativeGenAiAvailable() }

    fun available(): Boolean = runCatching { probe() }.getOrDefault(false)

    private external fun nativeGenAiAvailable(): Boolean
}

/**
 * Selects an engine and constructs a [ModelRuntime] with **transparent fallback to Native** — a GenAI
 * load failure NEVER reaches the caller (logged, then Native). The pure [selectEngine] decision is
 * JVM-testable; [create] performs the device construction.
 */
object ModelRuntimeFactory {
    /**
     * Pure selection: honor [requested] (or [defaultEngine]) only if the variant supports it AND GenAI is
     * available; otherwise Native (the floor).
     */
    fun selectEngine(
        requested: InferenceEngine?,
        supportedEngines: Set<String>,
        defaultEngine: InferenceEngine,
        genaiAvailable: Boolean,
    ): InferenceEngine {
        val want = requested ?: defaultEngine
        val genaiOk =
            want == InferenceEngine.GENAI &&
                "genai" in supportedEngines &&
                genaiAvailable
        return if (genaiOk) InferenceEngine.GENAI else InferenceEngine.NATIVE
    }

    /**
     * Construct + load a [ModelRuntime] over one `inference/` package with **transparent fallback to
     * Native**: a GenAI selection that fails to load is logged and falls through to Native — the caller
     * never sees a GenAI error. [supportedEngines] comes from the manifest variant (#13); when unknown,
     * pass both and let [GenAiSupport]/`config.engine` decide.
     */
    suspend fun create(
        cacheDir: String,
        tokenizer: ORTTokenizerNative,
        config: ORTGenerationConfig,
        supportedEngines: Set<String> = setOf("native", "genai"),
    ): ModelRuntime {
        val engine = selectEngine(
            config.engine, supportedEngines, InferenceEngine.NATIVE, GenAiSupport.available(),
        )
        if (engine == InferenceEngine.GENAI) {
            try {
                return ORTGeneratorGenAI(cacheDir, tokenizer, config).also { it.load(cacheDir, config) }
            } catch (e: Throwable) {
                Log.w("ModelRuntimeFactory", "GenAI engine unavailable, falling back to Native: ${e.message}")
            }
        }
        return ORTGeneratorNative(cacheDir, tokenizer, config).also { it.load(cacheDir, config) }
    }
}
