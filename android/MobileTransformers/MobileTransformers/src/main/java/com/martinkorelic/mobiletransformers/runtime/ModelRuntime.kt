package com.martinkorelic.mobiletransformers.runtime

import android.util.Log
import com.martinkorelic.mobiletransformers.EngineUnavailableException
import com.martinkorelic.mobiletransformers.NativeLibrary
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
    var probe: () -> Boolean = {
        NativeLibrary.ensureLoaded()
        nativeGenAiAvailable()
    }

    fun available(): Boolean = runCatching { probe() }.getOrDefault(false)

    private external fun nativeGenAiAvailable(): Boolean
}

/**
 * Selects an engine and constructs a [ModelRuntime]. GenAI that was **auto-selected** falls back to
 * Native transparently (#11's guaranteed floor); GenAI that the caller **explicitly asked for** fails
 * loudly instead. The pure [selectEngine] decision is JVM-testable; [create] performs the device
 * construction.
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
     * The engines a picker may offer for this package on this device — the same decision
     * [selectEngine] makes, asked ahead of time.
     *
     * ### Why this is here and not in the facade
     *
     * `RuntimeCapabilities.availableEngines` used to be built inline in `MobileTransformers`, from
     * **two** of the three conditions [selectEngine] applies: the package ships `genai_config.json`,
     * and the native probe succeeds. It ignored the third — the manifest variant's `supportedEngines`.
     *
     * FunctionGemma is exactly the package that separates them. Its `inference/` stage carries a
     * `genai_config.json` (optimum writes one), but its manifest declares `supportedEngines:
     * ["native"]`, because Gemma-3 inference export goes through optimum's `main_export` rather than
     * the vendored GenAI builder. So the facade advertised GenAI, the app's picker offered it, the
     * user chose it, and [create] then refused it — correctly — with "explicitly requested but not
     * selectable". The SDK contradicted itself within one load, and the app was blamed for it.
     *
     * Deriving both answers from one place is the fix; `EngineSelectionTest` asserts they agree for
     * every combination of the three inputs.
     *
     * @param declaredEngines the manifest variant's declaration, or `null` when the package declares
     *   none. Null stays permissive — an unknown declaration must not become a narrower one, which is
     *   the same rule [create]'s default argument encodes.
     */
    fun enginesAvailableFor(
        declaredEngines: Set<String>?,
        genaiConfigPresent: Boolean,
        genaiAvailable: Boolean,
    ): Set<InferenceEngine> = buildSet {
        add(InferenceEngine.NATIVE)
        val selectable = selectEngine(
            requested = InferenceEngine.GENAI,
            supportedEngines = declaredEngines ?: setOf("native", "genai"),
            defaultEngine = InferenceEngine.NATIVE,
            genaiAvailable = genaiAvailable,
        )
        if (genaiConfigPresent && selectable == InferenceEngine.GENAI) add(InferenceEngine.GENAI)
    }

    /**
     * Pure rule: may a GenAI that is unavailable or failed to load be silently replaced by Native?
     *
     * Only when the caller expressed no preference. `requested == null` means "pick for me", and Native
     * is #11's guaranteed floor. Naming an engine and receiving a different one is a wrong answer, not a
     * graceful degradation — see [create].
     */
    fun mayFallBackToNative(requested: InferenceEngine?): Boolean = requested != InferenceEngine.GENAI

    /**
     * Construct + load a [ModelRuntime] over one `inference/` package. [supportedEngines] comes from the
     * manifest variant (#13); when unknown, pass both and let [GenAiSupport]/`config.engine` decide.
     *
     * **Fallback is conditional on who chose the engine.** When GenAI was auto-selected (the caller
     * expressed no preference) a load failure falls through to Native, which is #11's guaranteed floor.
     * When the caller *named* GenAI it is raised as [EngineUnavailableException].
     *
     * The unconditional version of this was a real defect, not a hypothetical one: genai_config.json
     * carried a `config_entries` key that GenAI 0.14 rejects, so GenAI never loaded on any package the
     * training stage touched — and nothing said so. `DualEngineParityTest` compared Native with Native
     * and passed, and both `MemoryRssTest` rows recorded Native, so Gate 0.1 #1 and #4 were both read as
     * proven off measurements of a single engine. A degradation nobody can observe is worse than a
     * failure; asking for an engine and silently getting another one is not a floor, it is a lie.
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
        val explicitlyRequested = !mayFallBackToNative(config.engine)
        if (engine == InferenceEngine.GENAI) {
            try {
                return ORTGeneratorGenAI(cacheDir, tokenizer, config).also { it.load(cacheDir, config) }
            } catch (e: Throwable) {
                if (explicitlyRequested) {
                    throw EngineUnavailableException(
                        InferenceEngine.GENAI,
                        "explicitly requested but failed to load from '$cacheDir': ${e.message}. " +
                            "Not falling back to Native — that would silently answer a different " +
                            "question than the one asked. Pass engine=null to allow the fallback.",
                    )
                }
                Log.w("ModelRuntimeFactory", "GenAI engine unavailable, falling back to Native: ${e.message}")
            }
        } else if (explicitlyRequested) {
            // Selection itself rejected GenAI (unsupported by the variant, or unavailable on this
            // device). Same rule: the caller named it, so say so rather than quietly substituting.
            throw EngineUnavailableException(
                InferenceEngine.GENAI,
                "explicitly requested but not selectable: supportedEngines=$supportedEngines, " +
                    "genaiAvailable=${GenAiSupport.available()}. Pass engine=null to allow the fallback.",
            )
        }
        return ORTGeneratorNative(cacheDir, tokenizer, config).also { it.load(cacheDir, config) }
    }
}
