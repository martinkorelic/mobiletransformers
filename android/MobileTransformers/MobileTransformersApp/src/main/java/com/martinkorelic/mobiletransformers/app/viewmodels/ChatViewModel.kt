package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.GenerateCallback
import com.martinkorelic.mobiletransformers.GenerateProgress
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.SampleData
import com.martinkorelic.mobiletransformers.rag.PromptAssembler
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * #11/#24/#27 — chat with streaming, an engine picker, and RAG grounding.
 *
 * The engine picker offers exactly `capabilities.availableEngines`, which the facade now computes from
 * the same two conditions `ModelRuntimeFactory` applies (the package ships `genai_config.json` **and**
 * the native GenAI probe succeeds). Before that existed an app could only offer both engines and learn
 * the answer by catching `EngineUnavailableException` — using an exception as control flow for a
 * question the SDK already knew.
 */
class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(ChatUiState())
    val ui: StateFlow<ChatUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    fun onPromptChanged(value: String) {
        _ui.value = _ui.value.copy(prompt = value)
    }

    fun onRagToggled(value: Boolean) {
        _ui.value = _ui.value.copy(useRag = value)
    }

    /**
     * Ingest the bundled sample document into the on-device vector store.
     *
     * Retrieval reads a store that only `ingest` fills, and nothing in this app called it — so the RAG
     * switch above was structurally dead: every grounded query retrieved zero sources and the model
     * answered ungrounded while the UI implied otherwise. Ingest is the missing half of the #26/#27
     * story and belongs in the worked example, not just in the tests.
     *
     * Requires a package pulled WITH the RAG feature: the embedding encoder is its own download group,
     * so without it there is nothing to embed with. The failure says so rather than reporting an empty
     * store.
     */
    fun ingestSampleDocument() {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        if (!model.capabilities.supportsRag) {
            _ui.value = _ui.value.copy(
                error = "this package has no embedding stage — re-pull it with the RAG feature " +
                    "requested on the Models screen (it is a separate ~91 MB download group)",
            )
            return
        }
        viewModelScope.launch {
            _ui.value = _ui.value.copy(ingesting = true, error = null, ingestNote = null)
            try {
                val doc = SampleData.installRagDocument(getApplication())
                val result = model.ingest(doc.absolutePath, AppConfig.rag.value)
                _ui.value = _ui.value.copy(
                    ingestNote = "ingested ${doc.name}: ${result.chunkCount} chunks",
                )
            } catch (e: Throwable) {
                _ui.value = _ui.value.copy(error = e.message ?: e::class.java.simpleName)
            } finally {
                _ui.value = _ui.value.copy(ingesting = false)
            }
        }
    }

    fun send() {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        val prompt = _ui.value.prompt.trim()
        if (prompt.isEmpty() || _ui.value.generating) return

        _ui.value = _ui.value.copy(
            messages = _ui.value.messages + ChatMessage(prompt, fromUser = true),
            prompt = "",
            streaming = "",
            generating = true,
            error = null,
            sources = emptyList(),
        )

        viewModelScope.launch {
            try {
                if (_ui.value.useRag) {
                    // #27: retrieve -> assemble -> generate. The assembled prompt comes back for
                    // inspection, which is the point of the grounded API — an app that cannot show
                    // what the model was actually asked cannot debug a bad grounded answer.
                    val grounded = model.generateWithRag(
                        query = prompt,
                        rag = AppConfig.rag.value,
                        generation = AppConfig.generation.value,
                        promptStrategy = PromptAssembler.DEFAULT,
                    )
                    _ui.value = _ui.value.copy(
                        messages = _ui.value.messages + ChatMessage(grounded.text, fromUser = false),
                        sources = grounded.matches.map { SourceCard(it.text, it.score) },
                        assembledPrompt = grounded.prompt,
                    )
                } else {
                    val result = model.generate(
                        prompt = prompt,
                        config = AppConfig.generation.value,
                        callback = object : GenerateCallback {
                            override fun onPartialResult(progress: GenerateProgress) {
                                _ui.value = _ui.value.copy(streaming = _ui.value.streaming + progress.token)
                            }
                        },
                    )
                    _ui.value = _ui.value.copy(
                        messages = _ui.value.messages + ChatMessage(result.text, fromUser = false),
                        stats = "${result.tokenCount} tokens · " +
                            "%.1f tok/s".format(result.avgTokensPerSecond),
                    )
                }
            } catch (e: Throwable) {
                _ui.value = _ui.value.copy(error = e.message ?: e::class.java.simpleName)
            } finally {
                _ui.value = _ui.value.copy(generating = false, streaming = "")
            }
        }
    }

    fun clear() {
        _ui.value = ChatUiState(useRag = _ui.value.useRag)
    }
}

data class ChatUiState(
    val prompt: String = "",
    val messages: List<ChatMessage> = emptyList(),
    val streaming: String = "",
    val generating: Boolean = false,
    val useRag: Boolean = false,
    val ingesting: Boolean = false,
    val ingestNote: String? = null,
    val sources: List<SourceCard> = emptyList(),
    val assembledPrompt: String? = null,
    val stats: String? = null,
    val error: String? = null,
)

data class ChatMessage(val text: String, val fromUser: Boolean)

data class SourceCard(val text: String, val score: Double)

/** What the engine picker renders: the selected engine plus the ones this device/package allows. */
data class EnginePickerState(
    val selected: InferenceEngine,
    val available: Set<InferenceEngine>,
) {
    /**
     * GenAI missing is the common case, and the honest reason matters: it is either not in the package
     * or not on the device. Both collapse to "not selectable here", which is what the facade reports.
     */
    val genAiNote: String?
        get() = if (InferenceEngine.GENAI in available) {
            null
        } else {
            "GenAI is not selectable: the installed package ships no genai_config.json, or the GenAI " +
                "native probe failed on this device. Native is the guaranteed floor."
        }
}
