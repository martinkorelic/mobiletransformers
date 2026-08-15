package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.GenerateCallback
import com.martinkorelic.mobiletransformers.GenerateProgress
import com.martinkorelic.mobiletransformers.agent.ActionSpec
import com.martinkorelic.mobiletransformers.agent.FunctionCallValidator
import com.martinkorelic.mobiletransformers.agent.ToolCallParser
import com.martinkorelic.mobiletransformers.agent.ToolCallResult
import com.martinkorelic.mobiletransformers.agent.ToolPromptBuilder
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.AppSnackbar
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.SampleData
import com.martinkorelic.mobiletransformers.rag.PromptAssembler
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * #11/#24/#27/#37 — chat with streaming, retrieval attached to the answer it grounded, and tool calls
 * rendered in the conversation that produced them.
 *
 * The engine picker offers exactly `capabilities.availableEngines`, which the facade computes from the
 * same two conditions `ModelRuntimeFactory` applies (the package ships `genai_config.json` **and** the
 * native GenAI probe succeeds). Before that existed an app could only offer both engines and learn the
 * answer by catching `EngineUnavailableException` — using an exception as control flow for a question
 * the SDK already knew.
 */
class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(ChatUiState())
    val ui: StateFlow<ChatUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    /**
     * The same allowlist the Tool calls screen declares, so a call accepted there is accepted here.
     *
     * Kept as one value rather than two copies: the boundary and the training corpus are already
     * generated from a single object, and a third hand-written copy in Chat would be a third chance
     * for them to disagree.
     */
    val allowlist: List<ActionSpec> get() = ToolCallViewModel.ALLOWLIST

    private val validator = FunctionCallValidator(ToolCallViewModel.ALLOWLIST)

    fun onPromptChanged(value: String) {
        _ui.value = _ui.value.copy(prompt = value)
    }

    fun onRagToggled(value: Boolean) {
        _ui.value = _ui.value.copy(useRag = value)
    }

    /**
     * Whether each turn is asked for a tool call instead of prose.
     *
     * The Tool calls screen proves the mechanism in isolation; this puts it where a user would meet
     * it — mid-conversation, with the refusal or the call rendered as a turn rather than as a panel
     * on a different screen.
     */
    fun onToolsToggled(value: Boolean) {
        _ui.value = _ui.value.copy(useTools = value)
    }

    /**
     * Ingest a document into the on-device vector store.
     *
     * Retrieval reads a store that only `ingest` fills, and nothing in this app called it — so the RAG
     * switch was structurally dead: every grounded query retrieved zero sources and the model answered
     * ungrounded while the UI implied otherwise.
     *
     * @param uri a document the user picked, or `null` to install the bundled sample. Picking a file
     *   is the difference between a demo of retrieval and retrieval over something you care about.
     */
    fun ingest(uri: Uri? = null) {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        if (!model.capabilities.supportsRag) {
            val message = "this package has no embedding stage — re-pull it with the RAG feature " +
                "requested on the Models screen (it is a separate ~91 MB download group)"
            _ui.value = _ui.value.copy(error = message)
            AppSnackbar.error(message)
            return
        }
        viewModelScope.launch {
            _ui.value = _ui.value.copy(ingesting = true, error = null, ingestNote = null)
            try {
                val doc = uri?.let { copyToCache(it) } ?: SampleData.installRagDocument(getApplication())
                val result = model.ingest(doc.absolutePath, AppConfig.rag.value)
                _ui.value = _ui.value.copy(
                    ingestNote = "ingested ${doc.name}: ${result.chunkCount} chunks",
                    ingestedDocuments = _ui.value.ingestedDocuments + doc.name,
                )
                AppSnackbar.success("Ingested ${doc.name} — ${result.chunkCount} chunks")
            } catch (e: Throwable) {
                val reason = e.message ?: e::class.java.simpleName
                _ui.value = _ui.value.copy(error = reason)
                AppSnackbar.error(reason)
            } finally {
                _ui.value = _ui.value.copy(ingesting = false)
            }
        }
    }

    /**
     * Copy a picked document into app storage before ingesting it.
     *
     * `ingest` takes a filesystem path, and a `content://` URI from the document picker is not one —
     * it is a handle into another app's provider, valid only for this grant. Copying is what turns it
     * into something the SDK can open.
     */
    private fun copyToCache(uri: Uri): File {
        val name = uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() } ?: "document.txt"
        val target = File(getApplication<Application>().filesDir, "ingested-$name")
        getApplication<Application>().contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "could not open $uri" }
            target.outputStream().use { input.copyTo(it) }
        }
        return target
    }

    fun send() {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        val prompt = _ui.value.prompt.trim()
        if (prompt.isEmpty() || _ui.value.generating) return

        _ui.value = _ui.value.copy(
            messages = _ui.value.messages + ChatMessage(text = prompt, fromUser = true),
            prompt = "",
            streaming = "",
            generating = true,
            error = null,
        )

        viewModelScope.launch {
            try {
                when {
                    _ui.value.useTools -> sendAsToolCall(model, prompt)
                    _ui.value.useRag -> sendGrounded(model, prompt)
                    else -> sendPlain(model, prompt)
                }
            } catch (e: Throwable) {
                val reason = e.message ?: e::class.java.simpleName
                _ui.value = _ui.value.copy(error = reason)
                AppSnackbar.error(reason)
            } finally {
                _ui.value = _ui.value.copy(generating = false, streaming = "", phase = null)
            }
        }
    }

    private suspend fun sendPlain(
        model: com.martinkorelic.mobiletransformers.MobileTransformerModel,
        prompt: String,
    ) {
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
            messages = _ui.value.messages + ChatMessage(
                text = result.text,
                fromUser = false,
                stats = "${result.tokenCount} tokens · %.1f tok/s".format(result.avgTokensPerSecond),
            ),
        )
    }

    /**
     * #27: retrieve → assemble → generate, with both halves attached to the answer.
     *
     * The grounded path does not stream, so without a phase indicator a 20-second answer is a frozen
     * screen. Naming the phase is the difference between "it is retrieving" and "it has hung".
     */
    private suspend fun sendGrounded(
        model: com.martinkorelic.mobiletransformers.MobileTransformerModel,
        prompt: String,
    ) {
        _ui.value = _ui.value.copy(phase = "retrieving…")
        val grounded = model.generateWithRag(
            query = prompt,
            rag = AppConfig.rag.value,
            generation = AppConfig.generation.value,
            promptStrategy = PromptAssembler.DEFAULT,
        )
        _ui.value = _ui.value.copy(
            phase = null,
            messages = _ui.value.messages + ChatMessage(
                text = grounded.text,
                fromUser = false,
                // Carried ON the message, not in a screen-level field. The previous version put them
                // in a global "Sources" section that was detached from the answer and cleared by the
                // next question, so a conversation with three grounded answers showed one set of
                // sources belonging to none of them in particular.
                sources = grounded.matches.map { SourceCard(it.text, it.score) },
                assembledPrompt = grounded.prompt,
                stats = if (grounded.matches.isEmpty()) {
                    "no sources retrieved — the answer is ungrounded"
                } else {
                    "${grounded.matches.size} sources"
                },
            ),
        )
    }

    /**
     * #37 in the conversation: instruction → validated call → the intent it would fire.
     *
     * The parser comes from the loaded package's own model family. FunctionGemma does not emit JSON,
     * so a fixed JSON reader refused every well-formed call it made.
     */
    private suspend fun sendAsToolCall(
        model: com.martinkorelic.mobiletransformers.MobileTransformerModel,
        prompt: String,
    ) {
        _ui.value = _ui.value.copy(phase = "asking for a tool call…")
        val result = model.generateToolCall(
            instruction = prompt,
            validator = validator,
            config = AppConfig.generation.value,
        )
        val message = when (result) {
            is ToolCallResult.Accepted -> {
                val intended = result.dryRun()
                ChatMessage(
                    text = "",
                    fromUser = false,
                    toolCall = ToolCallCard(
                        accepted = true,
                        actionName = result.call.actionName,
                        parameters = result.call.parameters,
                        intentAction = intended.intent.action ?: "(none)",
                        willExecute = intended.willExecute,
                        raw = result.raw,
                    ),
                )
            }
            is ToolCallResult.Rejected -> ChatMessage(
                text = "",
                fromUser = false,
                // A refusal is a turn in the conversation, not an error banner: it is the *expected*
                // answer for untrusted output, and hiding it would hide the safety property working.
                toolCall = ToolCallCard(
                    accepted = false,
                    reason = result.reason,
                    raw = result.raw,
                ),
            )
        }
        _ui.value = _ui.value.copy(messages = _ui.value.messages + message)
    }

    /**
     * Feed a tool result back and let the model speak about it — the second half of the loop.
     *
     * Values are invented by this app, clearly: nothing is executed, so there is no real result to
     * report. What the turn demonstrates is that the model consumes a `<start_function_response>` and
     * answers in prose, which is the part a single call cannot show.
     */
    fun simulateToolResult(card: ToolCallCard) {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        val action = card.actionName ?: return
        viewModelScope.launch {
            _ui.value = _ui.value.copy(generating = true, phase = "feeding the result back…")
            try {
                val response = ToolPromptBuilder.functionResponse(
                    actionName = action,
                    values = mapOf("status" to "ok"),
                )
                val result = model.generate(
                    prompt = response,
                    config = AppConfig.generation.value,
                )
                _ui.value = _ui.value.copy(
                    messages = _ui.value.messages + ChatMessage(
                        text = result.text,
                        fromUser = false,
                        stats = "after a simulated result for $action",
                    ),
                )
            } catch (e: Throwable) {
                AppSnackbar.error(e.message ?: "could not continue after the tool result")
            } finally {
                _ui.value = _ui.value.copy(generating = false, phase = null)
            }
        }
    }

    fun clear() {
        _ui.value = ChatUiState(useRag = _ui.value.useRag, useTools = _ui.value.useTools)
    }
}

data class ChatUiState(
    val prompt: String = "",
    val messages: List<ChatMessage> = emptyList(),
    val streaming: String = "",
    val generating: Boolean = false,
    /** What a non-streaming turn is doing right now, so a long wait is legible rather than frozen. */
    val phase: String? = null,
    val useRag: Boolean = false,
    val useTools: Boolean = false,
    val ingesting: Boolean = false,
    val ingestNote: String? = null,
    val ingestedDocuments: List<String> = emptyList(),
    val error: String? = null,
)

/**
 * One turn.
 *
 * Sources, the assembled prompt and a tool call hang off the message that produced them rather than
 * off the screen, which is what lets a conversation keep more than one grounded answer.
 */
data class ChatMessage(
    val text: String,
    val fromUser: Boolean,
    val sources: List<SourceCard> = emptyList(),
    val assembledPrompt: String? = null,
    val toolCall: ToolCallCard? = null,
    val stats: String? = null,
)

data class SourceCard(val text: String, val score: Double)

/** An accepted or refused tool call, rendered inline. Accepted and refused are peers. */
data class ToolCallCard(
    val accepted: Boolean,
    val actionName: String? = null,
    val parameters: Map<String, String> = emptyMap(),
    val intentAction: String? = null,
    val willExecute: Boolean = false,
    val reason: String? = null,
    val raw: String = "",
)

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
