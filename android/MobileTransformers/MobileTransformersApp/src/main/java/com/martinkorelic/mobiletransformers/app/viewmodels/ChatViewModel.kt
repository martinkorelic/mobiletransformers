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
import com.martinkorelic.mobiletransformers.app.ActionAllowlist
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.AppSnackbar
import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.app.ModelActivity
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.PermissionGate
import com.martinkorelic.mobiletransformers.app.SampleData
import com.martinkorelic.mobiletransformers.rag.PromptAssembler
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * Chat with streaming, retrieval attached to the answer it grounded, and tool calls
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
 * The app's single action declaration — see [ActionAllowlist].
     *
 * Kept as one value rather than a copy: the boundary and the training corpus are already
 * generated from that one object, and a second hand-written list here would be another chance for
 * them to disagree, with a refusal no error message could explain.
     */
    val allowlist: List<ActionSpec> get() = ActionAllowlist.ENTRIES

    private val validator = FunctionCallValidator(ActionAllowlist.ENTRIES)

    fun onPromptChanged(value: String) {
        _ui.value = _ui.value.copy(prompt = value)
    }

    fun onRagToggled(value: Boolean) {
        _ui.value = _ui.value.copy(useRag = value)
    }

    /**
     * Whether an accepted tool call fires by itself or waits for a tap.
     *
     * Defaults to [ToolExecution.Approve]. The SDK never executes anything — `IntentBinder.dryRun`
     * holds no `Context` and has no `startActivity` call site, which is the structural half of the
     * "no model output is ever executed". Firing is therefore the **app's** deliberate act, taken
     * only on a `ValidatedCall` that cleared the allowlist, and the default keeps a human in the loop
     * for it.
     */
    fun onToolExecutionChanged(value: ToolExecution) {
        _ui.value = _ui.value.copy(toolExecution = value)
    }

    /**
     * Fire an accepted call's intent.
     *
     * Only reachable from a card the validator accepted, and the intent's action string comes from
     * the app's own [ActionSpec] rather than from anything the model produced — a model selects an
     * action, it cannot name an intent. `FLAG_ACTIVITY_NEW_TASK` because the launch originates from
     * a ViewModel holding an application context, not an Activity.
     */
    fun runToolCall(card: ToolCallCard) {
        val intent = card.intent ?: return
        val context = getApplication<Application>()

        // Ask before firing, rather than catching SecurityException afterwards. A missing permission
        // used to surface only as a failed startActivity, which is what made tool calling look broken
        // on device when the manifest simply did not declare SET_ALARM.
        val missing = PermissionGate.missing(context, card.requiredPermissions)
        if (missing.isNotEmpty()) {
            val (requestable, undeclared) = PermissionGate.classify(context, missing)
            if (undeclared.isNotEmpty()) {
                // No dialog can fix an install-time permission; saying "grant it" would be wrong.
                AppSnackbar.error(PermissionGate.undeclaredMessage(undeclared))
            return
    }
            // Hand the request to the screen: a runtime prompt needs an Activity, and this holds an
            // application context. The card is remembered so the call can resume once granted.
            _ui.value = _ui.value.copy(pendingPermissions = PendingPermissions(card, requestable))
            return
    }

        runCatching {
            context.startActivity(
                android.content.Intent(intent).addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK),
            )
        }
            .onSuccess {
                markExecuted(card)
                AppSnackbar.success("Ran ${card.actionName}")
            }
            .onFailure {
                // No handler for the intent is the common case on an emulator or a stripped ROM, and
                // it is a property of the device rather than a failure of the call.
                AppSnackbar.error(
                    "Could not run ${card.actionName}: ${it.message ?: "no app handles that intent"}",
            )
    }
    }

    /**
 * The screen has finished showing the system dialog.
 *
 * @param granted whether every requested permission was allowed. A refusal is a decision, not an
 * error: it is reported and the call is dropped, never retried in a loop.
 */
    fun onPermissionResult(granted: Boolean) {
        val pending = _ui.value.pendingPermissions ?: return
        _ui.value = _ui.value.copy(pendingPermissions = null)
        if (granted) {
            runToolCall(pending.card)
                } else {
                AppSnackbar.error(
                "${pending.card.actionName} needs ${pending.permissions.joinToString()}, which was " +
                "declined — nothing was run",
                )
            }
    }

    /** Flip the card to executed, so the conversation records that it fired. */
    private fun markExecuted(card: ToolCallCard) {
        _ui.value = _ui.value.copy(
            messages = _ui.value.messages.map { m ->
                if (m.toolCall === card) m.copy(toolCall = card.copy(executed = true)) else m
            },
        )
    }

    /**
     * Whether this conversation routes through the tool-call path at all.
     *
     * **Not a user-facing switch.** It used to be a chip the user had to set *before* sending, which
     * asks them to predict something only the reply can answer: whether "wake me at 07:30" is a tool
     * call or a question about alarms. With the toggle off a genuine call came back as prose; with it
     * on, "what time is it in Tokyo" came back as a refusal.
     *
     * So the allowlist is declared on every turn for a model that has a tool-call grammar, and the
     * *outcome* decides how the turn renders — `ToolCallResult.NoCall` for prose, `Accepted` or
     * `Rejected` for a call. Declaring tools costs prompt tokens, so it is skipped for models that
     * have no such grammar, where every turn would be prose anyway.
     */
    private fun toolsAvailable(model: MobileTransformerModel): Boolean =
        model.capabilities.supportsToolCalling

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
                val result = ModelHolder.withActivity(ModelActivity.Ingesting) {
                    model.ingest(doc.absolutePath, AppConfig.rag.value)
                }
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
                ModelHolder.withActivity(ModelActivity.Generating) {
                    when {
                        // Grounding is an explicit choice about *where the answer comes from*, which
                        // is a question the user genuinely can answer in advance. Tool calling is not.
                        _ui.value.useRag -> sendGrounded(model, prompt)
                        toolsAvailable(model) -> sendMaybeToolCall(model, prompt)
                        else -> sendPlain(model, prompt)
                    }
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
                turnStats = TurnStats.of(result),
            ),
        )
    }

    /**
     * Retrieve → assemble → generate, with both halves attached to the answer.
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
                turnStats = TurnStats.of(grounded.generation),
            ),
        )
    }

    /**
     * Tool calling in the conversation: one turn, with tools declared, rendered according to what
     * came back.
     *
     * Three outcomes, three renderings, and the model picks which one — that is the whole point:
     *
     * - [ToolCallResult.NoCall] — it answered in words. An ordinary reply bubble.
     * - [ToolCallResult.Accepted] — a call this app permits, shown with the intent it *would* fire.
     * - [ToolCallResult.Rejected] — a call this app does not permit. Also a turn, not an error
     *   banner: refusing untrusted output is the safety property working, and hiding it would hide
     *   the one thing worth showing.
     *
     * `NoCall` is what makes this usable as the only chat path. It used to be a `Rejected` carrying
     * "no tool call found in the model's output", so every ordinary sentence rendered as a refusal —
     * and that message masked the real defect underneath, which was that the JSON parser was being
     * handed FunctionGemma's grammar and could not have recognised a call in it.
     */
    private suspend fun sendMaybeToolCall(
        model: com.martinkorelic.mobiletransformers.MobileTransformerModel,
        prompt: String,
    ) {
        // Streams like any other turn. The tool-call path passed no callback at all, so on a
        // tool-capable model — which is every turn for FunctionGemma — the screen sat blank for the
        // whole generation and the tokens appeared at once at the end. Whether the reply turns out
        // to be a call is decided *after* it is complete, so there is no reason not to show it
        // arriving; if it does turn out to be a call, the streamed text is replaced by the card.
        var lastProgress: GenerateProgress? = null
        val result = model.generateToolCall(
            instruction = prompt,
            validator = validator,
            config = AppConfig.generation.value,
            callback = object : GenerateCallback {
                override fun onPartialResult(progress: GenerateProgress) {
                    lastProgress = progress
                    _ui.value = _ui.value.copy(
                        // Turn markers are prompt scaffolding, not content: a model that keeps
                        // talking past its turn would otherwise stream "<end_of_turn>" into the bubble.
                        streaming = cleanTurnMarkers(_ui.value.streaming + progress.token),
                    )
                }

                override fun onCompletion(progress: GenerateProgress) {
                    lastProgress = progress
                }
            },
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
                        raw = result.raw,
                        intent = intended.intent,
                        // From the app's own ActionSpec, carried through the validator and the
                        // binder — so Run can check before firing rather than after.
                        requiredPermissions = intended.requiredPermissions,
                    ),
                    turnStats = TurnStats.of(lastProgress),
                )
            }
            is ToolCallResult.Rejected -> ChatMessage(
                text = "",
                fromUser = false,
                toolCall = ToolCallCard(
                    accepted = false,
                    reason = result.reason,
                    raw = result.raw,
                ),
                turnStats = TurnStats.of(lastProgress),
            )
            is ToolCallResult.NoCall -> ChatMessage(
                // The model chose prose. Rendered as prose, with the tool-call framing stripped so a
                // stray turn marker does not leak into the bubble.
                text = cleanTurnMarkers(result.raw).ifBlank { "(the model returned nothing)" },
                fromUser = false,
                turnStats = TurnStats.of(lastProgress),
            )
        }
        _ui.value = _ui.value.copy(messages = _ui.value.messages + message)

        // Automatic mode fires here, after the card exists, so the conversation shows what ran even
        // when nobody approved it.
        val card = message.toolCall
        if (card != null && card.accepted && _ui.value.toolExecution == ToolExecution.Automatic) {
            runToolCall(card)
        }
    }

    /**
     * Drop the turn markers a tool-declaring prompt invites the model to echo.
     *
     * The prompt ends with `<start_of_turn>model`, and a model that keeps talking past its turn emits
     * `<end_of_turn><start_of_turn>user` and carries on with a conversation it invented. Only the
     * first turn is this model's answer; the rest is it playing both parts.
     */
    private fun cleanTurnMarkers(raw: String): String =
        raw.substringBefore("<end_of_turn>")
            .substringBefore("<start_of_turn>")
            .removePrefix("model")
            .trim()

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
              ModelHolder.withActivity(ModelActivity.Generating) {
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
                        text = cleanTurnMarkers(result.text),
                        fromUser = false,
                        stats = "after a simulated result for $action",
                        turnStats = TurnStats.of(result),
                    ),
                )
              }
            } catch (e: Throwable) {
                AppSnackbar.error(e.message ?: "could not continue after the tool result")
            } finally {
                _ui.value = _ui.value.copy(generating = false, phase = null)
            }
        }
    }

    fun clear() {
        _ui.value = ChatUiState(useRag = _ui.value.useRag, toolExecution = _ui.value.toolExecution)
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
    val toolExecution: ToolExecution = ToolExecution.Approve,
    val ingesting: Boolean = false,
    val ingestNote: String? = null,
    val ingestedDocuments: List<String> = emptyList(),
    /** Non-null while a tool call waits for the system permission dialog. */
    val pendingPermissions: PendingPermissions? = null,
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
    /** Structured per-turn numbers; [stats] stays for the one-off notes that are not measurements. */
    val turnStats: TurnStats? = null,
)

data class SourceCard(val text: String, val score: Double)

/**
 * The per-turn numbers shown under an assistant message.
 *
 * Speed alone was all the app reported, and speed does not answer the question that actually
 * predicts trouble: how full the window is. A turn that is fast and at 95% of context is about to
 * start truncating; one that is slow at 3% is merely slow.
 */
data class TurnStats(
    val tokens: Int,
    val tokensPerSecond: Double,
    val contextUsed: Int,
    val contextLimit: Int,
) {
    /** e.g. `"37 tokens · 4.2 tok/s · context 512 / 32768 (2%)"`, degrading as parts go unknown. */
    fun render(): String = buildString {
        append("$tokens tokens")
        if (tokensPerSecond > 0) append(" · %.1f tok/s".format(tokensPerSecond))
        if (contextLimit > 0) {
            val pct = (contextUsed * 100.0 / contextLimit)
            append(" · context %,d / %,d (%.0f%%)".format(contextUsed, contextLimit, pct))
        } else if (contextUsed > 0) {
            append(" · context %,d tokens".format(contextUsed))
        }
    }

    companion object {
        fun of(result: com.martinkorelic.mobiletransformers.runtime.GenerationResult?) = result?.let { TurnStats(
            tokens = it.tokenCount,
            tokensPerSecond = it.avgTokensPerSecond,
            contextUsed = it.contextUsedTokens,
            contextLimit = it.contextLimit,
        ) }

        /** From the last streamed progress, for paths that have no GenerationResult in hand. */
        fun of(progress: GenerateProgress?) = progress?.let {
            TurnStats(
                tokens = it.totalDecodedTokens,
                tokensPerSecond = it.avgTokensPerSecond,
                contextUsed = it.promptTokenCount + it.totalDecodedTokens,
                contextLimit = it.contextLimit,
            )
        }
    }
}

/** An accepted or refused tool call, rendered inline. Accepted and refused are peers. */
data class ToolCallCard(
    val accepted: Boolean,
    val actionName: String? = null,
    val parameters: Map<String, String> = emptyMap(),
    val intentAction: String? = null,
    val reason: String? = null,
    val raw: String = "",
    /**
     * The intent this call would fire, or null for a refusal.
     *
     * Held so the app can run it on request. It was built by `IntentBinder` from the app's own
     * `ActionSpec`, so what is stored here is not model output.
     */
    val intent: android.content.Intent? = null,
    /** Set once the app has actually started it. */
    val executed: Boolean = false,
    /**
 * Permissions the app must hold to start [intent], from the app's own `ActionSpec`.
 *
 * Carried on the card so the Run button can check before firing rather than discovering the
 * answer as a `SecurityException` after the tap.
 */
    val requiredPermissions: List<String> = emptyList(),
            )

/**
 * A tool call waiting on the system permission dialog.
 *
 * The request has to be launched from the Activity — a ViewModel holds an application context, which
 * cannot show a permission prompt — so this is the ViewModel asking the screen to do it, holding on
 * to the card so the call can be resumed if the user agrees.
 */
data class PendingPermissions(
    val card: ToolCallCard,
    val permissions: List<String>,
)

/** What happens when a tool call is accepted. */
enum class ToolExecution(val label: String) {
    /** Show it and wait for a tap. The default: a validated call is still an action on the device. */
    Approve("Ask before running"),

    /** Fire it as soon as it is accepted. */
    Automatic("Run automatically"),
}

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
