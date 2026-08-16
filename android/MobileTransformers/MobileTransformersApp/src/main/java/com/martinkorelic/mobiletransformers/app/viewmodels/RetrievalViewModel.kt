package com.martinkorelic.mobiletransformers.app.viewmodels

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.martinkorelic.mobiletransformers.app.AppConfig
import com.martinkorelic.mobiletransformers.app.AppSnackbar
import com.martinkorelic.mobiletransformers.app.ModelActivity
import com.martinkorelic.mobiletransformers.app.ModelHolder
import com.martinkorelic.mobiletransformers.app.ModelState
import com.martinkorelic.mobiletransformers.app.SampleData
import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Retrieval on its own: a query in, the closest passages out, and nothing generated.
 *
 * `MobileTransformerModel.retrieve()` has existed since retrieval did, and nothing in the app called
 * it — the only way to see retrieval was to turn on grounding in Chat and read the source cards
 * attached to an answer. That conflates two things that fail for different reasons. When a grounded
 * answer is wrong, there is no way to tell whether retrieval returned the wrong passages or the model
 * ignored the right ones, because the only view of retrieval is filtered through generation.
 *
 * This screen is the unfiltered view, and it is also the only part of the retrieval story an **encoder
 * package** can show at all: `all-MiniLM-L6-v2` has no generative head, so Chat is hidden for it and
 * grounding is unreachable. Ingestion lives here too, for the same reason — a store you fill is a
 * property of retrieval, not of a conversation.
 */
class RetrievalViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(RetrievalUiState())
    val ui: StateFlow<RetrievalUiState> = _ui.asStateFlow()

    val modelState: StateFlow<ModelState> = ModelHolder.state

    /** Queries whose best match is a different bundled document each time. */
    val exampleQueries: List<String> = SampleData.RAG_EXAMPLE_QUERIES

    fun onQueryChanged(value: String) {
        _ui.value = _ui.value.copy(query = value)
    }

    fun search() {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        val query = _ui.value.query.trim()
        if (query.isEmpty() || _ui.value.searching) return

        if (!model.capabilities.supportsRag) {
            report(NO_EMBEDDING_STAGE)
            return
        }

        viewModelScope.launch {
            _ui.value = _ui.value.copy(searching = true, error = null)
            try {
                val result = ModelHolder.withActivity(ModelActivity.Generating) {
                    model.retrieve(query, AppConfig.rag.value)
                }
                _ui.value = _ui.value.copy(
                    matches = result.matches,
                    // The query the matches answer, so an edited box cannot silently re-label results.
                    searchedQuery = query,
                    queryTimeMs = result.queryTimeMs,
                    // An empty store and a query that genuinely matches nothing look identical in the
                    // result list, and the fix for them is completely different. Record which it was.
                    searchedWithEmptyStore = _ui.value.ingestedDocuments.isEmpty(),
                )
            } catch (e: Throwable) {
                report(e.message ?: e::class.java.simpleName)
            } finally {
                _ui.value = _ui.value.copy(searching = false)
            }
        }
    }

    /**
     * Fill the store.
     *
     * @param uri a document the user picked, or `null` to ingest the whole bundled corpus. Retrieval
     *   over one document cannot demonstrate ranking — every result is a chunk of the only thing
     *   there is — so the sample button installs four separable subjects rather than one file.
     */
    fun ingestSamples() = ingestAll { SampleData.installRagCorpus(getApplication()) }

    fun ingest(uri: Uri) = ingestAll { listOf(copyToCache(uri)) }

    private fun ingestAll(resolve: () -> List<File>) {
        val model = (ModelHolder.state.value as? ModelState.Loaded)?.model ?: return
        if (!model.capabilities.supportsRag) {
            report(NO_EMBEDDING_STAGE)
            return
        }
        viewModelScope.launch {
            _ui.value = _ui.value.copy(ingesting = true, error = null, note = null)
            try {
                val documents = resolve()
                var chunks = 0
                val names = mutableListOf<String>()
                for (document in documents) {
                    val result = ModelHolder.withActivity(ModelActivity.Ingesting) {
                        model.ingest(document.absolutePath, AppConfig.rag.value)
                    }
                    chunks += result.chunkCount
                    names += document.name
                }
                _ui.value = _ui.value.copy(
                    note = "ingested ${names.size} document(s), $chunks chunks",
                    // Re-ingesting the same file adds chunks again; the list is what is ON SCREEN, so
                    // it must not grow duplicates and imply a bigger store than there is.
                    ingestedDocuments = (_ui.value.ingestedDocuments + names).distinct(),
                )
                AppSnackbar.success("Ingested $chunks chunks from ${names.size} document(s)")
            } catch (e: Throwable) {
                report(e.message ?: e::class.java.simpleName)
            } finally {
                _ui.value = _ui.value.copy(ingesting = false)
            }
        }
    }

    fun useExample(query: String) {
        _ui.value = _ui.value.copy(query = query)
    }

    fun clear() {
        _ui.value = _ui.value.copy(matches = emptyList(), searchedQuery = "", error = null)
    }

    private fun report(message: String) {
        _ui.value = _ui.value.copy(error = message)
        AppSnackbar.error(message)
    }

    /**
     * `ingest` takes a filesystem path, and a `content://` URI is not one — it is a handle into
     * another app's provider, valid only for this grant. Copying is what turns it into something the
     * SDK can open.
     */
    private fun copyToCache(uri: Uri): File {
        val name = uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() } ?: "document.txt"
        val target = File(getApplication<Application>().filesDir, name)
        getApplication<Application>().contentResolver.openInputStream(uri)?.use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        } ?: error("could not open $uri")
        return target
    }

    private companion object {
        const val NO_EMBEDDING_STAGE =
            "this package has no embedding stage — re-pull it with the RAG feature requested on the " +
                "Models screen (it is a separate download group)"
    }
}

data class RetrievalUiState(
    val query: String = "",
    val searching: Boolean = false,
    val ingesting: Boolean = false,
    /** Highest score first — `RetrievalResult.matches` is already ranked. */
    val matches: List<RetrievalMatch> = emptyList(),
    val searchedQuery: String = "",
    val queryTimeMs: Long = 0L,
    val searchedWithEmptyStore: Boolean = false,
    val ingestedDocuments: List<String> = emptyList(),
    val note: String? = null,
    val error: String? = null,
) {
    /** A search ran and found nothing — distinct from "no search has run yet". */
    val foundNothing: Boolean get() = searchedQuery.isNotEmpty() && matches.isEmpty()
}
