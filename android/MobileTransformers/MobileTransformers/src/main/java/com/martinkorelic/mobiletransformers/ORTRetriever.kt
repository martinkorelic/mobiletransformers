package com.martinkorelic.mobiletransformers

import android.content.Context
import com.martinkorelic.mobiletransformers.constants.SearchType
import com.martinkorelic.mobiletransformers.packages.PackagePaths
import com.martinkorelic.mobiletransformers.rag.DimensionRegistry
import com.martinkorelic.mobiletransformers.rag.IngestionPipeline
import com.martinkorelic.mobiletransformers.rag.IngestionProgress
import com.martinkorelic.mobiletransformers.rag.ObjectBoxVectorStore
import com.martinkorelic.mobiletransformers.rag.RagDocument
import com.martinkorelic.mobiletransformers.rag.VectorStore
import com.martinkorelic.mobiletransformers.repository.RagCallback

class ORTRetriever(val cacheDir : String, val applicationContext: Context, var _ragConfig : ORTRagConfig) {

    init {
        NativeLibrary.ensureLoaded()
    }


    private val LOG_TAG = "ORTRetriever"

    // G2: every path below comes from PackagePaths. The comments this replaces described the layout in
    // prose — and got it wrong: the vector store is at `embedding/database/`, not `database/`.
    private val pkgPaths get() = PackagePaths.forCache(cacheDir, ragConfig.repoName)

    var embeddingTokenizer : ORTTokenizerNative? = null
    private var embeddingModel : Long = 0L

    var vectorDatabase : ORTVectorDatabase? = null

    var ragConfig: ORTRagConfig
        get() = _ragConfig
        set(value) {
            if (_ragConfig != value) {
                _ragConfig = value
            }
        }

    suspend fun createEmbeddingModel() {

        // Create tokenizer session if not initialized
        if (embeddingTokenizer == null) {
            embeddingTokenizer = ORTTokenizerNative(pkgPaths.embeddingTokenizer.absolutePath)
            embeddingTokenizer?.createTokenizerModel()
        }

        // Create embedding session if not initialized
        if (embeddingModel == 0L) {

            // Check if ONNX name has .onnx extension, add it if missing
            if (!ragConfig.onnxName.endsWith(".onnx", ignoreCase = true)) {
                ragConfig.onnxName += ".onnx"
            }

            // Create the embedding model
            embeddingModel = createEmbeddingSession(
                pkgPaths.embedding.absolutePath,
                ragConfig.onnxName,
                cacheDir,
                ragConfig.deviceOptions.memoryConfigId,
                ragConfig.deviceOptions.coreConfigId,
                ragConfig.deviceOptions.executionProvider,
                ragConfig.deviceOptions.enableProfiling
            )
        }

        // Initialize vector database either from existing or new database
        if (vectorDatabase == null) {
            vectorDatabase = ORTVectorDatabase.getInstance(
                _ragConfig.repoName,
                applicationContext,
                cacheDir,
                _ragConfig
            )
        }
    }

    // The active VectorStore boundary over the live ObjectBox database (#25). Null until the DB exists.
    private fun vectorStore(): VectorStore? = vectorDatabase?.let { ObjectBoxVectorStore(it) }

    fun query(queryText : String, ragArgs : ORTRagConfig, ragCallback: RagCallback? = null) {
        // Generate input tokens, but do not add if we already have past attention mask
        ragCallback?.onQueryStart()

        try {

            // #6/#25: dispatch on the TYPED SearchType, not the raw wire string. (The old
            // `searchType == null` guard above this was dead — the field is non-null String.)
            // fromWire throws on an unrecognized value, which the surrounding catch reports.
            when (SearchType.fromWire(ragArgs.searchType)) {
                SearchType.SEMANTIC -> {
                    val inputTokens = embeddingTokenizer?.tokenize(queryText, prependCls = true, appendSep = true, dropZero = true)

                    val (inputIds, attentionMask, tokenTypeIds) = prepareEmbeddingInputs(
                        inputTokens = inputTokens!!,
                        maxSequenceLength = embeddingTokenizer?.maximumTokenLength ?: 512,
                        padTokenId = embeddingTokenizer?.padToken ?: 0
                    )

                    val embeddingStartTimeMs = System.currentTimeMillis()
                    val embeddings = performEmbeddingStep(
                        session = embeddingModel,
                        inputIds = inputIds,
                        attentionMask = attentionMask,
                        tokenTypeIds = tokenTypeIds,
                        batchSize = 1,
                        sequenceLength = inputIds.size,
                        embeddingDim = ragConfig.embeddingDimension
                    )
                    val embeddingTimeMs = System.currentTimeMillis() - embeddingStartTimeMs

                    // Vector search
                    if (embeddings != null) {

                        val queryStartTimeMs = System.currentTimeMillis()
                        // Retrieval routes through the VectorStore boundary (#25), not ObjectBox directly.
                        // #27: honor the configured similarity floor.
                        val documents = vectorStore()?.search(embeddings, ragArgs.topK, ragArgs.minScore)
                        val queryTimeMs = System.currentTimeMillis() - queryStartTimeMs

                        ragCallback?.onQueryResults(
                            RagResult(
                                documents = documents,
                                embeddingTimeMs = embeddingTimeMs,
                                queryTimeMs = queryTimeMs
                            )
                        )
                    } else {
                        ragCallback?.onError(Throwable("Failed to generate embeddings for query text"))
                    }
                }
                SearchType.TEXT -> {
                    val queryStartTimeMs = System.currentTimeMillis()
                    val documents = vectorStore()?.textSearch(queryText, ragArgs.topK)
                    val queryTimeMs = System.currentTimeMillis() - queryStartTimeMs

                    ragCallback?.onQueryResults(
                        RagResult(
                            documents = documents,
                            embeddingTimeMs = 0,
                            queryTimeMs = queryTimeMs
                        )
                    )
                }

            }


        } catch (e : Exception) {
            ragCallback?.onError(e)
        } finally {
            ragCallback?.onQueryEnd()
        }

    }

    fun prepareEmbeddingInputs(
        inputTokens: IntArray,
        maxSequenceLength: Int = 512,
        padTokenId: Int = 0
    ): Triple<LongArray, LongArray, LongArray> {

        val actualLength = inputTokens.size
        val sequenceLength = minOf(actualLength, maxSequenceLength)

        // Prepare input_ids (pad or truncate to maxSequenceLength)
        val inputIds = LongArray(sequenceLength) { padTokenId.toLong() }
        for (i in 0 until sequenceLength) {
            inputIds[i] = inputTokens[i].toLong()
        }

        // Prepare attention_mask (1 for real tokens, 0 for padding)
        val attentionMask = LongArray(sequenceLength) { 0L }
        for (i in 0 until sequenceLength) {
            attentionMask[i] = 1L
        }

        // Prepare token_type_ids (all 0s for single sentence - can be null for many models)
        val tokenTypeIds = LongArray(sequenceLength) { 0L }

        return Triple(inputIds, attentionMask, tokenTypeIds)
    }

    /**
     * #26: chunk → embed → store the given [documents]. Fails closed on an unsupported embedding
     * dimension before any work; binds the real on-device embedder (tokenizer + [performEmbeddingStep])
     * into the pure [IngestionPipeline]. Returns the number of chunks inserted.
     */
    suspend fun ingestData(documents: List<RagDocument>, progress: IngestionProgress? = null): Int {
        DimensionRegistry.requireSupported(ragConfig.embeddingDimension)
        val store = vectorStore()
            ?: throw IllegalStateException("vector store not initialized; call createEmbeddingModel() first")
        val tokenizer = embeddingTokenizer
            ?: throw IllegalStateException("embedding tokenizer not initialized")

        return IngestionPipeline.ingest(
            documents = documents,
            chunkSize = ragConfig.chunkSize,
            chunkOverlap = ragConfig.chunkOverlap,
            store = store,
            progress = progress,
            embed = { chunk ->
                val tokens = tokenizer.tokenize(chunk, prependCls = true, appendSep = true, dropZero = true)
                if (tokens == null) {
                    null
                } else {
                    val (inputIds, attentionMask, tokenTypeIds) = prepareEmbeddingInputs(
                        inputTokens = tokens,
                        maxSequenceLength = tokenizer.maximumTokenLength ?: 512,
                        padTokenId = tokenizer.padToken ?: 0,
                    )
                    performEmbeddingStep(
                        session = embeddingModel,
                        inputIds = inputIds,
                        attentionMask = attentionMask,
                        tokenTypeIds = tokenTypeIds,
                        batchSize = 1,
                        sequenceLength = inputIds.size,
                        embeddingDim = ragConfig.embeddingDimension,
                    )
                }
            },
        )
    }

    suspend fun destroyEmbeddingModel() {
        releaseEmbeddingSession(embeddingModel)
    }

    external fun createEmbeddingSession(embeddingModelPath : String,
                                        embeddingModelName : String,
                                        cacheDirPath : String,
                                        memoryConfigId : String,
                                        coreConfigId : String,
                                        executionProvider : String,
                                        enableProfiling : Boolean
                                        ) : Long

    external fun releaseEmbeddingSession(model : Long)

    external fun performEmbeddingStep(
        session: Long,
        inputIds: LongArray,
        attentionMask: LongArray?,
        tokenTypeIds: LongArray?,
        batchSize: Int,
        sequenceLength: Int,
        embeddingDim: Int
    ): FloatArray?
}