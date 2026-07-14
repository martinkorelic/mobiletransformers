package com.martinkorelic.mobiletransformers

import android.content.Context
import com.martinkorelic.mobiletransformers.repository.RagCallback

class ORTRetriever(val cacheDir : String, val applicationContext: Context, var _ragConfig : ORTRagConfig) {

    private val LOG_TAG = "ORTRetriever"

    // Tokenizer should be saved under cacheDir/modelName/embedding/tokenizer/...
    // Embedding model saved under cacheDir/modelName/embedding/
    // Vector database should be located under cacheDir/modelName/database/
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
            embeddingTokenizer = ORTTokenizerNative("$cacheDir/${ragConfig.repoName}/embedding/tokenizer")
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
                "$cacheDir/${ragConfig.repoName}/embedding",
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

    fun query(queryText : String, ragArgs : ORTRagConfig, ragCallback: RagCallback? = null) {
        // Generate input tokens, but do not add if we already have past attention mask
        ragCallback?.onQueryStart()

        try {

            if (ragArgs.searchType == null) {
                ragCallback?.onError(Throwable("No search type was defined."))
                return
            }

            when (ragArgs.searchType) {
                "semantic" -> {
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
                        val documents = vectorDatabase?.queryDocuments(embeddings, ragArgs.topK)
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
                "text" -> {
                    val queryStartTimeMs = System.currentTimeMillis()
                    val documents = vectorDatabase?.queryByContent(queryText, ragArgs.topK.toLong())?.map { it to 1.0 }
                    val queryTimeMs = System.currentTimeMillis() - queryStartTimeMs

                    ragCallback?.onQueryResults(
                        RagResult(
                            documents = documents,
                            embeddingTimeMs = 0,
                            queryTimeMs = queryTimeMs
                        )
                    )
                }

                else -> {
                    ragCallback?.onError(Throwable("Unknown searchType: ${ragArgs.searchType}."))
                    return
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

    suspend fun ingestData() {
        // TODO: Implement ingesting text data for now from filesystem (.md, .txt,...)
        // TODO: Simply chunking
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