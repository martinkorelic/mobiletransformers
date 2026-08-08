package com.martinkorelic.mobiletransformers

import android.content.Context
import android.util.Log
import com.martinkorelic.mobiletransformers.entity.MyObjectBox
import com.martinkorelic.mobiletransformers.entity.VectorEntity1024
import com.martinkorelic.mobiletransformers.entity.VectorEntity1024_
import com.martinkorelic.mobiletransformers.entity.VectorEntity256
import com.martinkorelic.mobiletransformers.entity.VectorEntity128
import com.martinkorelic.mobiletransformers.entity.VectorEntity128_
import com.martinkorelic.mobiletransformers.entity.VectorEntity1536
import com.martinkorelic.mobiletransformers.entity.VectorEntity1536_
import com.martinkorelic.mobiletransformers.entity.VectorEntity256_
import com.martinkorelic.mobiletransformers.entity.VectorEntity384
import com.martinkorelic.mobiletransformers.entity.VectorEntity384_
import com.martinkorelic.mobiletransformers.entity.VectorEntity512
import com.martinkorelic.mobiletransformers.entity.VectorEntity512_
import com.martinkorelic.mobiletransformers.entity.VectorEntity64
import com.martinkorelic.mobiletransformers.entity.VectorEntity64_
import com.martinkorelic.mobiletransformers.entity.VectorEntity768
import com.martinkorelic.mobiletransformers.entity.VectorEntity768_
import com.martinkorelic.mobiletransformers.entity.VectorEntityInterface
import com.martinkorelic.mobiletransformers.rag.DimensionRegistry
import io.objectbox.*
import io.objectbox.query.QueryBuilder
import java.io.File

class ORTVectorDatabase private constructor(context: Context, cacheDir : String, modelName : String, val ortRagConfig : ORTRagConfig) {

    private val boxStore: BoxStore

    // Boxes for different dimensions
    private val vectorBox64: Box<VectorEntity64>?
    private val vectorBox128: Box<VectorEntity128>?
    private val vectorBox256: Box<VectorEntity256>?
    private val vectorBox384: Box<VectorEntity384>?
    private val vectorBox512: Box<VectorEntity512>?
    private val vectorBox768: Box<VectorEntity768>?
    private val vectorBox1024: Box<VectorEntity1024>?
    private val vectorBox1536: Box<VectorEntity1536>?

    companion object {
        private const val TAG = "ORTVectorDatabase"

        @Volatile
        private var instances = mutableMapOf<String, ORTVectorDatabase>()

        // Single declared source of supported dimensions (#25). Adding a dimension is one
        // DimensionRegistry.register(dim) + its @HnswIndex VectorEntity<dim> entity.
        val SUPPORTED_DIMENSIONS: Set<Int> get() = DimensionRegistry.SUPPORTED_DIMENSIONS

        fun getInstance(
            modelName: String,
            context: Context,
            cacheDir: String,
            ortRagConfig: ORTRagConfig
        ): ORTVectorDatabase {
            return instances[modelName] ?: synchronized(this) {
                instances[modelName] ?: ORTVectorDatabase(
                    context.applicationContext,
                    cacheDir,
                    modelName,
                    ortRagConfig
                ).also { instances[modelName] = it }
            }
        }

        // Optional: Helper to check if instance exists
        fun hasInstance(key: String = "default"): Boolean = instances.containsKey(key)

        // Optional: Clear specific instance (useful for testing or cleanup)
        fun clearInstance(key: String) {
            synchronized(this) {
                instances.remove(key)?.close() // if you have a close method
            }
        }
    }

    init {
        // Initialize ObjectBox
        boxStore = MyObjectBox.builder()
            .androidContext(context)
            .directory(File("$cacheDir/$modelName/embedding/database"))
            .build()

        // Initialize only the box we need based on dimensions
        vectorBox64 = if (ortRagConfig.embeddingDimension == 64) boxStore.boxFor(VectorEntity64::class.java) else null
        vectorBox128 = if (ortRagConfig.embeddingDimension == 128) boxStore.boxFor(VectorEntity128::class.java) else null
        vectorBox256 = if (ortRagConfig.embeddingDimension == 256) boxStore.boxFor(VectorEntity256::class.java) else null
        vectorBox384 = if (ortRagConfig.embeddingDimension == 384) boxStore.boxFor(VectorEntity384::class.java) else null
        vectorBox512 = if (ortRagConfig.embeddingDimension == 512) boxStore.boxFor(VectorEntity512::class.java) else null
        vectorBox768 = if (ortRagConfig.embeddingDimension == 768) boxStore.boxFor(VectorEntity768::class.java) else null
        vectorBox1024 = if (ortRagConfig.embeddingDimension == 1024) boxStore.boxFor(VectorEntity1024::class.java) else null
        vectorBox1536 = if (ortRagConfig.embeddingDimension == 1536) boxStore.boxFor(VectorEntity1536::class.java) else null

        Log.d(TAG, "ORTVectorDatabase initialized with dimensions: ${ortRagConfig.embeddingDimension}")
        Log.d(TAG, "ObjectBox version: ${BoxStore.getVersion()}")
        Log.d(TAG, "Vector entities count: ${getVectorCount()}")
    }

    // Helper to get the active box
    private fun getActiveBox(): Box<*> {
        return when (ortRagConfig.embeddingDimension) {
            64 -> vectorBox64!!
            128 -> vectorBox128!!
            256 -> vectorBox256!!
            384 -> vectorBox384!!
            512 -> vectorBox512!!
            768 -> vectorBox768!!
            1024 -> vectorBox1024!!
            1536 -> vectorBox1536!!
            else -> throw IllegalStateException("Unsupported dimensions: $ortRagConfig.embeddingDimension")
        }
    }

    // Generic insert method
    fun insertVector(
        name: String,
        embedding: FloatArray,
        content: String = "",
        document : String = "",
        metadata: String = ""
    ): Long {
        require(embedding.size == ortRagConfig.embeddingDimension) {
            "Embedding size ${embedding.size} doesn't match configured dimensions $ortRagConfig.embeddingDimension"
        }

        return try {
            // Named arguments deliberately: the entities declare (id, name, document, content, …) while
            // this function's parameters read (name, content, document, …), and the positional calls that
            // used to be here passed `content` into `document` and vice versa for every dimension. Stored
            // documents came back with their text in `id` and their id in `text`, and `queryByContent`
            // (which indexes `content`) was searching over ids instead of document bodies.
            val id = when (ortRagConfig.embeddingDimension) {
                64 -> vectorBox64!!.put(VectorEntity64(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                128 -> vectorBox128!!.put(VectorEntity128(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                256 -> vectorBox256!!.put(VectorEntity256(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                384 -> vectorBox384!!.put(VectorEntity384(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                512 -> vectorBox512!!.put(VectorEntity512(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                768 -> vectorBox768!!.put(VectorEntity768(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                1024 -> vectorBox1024!!.put(VectorEntity1024(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                1536 -> vectorBox1536!!.put(VectorEntity1536(
                    id = 0, name = name, document = document,
                    content = content, embedding = embedding, metadata = metadata,
                ))
                else -> throw IllegalStateException("Unsupported dimensions: $ortRagConfig.embeddingDimension")
            }

            Log.d(TAG, "Inserted vector '$name' with ID: $id")
            id
        } catch (e: Exception) {
            Log.e(TAG, "Failed to insert vector: ${e.message}")
            -1L
        }
    }

    // Insert multiple vectors
    fun insertVectors(vectors: List<Triple<String, FloatArray, String>>): List<Long> {
        return try {
            val ids = mutableListOf<Long>()

            boxStore.runInTx {
                vectors.forEach { (name, embedding, description) ->
                    val id = insertVector(name, embedding, description)
                    if (id != -1L) {
                        ids.add(id)
                    }
                }
            }

            Log.d(TAG, "Inserted ${ids.size} vectors")
            ids
        } catch (e: Exception) {
            Log.e(TAG, "Failed to insert vectors: ${e.message}")
            emptyList()
        }
    }

    // Get vector by ID (returns generic interface)
    fun getVector(id: Long): VectorEntityInterface? {
        return try {
            val vector = when (ortRagConfig.embeddingDimension) {
                64 -> vectorBox64!!.get(id)
                128 -> vectorBox128!!.get(id)
                256 -> vectorBox256!!.get(id)
                384 -> vectorBox384!!.get(id)
                512 -> vectorBox512!!.get(id)
                768 -> vectorBox768!!.get(id)
                1024 -> vectorBox1024!!.get(id)
                1536 -> vectorBox1536!!.get(id)
                else -> null
            }

            if (vector != null) {
                Log.d(TAG, "Retrieved vector: ${vector.name}")
            }
            vector as? VectorEntityInterface
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get vector with ID $id: ${e.message}")
            null
        }
    }

    // Get all vectors
    fun getAllVectors(): List<VectorEntityInterface> {
        return try {
            val vectors = when (ortRagConfig.embeddingDimension) {
                64 -> vectorBox64!!.all.map { it as VectorEntityInterface }
                128 -> vectorBox128!!.all.map { it as VectorEntityInterface }
                256 -> vectorBox256!!.all.map { it as VectorEntityInterface }
                384 -> vectorBox384!!.all.map { it as VectorEntityInterface }
                512 -> vectorBox512!!.all.map { it as VectorEntityInterface }
                768 -> vectorBox768!!.all.map { it as VectorEntityInterface }
                1024 -> vectorBox1024!!.all.map { it as VectorEntityInterface }
                1536 -> vectorBox1536!!.all.map { it as VectorEntityInterface }
                else -> emptyList()
            }

            Log.d(TAG, "Retrieved ${vectors.size} vectors")
            vectors
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get all vectors: ${e.message}")
            emptyList()
        }
    }

    // Search for similar vectors using COSINE similarity
    fun queryDocuments(
        queryEmbedding: FloatArray,
        maxResults: Int = 10,
        minScore: Double = 0.0
    ): List<Pair<VectorEntityInterface, Double>> {
        require(queryEmbedding.size == ortRagConfig.embeddingDimension) {
            "Query embedding size ${queryEmbedding.size} doesn't match configured dimensions $ortRagConfig.embeddingDimension"
        }

        // TODO: Clear embeddings from returning
        return try {
            val results = when (ortRagConfig.embeddingDimension) {
                64 -> searchVectors(vectorBox64!!, VectorEntity64_.embedding, queryEmbedding, maxResults)
                128 -> searchVectors(vectorBox128!!, VectorEntity128_.embedding, queryEmbedding, maxResults)
                256 -> searchVectors(vectorBox256!!, VectorEntity256_.embedding, queryEmbedding, maxResults)
                384 -> searchVectors(vectorBox384!!, VectorEntity384_.embedding, queryEmbedding, maxResults)
                512 -> searchVectors(vectorBox512!!, VectorEntity512_.embedding, queryEmbedding, maxResults)
                768 -> searchVectors(vectorBox768!!, VectorEntity768_.embedding, queryEmbedding, maxResults)
                1024 -> searchVectors(vectorBox1024!!, VectorEntity1024_.embedding, queryEmbedding, maxResults)
                1536 -> searchVectors(vectorBox1536!!, VectorEntity1536_.embedding, queryEmbedding, maxResults)
                else -> emptyList()
            }.filter { it.second >= minScore }

            Log.d(TAG, "Found ${results.size} similar vectors (min score: $minScore)")
            results
        } catch (e: Exception) {
            Log.e(TAG, "Failed to search similar vectors: ${e.message}")
            emptyList()
        }
    }

    // Helper method for vector search
    private fun <T : VectorEntityInterface> searchVectors(
        box: Box<T>,
        embeddingProperty : Property<T>,
        queryEmbedding: FloatArray,
        maxResults: Int
    ): List<Pair<VectorEntityInterface, Double>> {
        val query = box.query()
            .nearestNeighbors(embeddingProperty, queryEmbedding, maxResults)
            .build()

        val results = query.findWithScores().map { result ->
            val entity = result.get() as VectorEntityInterface
            entity.embedding = floatArrayOf() // Clear embedding to save memory (as we don't need it)
            Pair(entity, 1 - result.score)
        }

        query.close()
        return results
    }

    // Search vectors by name (text search)
    fun queryByName(name: String): List<VectorEntityInterface> {
        return try {
            val results = when (ortRagConfig.embeddingDimension) {
                64 -> vectorBox64!!.query().contains(VectorEntity64_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                128 -> vectorBox128!!.query().contains(VectorEntity128_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                256 -> vectorBox256!!.query().contains(VectorEntity256_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                384 -> vectorBox384!!.query().contains(VectorEntity384_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                512 -> vectorBox512!!.query().contains(VectorEntity512_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                768 -> vectorBox768!!.query().contains(VectorEntity768_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                1024 -> vectorBox1024!!.query().contains(VectorEntity1024_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                1536 -> vectorBox1536!!.query().contains(VectorEntity1536_.name, name, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find().map { it as VectorEntityInterface }
                else -> emptyList()
            }

            Log.d(TAG, "Found ${results.size} vectors matching name: '$name'")
            results
        } catch (e: Exception) {
            Log.e(TAG, "Failed to search vectors by name: ${e.message}")
            emptyList()
        }
    }

    // Search vectors by content (text search)
    fun queryByContent(content: String, limit : Long = 10L): List<VectorEntityInterface> {
        return try {
            val results = when (ortRagConfig.embeddingDimension) {
                64 -> vectorBox64!!.query().contains(VectorEntity64_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                128 -> vectorBox128!!.query().contains(VectorEntity128_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                256 -> vectorBox256!!.query().contains(VectorEntity256_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                384 -> vectorBox384!!.query().contains(VectorEntity384_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                512 -> vectorBox512!!.query().contains(VectorEntity512_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                768 -> vectorBox768!!.query().contains(VectorEntity768_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                1024 -> vectorBox1024!!.query().contains(VectorEntity1024_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                1536 -> vectorBox1536!!.query().contains(VectorEntity1536_.content, content, QueryBuilder.StringOrder.CASE_INSENSITIVE).build().find(0, limit).map { it as VectorEntityInterface }
                else -> emptyList()
            }

            Log.d(TAG, "Found ${results.size} vectors matching content: '$content'")
            results
        } catch (e: Exception) {
            Log.e(TAG, "Failed to search vectors by content: ${e.message}")
            emptyList()
        }
    }

    // Update vector
    fun updateVector(id: Long, newEmbedding: FloatArray? = null, newName: String? = null): Boolean {
        return try {
            val updated = when (ortRagConfig.embeddingDimension) {
                64 -> updateVectorEntity(vectorBox64!!, id, newEmbedding, newName)
                128 -> updateVectorEntity(vectorBox128!!, id, newEmbedding, newName)
                256 -> updateVectorEntity(vectorBox256!!, id, newEmbedding, newName)
                384 -> updateVectorEntity(vectorBox384!!, id, newEmbedding, newName)
                512 -> updateVectorEntity(vectorBox512!!, id, newEmbedding, newName)
                768 -> updateVectorEntity(vectorBox768!!, id, newEmbedding, newName)
                1024 -> updateVectorEntity(vectorBox1024!!, id, newEmbedding, newName)
                1536 -> updateVectorEntity(vectorBox1536!!, id, newEmbedding, newName)
                else -> false
            }

            if (updated) {
                Log.d(TAG, "Updated vector with ID: $id")
            } else {
                Log.w(TAG, "Vector with ID $id not found for update")
            }
            updated
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update vector: ${e.message}")
            false
        }
    }

    // Helper method for updating vector entities
    private fun <T : VectorEntityInterface> updateVectorEntity(
        box: Box<T>,
        id: Long,
        newEmbedding: FloatArray?,
        newName: String?
    ): Boolean {
        val vector = box.get(id) ?: return false

        newEmbedding?.let {
            require(it.size == ortRagConfig.embeddingDimension) {
                "New embedding size ${it.size} doesn't match configured dimensions $ortRagConfig.embeddingDimension"
            }
            vector.embedding = it
        }
        newName?.let { vector.name = it }
        vector.timestamp = System.currentTimeMillis()

        box.put(vector)
        return true
    }

    // Delete vector by ID
    fun deleteVector(id: Long): Boolean {
        return try {
            val removed = (getActiveBox()).remove(id)
            if (removed) {
                Log.d(TAG, "Deleted vector with ID: $id")
            } else {
                Log.w(TAG, "Vector with ID $id not found for deletion")
            }
            removed
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete vector: ${e.message}")
            false
        }
    }

    // Delete all vectors
    fun clearAllVectors(): Boolean {
        return try {
            (getActiveBox()).removeAll()
            Log.d(TAG, "Cleared all vectors from database")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear vectors: ${e.message}")
            false
        }
    }

    // Get vector count
    fun getVectorCount(): Long {
        return try {
            getActiveBox().count()
        } catch (e: Exception) {
            0L
        }
    }

    // Get database statistics
    fun getStats(): Map<String, Any> {
        return try {
            mapOf(
                "total_vectors" to getVectorCount(),
                "embedding_dimensions" to ortRagConfig.embeddingDimension,
                "objectbox_version" to BoxStore.getVersion(),
                "database_size_kb" to (boxStore.sizeOnDisk() / 1024),
                "last_updated" to System.currentTimeMillis()
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get stats: ${e.message}")
            emptyMap()
        }
    }

    // Close database (call in onDestroy)
    fun close() {
        try {
            boxStore.close()
            Log.d(TAG, "ObjectBox database closed")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to close database: ${e.message}")
        }
    }
}