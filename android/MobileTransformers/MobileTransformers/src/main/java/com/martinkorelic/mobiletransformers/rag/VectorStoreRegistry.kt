package com.martinkorelic.mobiletransformers.rag

import com.martinkorelic.mobiletransformers.ORTVectorDatabase

/**
 * The single declared set of supported embedding dimensions (#25, F4). Replaces the scattered
 * `when (dimension)` literals and the `VectorEntity.kt:164` "could add other popular dimensions" TODO
 * with one place to declare support. An unsupported dimension fails closed with a clear message — the
 * store never silently picks a box.
 *
 * ObjectBox additionally needs a declared `@HnswIndex VectorEntity<dim>` entity per dimension (a
 * platform constraint), so "add a dimension" = one [register] call + its entity class. In-memory /
 * future backends only need the registry entry.
 */
object DimensionRegistry {
    private val supported: MutableSet<Int> = mutableSetOf(64, 128, 256, 384, 512, 768, 1024, 1536)

    /** The supported dimensions, sorted (snapshot). */
    val SUPPORTED_DIMENSIONS: Set<Int>
        get() = supported.toSortedSet()

    fun isSupported(dimension: Int): Boolean = dimension in supported

    /** Declare support for a new dimension (backend must still provide the entity/box). */
    fun register(dimension: Int) {
        require(dimension > 0) { "embedding dimension must be positive, got $dimension" }
        supported.add(dimension)
    }

    /** Fail closed unless `dimension` is registered; returns it for chaining. */
    fun requireSupported(dimension: Int): Int {
        require(dimension in supported) {
            "Unsupported embedding dimension $dimension; supported: ${SUPPORTED_DIMENSIONS}. " +
                "Add it via DimensionRegistry.register(dim) plus a declared @HnswIndex VectorEntity$dimension."
        }
        return dimension
    }
}

/** Construction context passed to a [VectorStore] factory. ObjectBox needs a live DB; others need only the dimension. */
data class VectorStoreContext(
    val embeddingDimension: Int,
    val objectBox: ORTVectorDatabase? = null,
)

/**
 * Pluggable [VectorStore] backends keyed by name (F4). ObjectBox is the default key; a new backend
 * (remote / NPU-accelerated / the test-only in-memory store) is one [register] row, not an edit to the
 * retrieval / ingestion call sites. The in-memory backend lives in the test source set and registers
 * itself there, so this default registry stays Android-only.
 */
object VectorStoreRegistry {
    const val DEFAULT_KEY: String = "objectbox"

    private val factories: MutableMap<String, (VectorStoreContext) -> VectorStore> = mutableMapOf(
        DEFAULT_KEY to { ctx ->
            val db = ctx.objectBox
                ?: throw IllegalArgumentException("VectorStore backend '$DEFAULT_KEY' requires a live ORTVectorDatabase")
            ObjectBoxVectorStore(db)
        },
    )

    fun register(key: String, factory: (VectorStoreContext) -> VectorStore) {
        factories[key] = factory
    }

    fun keys(): Set<String> = factories.keys.toSet()

    fun create(key: String, context: VectorStoreContext): VectorStore {
        val factory = factories[key]
            ?: throw IllegalArgumentException("Unknown VectorStore backend '$key'; registered: ${factories.keys}")
        return factory(context)
    }
}
