package com.martinkorelic.mobiletransformers.packages

import java.io.File

/**
 * One resolver for every stage directory. **No consumer builds a stage path by appending a string.**
 *
 * Kotlin mirror of `artifacts/package_paths.py`; `cpp/package_paths.h` is the third. All three read the
 * same package, so all three must agree.
 *
 * Two layouts exist and they are NOT the same shape:
 *
 * | layout        | shape                                                              |
 * |---------------|--------------------------------------------------------------------|
 * | hub package   | `variants/<variantId>/{train,inference,embedding}` + `shared/tokenizer` |
 * | device cache  | `<cacheDir>/<repoId>/{train,inference,embedding,tokenizer}` (FLAT)  |
 *
 * The manifest has always declared the hub layout in `variant.paths`, and on the Kotlin side **nothing
 * read those values** — `ManifestValidator` only checked that the keys were present, while nine call
 * sites spelled `"$cacheDir/$repoId/inference"` by hand. `TrainingWorker` carried a comment marking
 * itself as "one place that knows the cache layout"; this is now that place, for everyone.
 *
 * This is the layer-identity problem in a second namespace, and it gets the same structural answer
 * `layer_name.h` gave the first: one spelling, one owner, a guard to keep it.
 */
class PackagePaths private constructor(
    val root: File,
    private val stages: Map<String, File>,
    private val layout: String,
) {

    companion object {
        const val STAGE_INFERENCE = "inference"
        const val STAGE_TRAIN = "train"
        const val STAGE_EMBEDDING = "embedding"
        const val STAGE_TOKENIZER = "tokenizer"

        val STAGES = listOf(STAGE_INFERENCE, STAGE_TRAIN, STAGE_EMBEDDING, STAGE_TOKENIZER)

        const val WEIGHT_HANDOFF_FILENAME = "weight_handoff_map.json"

        /** The ObjectBox store's directory name, inside the embedding stage. */
        const val EMBEDDING_DATABASE_DIRNAME = "database"

        /**
         * Resolve against a hub package using the variant's DECLARED paths.
         *
         * The manifest decides. A variant may legitimately place a stage somewhere this class would not
         * guess — `tokenizer` already lives at `shared/tokenizer` rather than under `variants/<id>/` —
         * so re-deriving the convention would quietly ignore the declaration.
         */
        @JvmStatic
        fun forHub(packageDir: File, variant: MobileTransformersManifest.Variant): PackagePaths {
            val declared = variant.paths
            require(declared.isNotEmpty()) {
                "variant '${variant.id}' declares no `paths`; a package built before the manifest " +
                    "carried per-variant paths cannot be resolved — re-export it."
            }
            val stages = declared
                .filterValues { it.isNotBlank() }
                .mapValues { (_, rel) -> File(packageDir, rel) }
            return PackagePaths(packageDir, stages, layout = "hub")
        }

        /**
         * Resolve against the FLAT on-device cache layout.
         *
         * [repoId] must already be sanitized — that mapping belongs to [PackageFormat.sanitizeRepoId]
         * and is deliberately not repeated here.
         */
        @JvmStatic
        fun forCache(cacheDir: File, repoId: String): PackagePaths {
            val base = File(cacheDir, repoId)
            return PackagePaths(
                root = base,
                stages = STAGES.associateWith { File(base, it) },
                layout = "cache",
            )
        }

        /** Convenience for the many call sites that hold the cache dir as a string. */
        @JvmStatic
        fun forCache(cacheDir: String, repoId: String): PackagePaths = forCache(File(cacheDir), repoId)
    }

    /**
     * The directory for [name], or [IllegalArgumentException] naming what this layout does declare.
     *
     * Fails closed rather than handing back a plausible path that does not exist: a silently-wrong
     * stage surfaces much later as an unrelated-looking IO error — the #35 client's
     * `INVALID_ARGUMENT : Invalid fd was supplied: -1`, which named no file at all.
     */
    fun stage(name: String): File {
        require(name in STAGES) { "unknown stage '$name'; known stages are $STAGES" }
        return stages[name] ?: throw IllegalArgumentException(
            "this $layout package does not declare a '$name' stage (declared: ${stages.keys.sorted()})"
        )
    }

    val inference: File get() = stage(STAGE_INFERENCE)
    val train: File get() = stage(STAGE_TRAIN)
    val embedding: File get() = stage(STAGE_EMBEDDING)
    val tokenizer: File get() = stage(STAGE_TOKENIZER)

    /** The handoff map, which lives inside the inference stage in both layouts. */
    val weightHandoff: File get() = File(inference, WEIGHT_HANDOFF_FILENAME)

    /**
     * The RAG vector store, INSIDE the embedding stage.
     *
     * Not shipped — ingestion creates it — which is why the two RAG sites that used to spell
     * `"$cacheDir/$repoId/embedding/database"` were carried as guard debt rather than exempted. A
     * sub-path of a stage still has to start from a resolved stage, or it drifts the same way a stage
     * does: the retriever's own comment claimed `cacheDir/modelName/database/` while the code wrote
     * `embedding/database/`.
     */
    val embeddingDatabase: File get() = File(embedding, EMBEDDING_DATABASE_DIRNAME)

    /**
     * The EMBEDDER's tokenizer, inside the embedding stage — a different tokenizer from
     * [tokenizer], which belongs to the generation model.
     */
    val embeddingTokenizer: File get() = File(embedding, STAGE_TOKENIZER)

    /** Whether the layout declares [name] at all (says nothing about what is on disk). */
    fun has(name: String): Boolean = stages.containsKey(name)
}
