package com.martinkorelic.mobiletransformers.packages

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.martinkorelic.mobiletransformers.constants.TaskType
import java.io.File

/**
 * What objective a package's inference graph was exported for, read from the side-car the exporter
 * already writes beside that graph (`inference/optimum_config.json`).
 *
 * ### Why the device needs this
 *
 * Everything the SDK reported about a package answered "can it train / retrieve / run GenAI" — never
 * "what *kind* of model is it". So a sequence-classification encoder and a chat decoder were
 * indistinguishable to any caller, and the showcase app offered a chat box for a BERT classifier that
 * has no generative head at all. The user could type, press Send, and get a failure from deep inside
 * the runtime for a thing the package could never have done.
 *
 * The exporter has recorded `task` since the inference stage was written; nothing on the device ever
 * read it.
 *
 * ### Why the raw string is kept
 *
 * Optimum's task ids are finer-grained than [TaskType]: a decoder exports as
 * `text-generation-with-past`, which no [TaskType] entry spells. Narrowing that to the enum at read
 * time would either lose the `-with-past` distinction or fail closed on a perfectly good package, so
 * the declared string is preserved and [taskType] is offered as the best-effort mapping beside it.
 */
data class PackageTask(
    /** Exactly what the exporter declared, e.g. `text-generation-with-past`. */
    val declaredTask: String?,
    /** Model architecture family, e.g. `llama`, `bert`. */
    val modelType: String?,
    /** Label names by class index, for a classification head. Empty for every other task. */
    val id2label: Map<Int, String> = emptyMap(),
) {
    /** The shared enum entry this task corresponds to, or `null` when it names something finer. */
    val taskType: TaskType?
        get() = declaredTask?.let { declared ->
            TaskType.entries.firstOrNull { declared == it.wire || declared.startsWith("${it.wire}-") }
        }

    /** A sequence-classification package: it emits logits over labels, never tokens. */
    val isClassifier: Boolean get() = taskType == TaskType.SEQUENCE_CLASSIFICATION

    /** How many classes the head predicts, when the package names them. */
    val labelCount: Int get() = id2label.size

    private data class Wire(
        @SerializedName("task") val task: String? = null,
        @SerializedName("modelType") val modelType: String? = null,
        @SerializedName("id2label") val id2label: Map<String, String>? = null,
    )

    companion object {
        const val FILENAME = "optimum_config.json"

        private val gson = Gson()

        /** The empty answer: an older package that declares nothing is not a failure, just unknown. */
        val UNKNOWN = PackageTask(declaredTask = null, modelType = null)

        /**
         * Read the side-car from an installed package's `inference/` stage.
         *
         * Never throws. A package whose side-car is absent, truncated or from a future schema still
         * loads and still generates — the task declaration only decides which screens are *offered*,
         * and refusing to load a working model over a missing hint would be a worse trade.
         */
        fun read(inferenceDir: File): PackageTask {
            val file = File(inferenceDir, FILENAME)
            if (!file.isFile) return UNKNOWN
            val wire = runCatching { gson.fromJson(file.readText(Charsets.UTF_8), Wire::class.java) }
                .getOrNull() ?: return UNKNOWN
            return PackageTask(
                declaredTask = wire.task?.takeIf { it.isNotBlank() },
                modelType = wire.modelType?.takeIf { it.isNotBlank() },
                // HF writes id2label keyed by stringified index, which is what the exporter copies.
                id2label = wire.id2label.orEmpty()
                    .mapNotNull { (k, v) -> k.toIntOrNull()?.let { it to v } }
                    .toMap(),
            )
        }
    }
}
