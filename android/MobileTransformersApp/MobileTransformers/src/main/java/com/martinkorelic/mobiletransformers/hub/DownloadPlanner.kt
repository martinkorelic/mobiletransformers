package com.martinkorelic.mobiletransformers.hub

import com.martinkorelic.mobiletransformers.packages.MobileTransformersManifest

/**
 * #21: expands `manifest.downloadPlan[variant][group]` patterns into the concrete repo-relative file list
 * to fetch (the Kotlin mirror of the Python `_allow_patterns` in `hub/pull.py`). Pure (JVM-testable):
 * `**`/`*` suffixes are prefix-matched against `manifest.fileSizes` keys (the actual files); literals are
 * kept iff they exist. Groups fetched: always core + checksums + inference; + train/rag when requested;
 * + genai when the GenAI engine is requested.
 */
object DownloadPlanner {

    fun groupsFor(features: Set<String>, genai: Boolean): Set<String> {
        val groups = linkedSetOf("core", "checksums", "inference")
        if ("train" in features) groups += "train"
        if ("rag" in features) groups += "rag"
        if (genai) groups += "genai"
        return groups
    }

    fun planFiles(
        manifest: MobileTransformersManifest,
        variantId: String,
        features: Set<String>,
        genai: Boolean,
    ): List<String> {
        val plan = manifest.downloadPlan[variantId] ?: emptyMap()
        val allFiles = manifest.fileSizes.keys
        val out = linkedSetOf<String>()
        for (group in groupsFor(features, genai)) {
            for (pattern in plan[group].orEmpty()) {
                out += expand(pattern, allFiles)
            }
        }
        return out.toList().sorted()
    }

    private fun expand(pattern: String, allFiles: Set<String>): List<String> =
        when {
            pattern.endsWith("/**") -> {
                val prefix = pattern.removeSuffix("**")
                allFiles.filter { it.startsWith(prefix) }
            }
            pattern.endsWith("*") -> {
                val prefix = pattern.removeSuffix("*")
                allFiles.filter { it.startsWith(prefix) }
            }
            else -> if (pattern in allFiles) listOf(pattern) else emptyList()
        }
}
