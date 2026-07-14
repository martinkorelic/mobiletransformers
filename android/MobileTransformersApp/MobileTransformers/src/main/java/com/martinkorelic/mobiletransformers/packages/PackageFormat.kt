package com.martinkorelic.mobiletransformers.packages

/**
 * Cross-language package-format constants + primitives (#14/#13). [sanitizeRepoId] and [checkCompat]
 * are mirrored byte-for-byte with the Python side (`hub/package_format.py`, `artifacts/versioning.py`);
 * shared JSON oracles (`sanitize_repo_id_cases.json`, `check_compat_cases.json`) pin the parity.
 */
object PackageFormat {
    const val SCHEMA_VERSION = "1.0"
    const val MANIFEST_READER_VERSION = "1.0"
    const val MANIFEST_FILENAME = "mobiletransformers_manifest.json"

    val FEATURE_GROUPS = listOf("core", "inference", "train", "rag", "genai", "checksums")
    val VARIANT_SUBDIRS = listOf("train", "inference", "embedding")

    private val SAFE = ('a'..'z') + ('A'..'Z') + ('0'..'9') + listOf('.', '_', '-')

    /** '/' -> "__"; any other char not in [A-Za-z0-9._-] -> single '_'; no trim/case-fold/length-cap. */
    fun sanitizeRepoId(repoId: String): String {
        val sb = StringBuilder(repoId.length + 4)
        for (ch in repoId) {
            when {
                ch == '/' -> sb.append("__")
                ch in SAFE -> sb.append(ch)
                else -> sb.append('_')
            }
        }
        return sb.toString()
    }

    /** (major, minor) parsed from "MAJOR.MINOR"; null if malformed (fail closed). */
    fun parseVersion(version: String): Pair<Int, Int>? {
        val dot = version.indexOf('.')
        if (dot < 0) return null
        return try {
            val major = version.substring(0, dot).toInt()
            val minor = version.substring(dot + 1).toInt()
            if (major < 0 || minor < 0) null else Pair(major, minor)
        } catch (e: NumberFormatException) {
            null
        }
    }

    /**
     * Mirror of `artifacts/versioning.py::check_compat`. Returns true iff a reader at [readerSchema]
     * may read a doc at [docSchema] whose [docMinReader] floor is satisfied: reject when the doc needs
     * a newer major SDK, or when the reader is below the doc's minReaderVersion.
     */
    fun checkCompat(docSchema: String, docMinReader: String, readerSchema: String): Boolean {
        val doc = parseVersion(docSchema) ?: return false
        val req = parseVersion(docMinReader) ?: return false
        val rdr = parseVersion(readerSchema) ?: return false
        if (doc.first > rdr.first) return false
        if (rdr.first < req.first || (rdr.first == req.first && rdr.second < req.second)) return false
        return true
    }
}
