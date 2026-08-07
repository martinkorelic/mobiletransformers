package com.martinkorelic.mobiletransformers.constants

/**
 * Mirror of mobiletransformers.config.constants.IndexingMode (#27). Wire values are the on-disk/JSON
 * strings; parity is CI-enforced (python -m mobiletransformers.codegen.enums --check). v1 supports
 * [PRECOMPUTE] only; [DYNAMIC] is a fail-closed stub (F7).
 */
enum class IndexingMode(val wire: String) {
    PRECOMPUTE("precompute"),
    DYNAMIC("dynamic");

    companion object {
        fun fromWire(value: String): IndexingMode =
            entries.firstOrNull { it.wire == value } ?: error("Unknown IndexingMode: $value")
    }
}
