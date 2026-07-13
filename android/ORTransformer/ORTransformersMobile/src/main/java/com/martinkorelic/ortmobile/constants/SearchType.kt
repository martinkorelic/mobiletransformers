package com.martinkorelic.ortmobile.constants

/**
 * Mirror of mobiletransformers.config.constants.SearchType. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class SearchType(val wire: String) {
    SEMANTIC("semantic"),
    TEXT("text");

    companion object {
        fun fromWire(value: String): SearchType =
            entries.firstOrNull { it.wire == value } ?: error("Unknown SearchType: $value")
    }
}
