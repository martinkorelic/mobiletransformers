package com.martinkorelic.ortmobile.constants

/**
 * Mirror of mobiletransformers.config.constants.MemoryConfigId. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class MemoryConfigId(val wire: String) {
    LOW_MEM("low_mem"),
    HIGH_PERF("high_perf");

    companion object {
        fun fromWire(value: String): MemoryConfigId =
            entries.firstOrNull { it.wire == value } ?: error("Unknown MemoryConfigId: $value")
    }
}
