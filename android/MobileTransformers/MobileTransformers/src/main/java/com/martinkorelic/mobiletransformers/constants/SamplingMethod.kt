package com.martinkorelic.mobiletransformers.constants

/**
 * Mirror of mobiletransformers.config.constants.SamplingMethod. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class SamplingMethod(val wire: String) {
    GREEDY("greedy"),
    TOP_K("top_k"),
    TOP_P("top_p");

    /**
     * #24: the integer the native sampler expects, matching the C++ `sampling.h` enum
     * (`SamplingMethod { GREEDY=0, TOP_K=1, TOP_P=2 }`). Declared as a `when` (not an enum constructor
     * arg) so the enum-parity regex, which scans each entry's wire literal, still sees only the 3 entries.
     */
    val nativeOrdinal: Int
        get() =
            when (this) {
                GREEDY -> 0
                TOP_K -> 1
                TOP_P -> 2
            }

    companion object {
        fun fromWire(value: String): SamplingMethod =
            entries.firstOrNull { it.wire == value } ?: error("Unknown SamplingMethod: $value")
    }
}
