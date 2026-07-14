package com.martinkorelic.mobiletransformers.constants

/**
 * Mirror of mobiletransformers.config.constants.SamplingMethod. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class SamplingMethod(val wire: String) {
    GREEDY("greedy"),
    TOP_K("top_k"),
    TOP_P("top_p");

    companion object {
        fun fromWire(value: String): SamplingMethod =
            entries.firstOrNull { it.wire == value } ?: error("Unknown SamplingMethod: $value")
    }
}
