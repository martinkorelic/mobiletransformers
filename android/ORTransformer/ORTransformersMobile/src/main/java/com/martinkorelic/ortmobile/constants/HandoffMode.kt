package com.martinkorelic.ortmobile.constants

/**
 * Mirror of mobiletransformers.config.constants.HandoffMode. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class HandoffMode(val wire: String) {
    EXTERNAL_INITIALIZER("external_initializer"),
    MODEL_INPUT("model_input"),
    ADAPTER("adapter");

    companion object {
        fun fromWire(value: String): HandoffMode =
            entries.firstOrNull { it.wire == value } ?: error("Unknown HandoffMode: $value")
    }
}
