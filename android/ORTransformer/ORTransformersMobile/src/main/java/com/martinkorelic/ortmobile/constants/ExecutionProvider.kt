package com.martinkorelic.ortmobile.constants

/**
 * Mirror of mobiletransformers.config.constants.ExecutionProvider. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class ExecutionProvider(val wire: String) {
    CPU("cpu"),
    XNNPACK("xnnpack"),
    NNAPI("nnapi");

    companion object {
        fun fromWire(value: String): ExecutionProvider =
            entries.firstOrNull { it.wire == value } ?: error("Unknown ExecutionProvider: $value")
    }
}
