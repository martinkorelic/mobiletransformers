package com.martinkorelic.ortmobile.constants

/**
 * Mirror of mobiletransformers.config.constants.SchedulerType. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class SchedulerType(val wire: String) {
    LINEAR("linear"),
    COSINE("cosine");

    companion object {
        fun fromWire(value: String): SchedulerType =
            entries.firstOrNull { it.wire == value } ?: error("Unknown SchedulerType: $value")
    }
}
