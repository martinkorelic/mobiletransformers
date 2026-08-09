package com.martinkorelic.mobiletransformers.constants

/**
 * Mirror of mobiletransformers.config.constants.TaskType. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class TaskType(val wire: String) {
    TEXT_GENERATION("text-generation"),
    FEATURE_EXTRACTION("feature-extraction"),
    SEQUENCE_CLASSIFICATION("text-classification");

    companion object {
        fun fromWire(value: String): TaskType =
            entries.firstOrNull { it.wire == value } ?: error("Unknown TaskType: $value")
    }
}
