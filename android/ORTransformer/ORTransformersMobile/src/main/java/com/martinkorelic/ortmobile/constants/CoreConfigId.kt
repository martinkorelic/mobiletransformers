package com.martinkorelic.ortmobile.constants

/**
 * Mirror of mobiletransformers.config.constants.CoreConfigId. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class CoreConfigId(val wire: String) {
    OPT1("opt1"),
    OPT2("opt2"),
    OPT3("opt3");

    companion object {
        fun fromWire(value: String): CoreConfigId =
            entries.firstOrNull { it.wire == value } ?: error("Unknown CoreConfigId: $value")
    }
}
