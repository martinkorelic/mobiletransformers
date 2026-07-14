package com.martinkorelic.mobiletransformers.constants

/**
 * Mirror of mobiletransformers.config.constants.PEFTMethod. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class PEFTMethod(val wire: String) {
    LORA("lora"),
    LORA_XS("lora-xs"),
    MARS("mars"),
    ALL("all"),
    NOLORA("nolora");

    companion object {
        fun fromWire(value: String): PEFTMethod =
            entries.firstOrNull { it.wire == value } ?: error("Unknown PEFTMethod: $value")
    }
}
