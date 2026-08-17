package com.martinkorelic.mobiletransformers.constants

/**
 * Mirror of mobiletransformers.config.constants.MergerVariant. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class MergerVariant(val wire: String) {
    LORA("lora"),
    LORA_Q("lora_q"),
    MARS_Q("mars_q");

    companion object {
        fun fromWire(value: String): MergerVariant =
            entries.firstOrNull { it.wire == value } ?: error("Unknown MergerVariant: $value")
    }
}
