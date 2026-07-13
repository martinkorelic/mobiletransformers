package com.martinkorelic.ortmobile.constants

/**
 * Mirror of mobiletransformers.config.constants.QuantizationType. Wire values are the on-disk/JSON strings.
 * Parity with the Python source is CI-enforced (python -m mobiletransformers.codegen.enums --check).
 */
enum class QuantizationType(val wire: String) {
    QINT8("QInt8"),
    QUINT8("QUInt8"),
    INT4("int4");

    companion object {
        fun fromWire(value: String): QuantizationType =
            entries.firstOrNull { it.wire == value } ?: error("Unknown QuantizationType: $value")
    }
}
