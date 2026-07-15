package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.config.SamplingConfig
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

/**
 * #24: HF-aligned sampling names map to the native sampler with the exact C++ ordinals, and the public
 * `maxNewTokens` maps to the internal `maxSequenceLength`.
 */
class SamplingMappingTest {

    @Test
    fun nativeOrdinalMatchesCppEnum() {
        assertEquals(0, SamplingMethod.GREEDY.nativeOrdinal)
        assertEquals(1, SamplingMethod.TOP_K.nativeOrdinal)
        assertEquals(2, SamplingMethod.TOP_P.nativeOrdinal)
    }

    @Test
    fun fromWireRoundTrips() {
        assertEquals(SamplingMethod.TOP_K, SamplingMethod.fromWire("top_k"))
        assertEquals(SamplingMethod.GREEDY, SamplingMethod.fromWire("greedy"))
    }

    @Test
    fun fromWireFailsClosedOnUnknown() {
        assertThrows(IllegalStateException::class.java) { SamplingMethod.fromWire("beam") }
    }

    @Test
    fun samplingConfigMapsMethodToWire() {
        assertEquals("top_k", SamplingConfig(method = SamplingMethod.TOP_K).toOrt().method)
        assertEquals("greedy", SamplingConfig().toOrt().method)
    }

    @Test
    fun maxNewTokensMapsToMaxSequenceLength() {
        assertEquals(256, GenerationConfig(maxNewTokens = 256).toOrt().maxSequenceLength)
    }
}
