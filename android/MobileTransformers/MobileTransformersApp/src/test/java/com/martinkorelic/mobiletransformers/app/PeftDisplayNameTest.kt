package com.martinkorelic.mobiletransformers.app

import com.martinkorelic.mobiletransformers.app.viewmodels.peftDisplayName
import com.martinkorelic.mobiletransformers.app.viewmodels.peftOf
import com.martinkorelic.mobiletransformers.app.viewmodels.peftOptions
import com.martinkorelic.mobiletransformers.config.PeftConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * PEFT methods are spelled `LoRA` and `MARS` for a reader and `lora`/`mars` on the wire.
 *
 * The separation is the whole point of these tests. The lowercase forms are the `PEFTMethod` enum
 * mirrored from Python and pinned by `make parity`, they are what a package manifest's `peftMethods`
 * field contains, and they are what [peftOf] matches on. The obvious "fix" for the casing — editing
 * the strings in [peftOptions] — would look right on screen and silently break the picker, because
 * `peftOf("MARS-opt0", …)` falls through to its `else` branch and hands back **LoRA**.
 *
 * So: one test that the display is pretty, and one that the wire values underneath are untouched.
 */
class PeftDisplayNameTest {

    @Test
    fun `acronyms are spelled the way they are written`() {
        assertEquals("LoRA", peftDisplayName("lora"))
        assertEquals("MARS", peftDisplayName("mars"))
        assertEquals("LoRA-XS", peftDisplayName("lora-xs"))
        assertEquals("MARS-opt0", peftDisplayName("mars-opt0"))
        assertEquals("MARS-opt1", peftDisplayName("mars-opt1"))
        assertEquals("MARS-quantized", peftDisplayName("mars-quantized"))
    }

    @Test
    fun `the picker's options are still WIRE values, not display names`() {
        // If this fails, someone prettied `peftOptions` itself. See the class docstring: the picker
        // would keep rendering correctly and quietly select LoRA for every MARS variant.
        peftOptions.forEach { option ->
            assertEquals("peftOptions must hold lowercase wire values", option.lowercase(), option)
        }
        assertTrue("lora" in peftOptions)
    }

    @Test
    fun `every option round-trips through peftOf to a distinct config`() {
        // The guarantee the wire values exist to provide, asserted end to end rather than assumed.
        assertTrue(peftOf("lora", 8, 16) is PeftConfig.Lora)
        assertTrue(peftOf("mars-opt0", 8, 16) is PeftConfig.MarsOpt0)
        assertTrue(peftOf("mars-opt1", 8, 16) is PeftConfig.MarsOpt1)
        assertTrue(peftOf("mars-quantized", 8, 16) is PeftConfig.MarsQuantized)
    }

    @Test
    fun `a display name fed back to peftOf does NOT resolve - which is why options stay wire`() {
        // Not a wish, a demonstration: this is exactly what breaks if the two are conflated.
        val fromDisplay = peftOf(peftDisplayName("mars-opt1"), 8, 16)
        assertTrue("a display name falls through to the LoRA default", fromDisplay is PeftConfig.Lora)
        assertNotEquals(peftOf("mars-opt1", 8, 16)::class, fromDisplay::class)
    }

    @Test
    fun `an unknown method shows its wire value rather than being guessed at`() {
        assertEquals("some-future-method", peftDisplayName("some-future-method"))
    }
}
