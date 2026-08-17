package com.martinkorelic.mobiletransformers.rag

import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** #27: default grounded prompt template + caller override. */
class PromptAssemblerTest {

    private val matches = listOf(
        RetrievalMatch("the sky is blue", 0.9),
        RetrievalMatch("grass is green", 0.7),
    )

    @Test
    fun defaultTemplateIncludesContextBulletsAndQuery() {
        val prompt = PromptAssembler.assemble("what color is the sky?", matches)
        assertTrue(prompt.contains("Context:"))
        assertTrue(prompt.contains("- the sky is blue"))
        assertTrue(prompt.contains("- grass is green"))
        assertTrue(prompt.contains("Question: what color is the sky?"))
        assertTrue(prompt.trimEnd().endsWith("Answer:"))
        // bullet order follows match order
        assertTrue(prompt.indexOf("- the sky is blue") < prompt.indexOf("- grass is green"))
    }

    @Test
    fun callerOverrideReplacesTemplate() {
        val custom = PromptStrategy { query, m -> "Q=$query|N=${m.size}" }
        assertEquals("Q=hi|N=2", PromptAssembler.assemble("hi", matches, custom))
    }

    @Test
    fun emptyMatchesStillProducesInspectablePrompt() {
        val prompt = PromptAssembler.assemble("q", emptyList())
        assertTrue(prompt.contains("Question: q"))
    }
}
