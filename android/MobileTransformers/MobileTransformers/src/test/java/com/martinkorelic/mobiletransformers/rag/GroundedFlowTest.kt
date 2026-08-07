package com.martinkorelic.mobiletransformers.rag

import com.martinkorelic.mobiletransformers.runtime.GroundedResult
import com.martinkorelic.mobiletransformers.runtime.RetrievalMatch
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #27: the grounded composition — retrieve (over InMemoryVectorStore) → assemble → generate — asserting
 * the result carries the retrieved matches and the exact prompt handed to generation. Uses a fake
 * generate lambda so it runs on the JVM with no JNI.
 */
class GroundedFlowTest {

    private val dim = 64
    private fun unit(vararg on: Int): FloatArray = FloatArray(dim) { if (it in on) 1f else 0f }

    @Test
    fun groundedResultCarriesMatchesAndExactPrompt() {
        val store = InMemoryVectorStore(dim)
        store.insert(RagDocument("d1", "D1", "the sky is blue"), unit(0))
        store.insert(RagDocument("d2", "D2", "grass is green"), unit(1))

        // retrieve leg over the real VectorStore boundary
        val matches = store.search(unit(0), topK = 2, minScore = 0.0)
            .map { RetrievalMatch(it.document.text, it.score) }
        assertTrue(matches.isNotEmpty())

        // assemble + (fake) generate
        val prompt = PromptAssembler.assemble("what color is the sky?", matches)
        var seenPrompt: String? = null
        val fakeGenerate: (String) -> String = { p -> seenPrompt = p; "The sky is blue." }
        val result = GroundedResult(text = fakeGenerate(prompt), matches = matches, prompt = prompt)

        assertEquals(prompt, seenPrompt) // generation received the assembled prompt
        assertEquals("The sky is blue.", result.text)
        assertEquals(matches, result.matches)
        assertTrue(result.prompt.contains("- the sky is blue"))
        assertTrue(result.prompt.contains("Question: what color is the sky?"))
    }
}
