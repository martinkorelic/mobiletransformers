package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #23 device leg: map-driven load-and-generate over a real #9 package, plus the multi-turn
 * conversation check (validates `ORTConversationState.addAssistantMessage` rendered-offset fix +
 * `resetConversation()` on load + the KV-cache/position-ids continuation in `GenerationInputs`).
 *
 * **Three turns, not two, and the assertions are non-vacuous.** The previous version asserted
 * `tokenCount >= 0` on two turns, which is true of every possible outcome — it could only ever catch a
 * throw. It did catch one (the 4.57.6 `Gather` out-of-bounds), but an off-by-N in the position ids that
 * merely *degrades* output would have passed silently, and a two-turn test can pass on an off-by-N that
 * compounds only from the third turn onward.
 */
@RunWith(AndroidJUnit4::class)
class ConversationResetTest {

    private companion object {
        const val MAX_NEW_TOKENS = 4
    }

    @Test
    fun threeSequentialPromptsEachEmitTheRequestedTokens() = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val model = MobileTransformers.fromPretrained(ctx, repoId, cacheDir = root.absolutePath)
        try {
            val cfg = GenerationConfig(maxNewTokens = MAX_NEW_TOKENS)
            val prompts = listOf("Name a color.", "Name an animal.", "Name a country.")

            prompts.forEachIndexed { index, prompt ->
                val turn = index + 1
                val result = model.generate(prompt, cfg)

                // A turn that inherits a corrupted prefix either throws (what 4.57.6 did) or returns
                // early with nothing. Both are failures, and both are invisible to `tokenCount >= 0`.
                assertTrue(
                    "turn $turn produced blank text — the conversation state or the KV cache is corrupt",
                    result.text.isNotBlank(),
                )
                // #24 locked maxNewTokens as an EXCLUSIVE bound: N means exactly N tokens, unless the
                // model emits EOS first (legitimate, and it truncates rather than over-runs).
                assertTrue(
                    "turn $turn emitted ${result.tokenCount} tokens, more than the requested $MAX_NEW_TOKENS",
                    result.tokenCount <= MAX_NEW_TOKENS,
                )
                assertTrue(
                    "turn $turn emitted no tokens at all",
                    result.tokenCount > 0,
                )
            }
        } finally {
            model.close()
        }
    }

    @Test
    fun aFreshSessionDoesNotInheritThePreviousConversation() = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val repoId = DeviceModel.repoId(root)
        DeviceModel.requireDecoder(root, repoId)
        // SamplingConfig defaults to GREEDY, so this is deterministic without spelling it out.
        val cfg = GenerationConfig(maxNewTokens = MAX_NEW_TOKENS)

        // Same prompt, same greedy config, two independent sessions. `load()` calls
        // `resetConversation()`, so the second session must reproduce the first exactly. If any
        // conversation or KV state survived `close()`, the continuation would differ.
        val first = generateOnce(ctx, repoId, root.absolutePath, cfg)
        val second = generateOnce(ctx, repoId, root.absolutePath, cfg)

        assertEquals("a fresh session did not start from a clean conversation state", first, second)
    }

    /** One prompt through a session opened and closed for it alone. */
    private suspend fun generateOnce(
        ctx: android.content.Context,
        repoId: String,
        cacheDir: String,
        cfg: GenerationConfig,
    ): String {
        val model = MobileTransformers.fromPretrained(ctx, repoId, cacheDir = cacheDir)
        try {
            return model.generate("Name a color.", cfg).text
        } finally {
            model.close()
        }
    }
}
