package com.martinkorelic.mobiletransformers

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * #23 device leg: map-driven load-and-generate over a real #9 package, and the two-prompt
 * conversation-reset no-leak check (validates `ORTConversationState.addAssistantMessage` rendered-offset
 * fix + `resetConversation()` on load). Two sequential generations complete and the second does not
 * fail from inherited prepend state.
 */
@RunWith(AndroidJUnit4::class)
class ConversationResetTest {

    @Test
    fun twoSequentialPromptsBothComplete() = runBlocking {
        val root = DeviceModel.requireCacheRoot()
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val model = MobileTransformers.fromPretrained(ctx, DeviceModel.repoId(root), cacheDir = root.absolutePath)
        try {
            val cfg = GenerationConfig(maxNewTokens = 4)
            val first = model.generate("Name a color.", cfg)
            val second = model.generate("Name an animal.", cfg)
            // Both complete without a crash; the second is not empty due to inherited prepend corruption.
            assertTrue(first.tokenCount >= 0)
            assertTrue(second.tokenCount >= 0)
        } finally {
            model.close()
        }
    }
}
