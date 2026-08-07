package com.martinkorelic.mobiletransformers

import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * #6/#10: config parsing is fail-closed on every closed-set field (JVM, Robolectric — no device).
 *
 * These parsers were untestable before Robolectric was on the classpath: `FileUtil` uses
 * `org.json.JSONObject`, which throws "Method ... not mocked" under plain JUnit. That gap is why an
 * unknown `schedulerType` silently defaulted to Linear (with a `println` that release builds drop)
 * for as long as it did — no test could reach the code.
 */
@RunWith(RobolectricTestRunner::class)
class FileUtilParseTest {

    private val dir: File = Files.createTempDirectory("fileutil-parse").toFile()

    @After
    fun cleanup() {
        dir.deleteRecursively()
    }

    private fun writeConfig(name: String, json: String): String =
        File(dir, name).apply { writeText(json) }.absolutePath

    // --- scheduler -------------------------------------------------------------------------------

    @Test
    fun linearSchedulerParsesItsOwnOptions() {
        val path = writeConfig(
            "train.json",
            """{"schedulerType":"linear","schedulerOptions":{"learningRate":0.001,"startFactor":1.0,"endFactor":0.5}}""",
        )
        val config = parseTrainingArguments(path)
        val scheduler = config.schedulerConfig as SchedulerConfig.Linear
        assertEquals(0.001f, scheduler.learningRate, 1e-6f)
        assertEquals(0.5f, scheduler.endFactor, 1e-6f)
    }

    @Test
    fun cosineSchedulerParsesItsOwnOptions() {
        val path = writeConfig(
            "train.json",
            """{"schedulerType":"cosine","schedulerOptions":{"learningRate":0.002,"warmupSteps":25}}""",
        )
        val scheduler = parseTrainingArguments(path).schedulerConfig as SchedulerConfig.Cosine
        assertEquals(0.002f, scheduler.learningRate, 1e-6f)
        assertEquals(25, scheduler.warmupSteps)
    }

    /** The regression: an unknown scheduler used to become Linear with an invisible warning. */
    @Test
    fun unknownSchedulerTypeFailsClosed() {
        val path = writeConfig("train.json", """{"schedulerType":"consine"}""")
        assertThrows(IllegalStateException::class.java) { parseTrainingArguments(path) }
    }

    /** Root-level scheduler options are the backward-compat path; it must fail closed identically. */
    @Test
    fun unknownSchedulerTypeFailsClosedOnTheRootLevelPath() {
        val path = writeConfig("train.json", """{"schedulerType":"exponential","learningRate":0.001}""")
        assertThrows(IllegalStateException::class.java) { parseTrainingArguments(path) }
    }

    // --- device options --------------------------------------------------------------------------

    @Test
    fun deviceOptionsParseFromEitherNestedOrRootLevel() {
        val nested = writeConfig(
            "a.json",
            """{"deviceOptions":{"coreConfigId":"opt2","memoryConfigId":"low_mem","executionProvider":"nnapi"}}""",
        )
        val root = writeConfig(
            "b.json",
            """{"coreConfigId":"opt2","memoryConfigId":"low_mem","executionProvider":"nnapi"}""",
        )
        for (path in listOf(nested, root)) {
            val opts = parseTrainingArguments(path).deviceOptions
            assertEquals("opt2", opts.coreConfigId)
            assertEquals("low_mem", opts.memoryConfigId)
            assertEquals("nnapi", opts.executionProvider)
        }
    }

    @Test
    fun unknownExecutionProviderFailsClosed() {
        val path = writeConfig("train.json", """{"deviceOptions":{"executionProvider":"cuda"}}""")
        assertThrows(IllegalStateException::class.java) { parseTrainingArguments(path) }
    }

    @Test
    fun unknownMemoryConfigIdFailsClosed() {
        val path = writeConfig("train.json", """{"deviceOptions":{"memoryConfigId":"turbo"}}""")
        assertThrows(IllegalStateException::class.java) { parseTrainingArguments(path) }
    }

    // --- generation + rag ------------------------------------------------------------------------

    @Test
    fun unknownSamplingMethodFailsClosed() {
        val path = writeConfig("gen.json", """{"sampling":{"method":"beam"}}""")
        assertThrows(IllegalStateException::class.java) { parseGenerationArguments(path) }
    }

    @Test
    fun unknownSearchTypeFailsClosed() {
        val path = writeConfig("rag.json", """{"searchType":"hybrid"}""")
        assertThrows(IllegalStateException::class.java) { parseRagArguments(path) }
    }

    @Test
    fun unknownIndexingModeFailsClosed() {
        val path = writeConfig("rag.json", """{"indexingMode":"streaming"}""")
        assertThrows(IllegalStateException::class.java) { parseRagArguments(path) }
    }

    @Test
    fun validRagFieldsRoundTrip() {
        val path = writeConfig(
            "rag.json",
            """{"searchType":"text","indexingMode":"precompute","topK":7,"minScore":0.25}""",
        )
        val config = parseRagArguments(path)
        assertEquals("text", config.searchType)
        assertEquals("precompute", config.indexingMode)
        assertEquals(7, config.topK)
        assertEquals(0.25, config.minScore, 1e-9)
    }

    /** An absent/blank path is not an error — it yields defaults (unchanged behavior). */
    @Test
    fun absentConfigYieldsDefaults() {
        assertEquals(ORTTrainingConfig(), parseTrainingArguments(""))
        assertEquals(ORTRagConfig(), parseRagArguments(File(dir, "nope.json").absolutePath))
    }
}
