package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.config.TrainConfig
import com.martinkorelic.mobiletransformers.constants.MemoryConfigId
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Training defaults to the low-memory allocator profile. Inference does not.
 *
 * ### What `high_perf` actually does to a training run
 *
 * In `session_cache.h` it maps to `EnableMemPattern()` + `EnableCpuMemArena()`. For a forward-only
 * inference session that is the right trade. For training it is not: the memory-pattern planner
 * pre-allocates the whole backward activation plan, and the CPU arena grows to the peak and never
 * gives it back.
 *
 * Measured on an S21 FE (5.5 GB): FunctionGemma-270M — 268,098,176 parameters, ~1.07 GB of fp32
 * weights, 368,640 of them trainable — reached **2.35 GB RSS + 1.02 GB swap** and was SIGKILLed by
 * `lmkd`. Roughly 3x the model, none of which the model needs.
 *
 * Three places had to agree, and each was a separate way to get `high_perf` back:
 *  - `ORTTrainingConfig`'s own default,
 *  - `parseTrainingArguments`, since exported `training_config.json` files carry **no** device
 *    section at all, so its fallback *is* the setting for every real run,
 *  - the public `TrainConfig`, which is what the app passes.
 */
@RunWith(RobolectricTestRunner::class)
class TrainingMemoryProfileTest {

    // FileUtil's parsers take a path and use org.json, which needs Robolectric on the JVM.
    private val dir: File = Files.createTempDirectory("training-memory-profile").toFile()

    @After
    fun cleanup() = dir.deleteRecursively().let { }

    private fun writeConfig(name: String, json: String): String =
        File(dir, name).apply { writeText(json) }.absolutePath

    @Test
    fun theEngineLevelTrainingConfigDefaultsToLowMem() {
        assertEquals(MemoryConfigId.LOW_MEM.wire, ORTTrainingConfig().deviceOptions.memoryConfigId)
    }

    @Test
    fun thePublicTrainConfigDefaultsToLowMem() {
        assertEquals(MemoryConfigId.LOW_MEM, TrainConfig().device.memoryConfigId)
    }

    @Test
    fun thePublicDefaultSurvivesTheMappingToTheEngineConfig() {
        // The app hands a TrainConfig to trainingJob().start(); if the mapper dropped the device
        // options the default would be silently undone one layer down.
        assertEquals(MemoryConfigId.LOW_MEM.wire, TrainConfig().toOrt().deviceOptions.memoryConfigId)
    }

    @Test
    fun anExportedTrainingConfigWithNoDeviceSectionGetsLowMem() {
        // Exactly what `variants/*/train/training_config.json` looks like: requires_grad and PEFT
        // metadata, and nothing about the device. This fallback is the real-world setting.
        val exported = writeConfig(
            "training_config.json",
            """{"requires_grad": ["a.lora_A.lora.weight"], "rank": 8, "alpha": 8}""",
        )

        val parsed = parseTrainingArguments(exported)

        assertEquals(MemoryConfigId.LOW_MEM.wire, parsed.deviceOptions.memoryConfigId)
    }

    @Test
    fun anExplicitHighPerfInTheConfigIsStillHonoured() {
        // The default is a default, not a policy: a caller who wants the arena can have it.
        val declared = writeConfig(
            "training_config.json",
            """{"deviceOptions": {"memoryConfigId": "high_perf", "coreConfigId": "opt1"}}""",
        )

        val parsed = parseTrainingArguments(declared)

        assertEquals(MemoryConfigId.HIGH_PERF.wire, parsed.deviceOptions.memoryConfigId)
    }

    @Test
    fun generationKeepsHighPerf() {
        // The change must not leak into inference, where the arena is a win and no backward plan
        // exists to pre-allocate.
        val generationConfig =
            parseGenerationArguments(writeConfig("generation_config.json", """{"repoName": "m"}"""))

        assertEquals(MemoryConfigId.HIGH_PERF.wire, generationConfig.deviceOptions.memoryConfigId)
    }
}
