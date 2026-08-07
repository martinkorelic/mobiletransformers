package com.martinkorelic.mobiletransformers.training

import com.google.gson.Gson
import com.martinkorelic.mobiletransformers.SchedulerState
import com.martinkorelic.mobiletransformers.TrainingState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** #18: CheckpointInfo is a read-only projection of training_state.json — format unchanged after reading. */
class CheckpointInfoTest {

    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun projectsStateAndPreservesFormat() {
        val state =
            TrainingState(
                schedulerState =
                    SchedulerState(
                        totalSteps = 200,
                        warmupSteps = 10,
                        minLr = 0f,
                        initialLr = 1e-4f,
                        currentStep = 120,
                    ),
                currentGlobalStep = 120,
                currentEpoch = 2,
            )
        val train = tmp.newFolder("train")
        val stateFile = train.resolve("training_state.json")
        val json = Gson().toJson(state)
        stateFile.writeText(json)

        val info =
            CheckpointInfo.read(train.resolve("checkpoint").absolutePath, stateFile.absolutePath)

        assertTrue(info.exists)
        assertEquals(120, info.currentGlobalStep)
        assertEquals(2, info.currentEpoch)
        assertEquals(120, info.schedulerStep)
        assertEquals(200, info.totalSteps)
        // reading must not rewrite the file (format preserved).
        assertEquals(json, stateFile.readText())
    }

    @Test
    fun absentStateFileProjectsExistsFalse() {
        val info = CheckpointInfo.read("/nope/checkpoint", "/nope/training_state.json")
        assertFalse(info.exists)
        assertEquals(0, info.currentGlobalStep)
    }
}
