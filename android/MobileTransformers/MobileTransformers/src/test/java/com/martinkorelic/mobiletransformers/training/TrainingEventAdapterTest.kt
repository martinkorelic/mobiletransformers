package com.martinkorelic.mobiletransformers.training

import com.martinkorelic.mobiletransformers.TrainingProgress
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** #18: the callback->status/event mapping is one-to-one and ordered (pure, no native handle). */
class TrainingEventAdapterTest {

    private fun progress(step: Int) =
        TrainingProgress(
            currentStep = step,
            currentEpoch = 0,
            totalLoss = 1.5f,
            epochLoss = 1.5f,
            stepLoss = 0.5f,
            learningRate = 1e-4f,
            stepDurationMs = 10,
            epochDurationMs = 100,
            totalDurationMs = 1000,
        )

    @Test
    fun statusTransitionsFollowCallbacks() {
        val adapter = TrainingEventAdapter()
        assertEquals(TrainingStatus.Idle, adapter.status.value)

        adapter.onModelLoadStart()
        assertEquals(TrainingStatus.Preparing, adapter.status.value)

        adapter.onStepEnd(progress(1))
        assertTrue(adapter.status.value is TrainingStatus.Running)

        adapter.onMergeStart(progress(2))
        assertEquals(TrainingStatus.Merging, adapter.status.value)

        adapter.onSaveModelStart(progress(2))
        assertEquals(TrainingStatus.Saving, adapter.status.value)

        adapter.onCompletion(progress(2))
        val completed = adapter.status.value
        assertTrue(completed is TrainingStatus.Completed)
        completed as TrainingStatus.Completed
        assertEquals(2, completed.result.finalStep)
    }

    @Test
    fun eventStreamOrderMatchesScriptedCallbacks() = runBlocking {
        val adapter = TrainingEventAdapter()
        val collected = mutableListOf<TrainingEvent>()
        val collector = launch { adapter.events.collect { collected += it } }
        yield() // let the collector subscribe before we drive callbacks

        adapter.onDataLoadEnd(totalSteps = 4, stepsPerEpoch = 2)
        adapter.onStepEnd(progress(1))
        adapter.onOptimizerStep(progress(1))
        adapter.onEpochEnd(progress(2))
        adapter.onMergeStart(progress(2))
        adapter.onMergeEnd(progress(2))
        adapter.onSaveModelEnd(progress(2))
        adapter.onCompletion(progress(2))
        yield() // let the collector drain the buffered emissions

        collector.cancel()

        assertEquals(
            listOf(
                "DataLoaded",
                "Step",
                "OptimizerStep",
                "Epoch",
                "MergeStarted",
                "MergeFinished",
                "Saved",
                "Done",
            ),
            collected.map { it::class.simpleName },
        )
        // merge fired, so the completed result records merged = true
        val done = collected.last() as TrainingEvent.Done
        assertTrue(done.result.merged)
    }

    @Test
    fun errorTransitionsToFailed() {
        val adapter = TrainingEventAdapter()
        adapter.onError(IllegalStateException("boom"))
        val s = adapter.status.value
        assertTrue(s is TrainingStatus.Failed)
        assertEquals("boom", (s as TrainingStatus.Failed).error.message)
    }
}
