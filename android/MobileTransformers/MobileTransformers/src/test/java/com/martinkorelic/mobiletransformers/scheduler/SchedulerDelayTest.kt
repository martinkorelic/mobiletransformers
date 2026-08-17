package com.martinkorelic.mobiletransformers.scheduler

import androidx.work.Data
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

/**
 * The scheduler's start delay survives the trip through WorkManager's `Data`.
 *
 * A scheduled chunk is rebuilt from primitives after process death, so a field the codec does not
 * carry is a field that silently reverts — which for a delay means a run the user deferred starting
 * immediately. `TrainingScheduleConfigCodec` is the seam, and it is only checkable here: the delay
 * itself is applied by `setInitialDelay` inside `enqueueChunk`, which needs a real WorkManager.
 */
class SchedulerDelayTest {

    private fun roundTrip(config: TrainingScheduleConfig): TrainingScheduleConfig {
        val builder = Data.Builder()
        for ((key, value) in TrainingScheduleConfigCodec.toPairs(config)) {
            when (value) {
                is Boolean -> builder.putBoolean(key, value)
                is Int -> builder.putInt(key, value)
                is Long -> builder.putLong(key, value)
                is String -> builder.putString(key, value)
                else -> error("unencodable $key")
            }
        }
        return TrainingScheduleConfigCodec.fromData(builder.build())
    }

    @Test
    fun theDelaySurvivesTheCodec() {
        assertEquals(240L, roundTrip(TrainingScheduleConfig(initialDelayMinutes = 240)).initialDelayMinutes)
    }

    @Test
    fun noDelayIsTheDefaultAndRoundTripsAsZero() {
        assertEquals(0L, TrainingScheduleConfig().initialDelayMinutes)
        assertEquals(0L, roundTrip(TrainingScheduleConfig()).initialDelayMinutes)
    }

    @Test
    fun theOtherFieldsStillRoundTrip() {
        // The codec is hand-written, so adding a key is exactly where the others get dropped.
        val config = TrainingScheduleConfig(
            requiresCharging = false,
            requiresDeviceIdle = true,
            maxRuntimeMinutes = 12,
            maxStepsPerChunk = 7,
            checkpointEverySteps = 3,
            totalSteps = 99,
            initialDelayMinutes = 60,
        )

        assertEquals(config, roundTrip(config))
    }

    @Test
    fun aNegativeDelayIsRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            TrainingScheduleConfig(initialDelayMinutes = -1)
        }
    }
}
