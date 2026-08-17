package com.martinkorelic.mobiletransformers.training

import com.google.gson.Gson
import com.martinkorelic.mobiletransformers.TrainingState
import java.io.File

/**
 * Read-only projection of `train/training_state.json` (#18). **No format change** — it parses the existing
 * [TrainingState] (`{ schedulerState, currentGlobalStep, currentEpoch }`) written by
 * `ORTTrainerNative.saveTrainingState` and projects the fields the public surface exposes. Reading never
 * rewrites the file.
 */
data class CheckpointInfo(
    val currentGlobalStep: Int,
    val currentEpoch: Int,
    val schedulerStep: Int,
    val totalSteps: Int,
    val checkpointDirPath: String,
    val stateJsonPath: String,
    val exists: Boolean,
) {
    companion object {
        private val gson = Gson()

        /**
         * Project the checkpoint at [checkpointDirPath] with its sibling [stateJsonPath]. If the state file
         * is absent, returns a projection with [exists] = false and zeroed counters.
         */
        fun read(checkpointDirPath: String, stateJsonPath: String): CheckpointInfo {
            val file = File(stateJsonPath)
            if (!file.isFile) {
                return CheckpointInfo(0, 0, 0, 0, checkpointDirPath, stateJsonPath, exists = false)
            }
            val state = gson.fromJson(file.readText(), TrainingState::class.java)
            return CheckpointInfo(
                currentGlobalStep = state.currentGlobalStep,
                currentEpoch = state.currentEpoch,
                schedulerStep = state.schedulerState.currentStep,
                totalSteps = state.schedulerState.totalSteps,
                checkpointDirPath = checkpointDirPath,
                stateJsonPath = stateJsonPath,
                exists = true,
            )
        }
    }
}
