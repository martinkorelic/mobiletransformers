package com.martinkorelic.mobiletransformers

import kotlin.math.*

interface LearningRateScheduler {
    fun step(): Float  // Return the new learning rate
    fun getLR(): Float
    fun reset() // Reset scheduler state
    fun loadFromState(state: SchedulerState)
    fun stateDict() : SchedulerState
}

data class SchedulerState(
    val totalSteps: Int,
    val warmupSteps: Int,
    val minLr: Float,
    val initialLr: Float,
    val currentStep: Int
)

class CosineLRScheduler(
    private val totalSteps: Int,
    private val warmupSteps: Int = 0,
    private val minLr: Float = 0f,
    private val initialLr: Float = 1e-3f
) : LearningRateScheduler {

    private var currentStep = 0
    private var currentLR = if (warmupSteps > 0) 0f else initialLr

    override fun step(): Float {
        currentStep++
        return stepInternal(increment = false) // Already incremented above
    }

    private fun stepInternal(increment: Boolean = true): Float {
        if (increment) {
            currentStep++
        }

        val newLr = when {
            currentStep <= warmupSteps -> {
                // Linear warmup: Increase LR from 0 to initial_lr
                if (warmupSteps == 0) {
                    initialLr
                } else {
                    (initialLr * currentStep.toFloat()) / warmupSteps.toFloat()
                }
            }
            else -> {
                // Cosine decay after warmup
                val decayStep = currentStep - warmupSteps
                val decayTotal = totalSteps - warmupSteps

                // Handle edge case where decayTotal might be 0
                if (decayTotal <= 0) {
                    minLr
                } else {
                    val cosDecay = 0.5f * (1f + kotlin.math.cos(kotlin.math.PI * decayStep.toDouble() / decayTotal.toDouble())).toFloat()
                    minLr + (initialLr - minLr) * cosDecay
                }
            }
        }

        // Ensure LR doesn't go below min_lr
        currentLR = max(newLr, minLr)
        return currentLR
    }

    override fun getLR(): Float = currentLR

    override fun reset() {
        currentStep = 0
        currentLR = if (warmupSteps > 0) 0f else initialLr
    }

    override fun loadFromState(state: SchedulerState) {
        this.currentStep = state.currentStep
        this.currentLR = calculateLR(currentStep)
    }

    // Helper method to calculate LR without incrementing step
    private fun calculateLR(step: Int): Float {
        val newLr = when {
            step <= warmupSteps -> {
                // Linear warmup: Increase LR from 0 to initial_lr
                if (warmupSteps == 0) {
                    initialLr
                } else {
                    (initialLr * step.toFloat()) / warmupSteps.toFloat()
                }
            }
            else -> {
                // Cosine decay after warmup
                val decayStep = step - warmupSteps
                val decayTotal = totalSteps - warmupSteps

                // Handle edge case where decayTotal might be 0
                if (decayTotal <= 0) {
                    minLr
                } else {
                    val cosDecay = 0.5f * (1f + cos(PI * decayStep.toDouble() / decayTotal.toDouble())).toFloat()
                    minLr + (initialLr - minLr) * cosDecay
                }
            }
        }

        return max(newLr, minLr)
    }

    override fun stateDict(): SchedulerState {
        return SchedulerState(
            totalSteps = totalSteps,
            warmupSteps = warmupSteps,
            minLr = minLr,
            initialLr = initialLr,
            currentStep = currentStep
        )
    }
}


class LinearLRScheduler(
    private val baseLr: Float,
    private val startFactor: Float = 1.0f,
    private val endFactor: Float = 1.0f / 3.0f,
    private val totalIters: Int = 5
) : LearningRateScheduler {

    private var currentStep = 0
    private var currentLR = baseLr * startFactor

    override fun step(): Float {
        val factor = when {
            currentStep >= totalIters -> endFactor
            totalIters <= 1 -> endFactor
            else -> {
                // Linear interpolation between startFactor and endFactor
                val progress = currentStep.toFloat() / (totalIters - 1).toFloat()
                startFactor + (endFactor - startFactor) * progress
            }
        }

        currentLR = baseLr * factor
        currentStep++
        return currentLR
    }

    override fun getLR(): Float = currentLR

    override fun reset() {
        currentStep = 0
        currentLR = baseLr * startFactor
    }

    override fun loadFromState(state: SchedulerState) {
        TODO("Not yet implemented")
    }

    override fun stateDict(): SchedulerState {
        TODO("Not yet implemented")
    }
}