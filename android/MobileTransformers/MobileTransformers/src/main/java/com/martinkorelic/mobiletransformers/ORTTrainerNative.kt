package com.martinkorelic.mobiletransformers

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.util.Log
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.martinkorelic.mobiletransformers.repository.TrainingCallback
import java.io.File
import java.io.IOException
import com.martinkorelic.mobiletransformers.packages.PackagePaths

data class TrainingState(
    val schedulerState: SchedulerState,
    val currentGlobalStep: Int,
    val currentEpoch: Int
)

class ORTTrainerNative(private val context: Context, private val cacheDirPath: String, private var tokenizer: ORTTokenizerNative, private val trainingConfig: ORTTrainingConfig) {

    init {
        NativeLibrary.ensureLoaded()
    }


    private val LOG_TAG = "ORTTrainerNative"

    // Pointer to the model
    var model : Long = 0;
    var scheduler : LearningRateScheduler? = null

    // G2: one resolver for this package's stages; nothing below appends a stage name to a string.
    private val pkgPaths = PackagePaths.forCache(cacheDirPath, trainingConfig.repoName)
    private val trainDir = pkgPaths.train.absolutePath

    val trainingModelPath = "$trainDir/training_model.onnx"
    val evalModelPath = "$trainDir/eval_model.onnx"
    val checkpointPath = "$trainDir/checkpoint"
    val optimizerPath = "$trainDir/optimizer_model.onnx"

    val dataCurator = ORTDataCurator(tokenizer, "$trainDir/${trainingConfig.datasetOptions.trainFile}", trainingConfig.batchSize, trainingConfig.datasetOptions.maxSequenceLength, trainingConfig.datasetOptions.removeLongSamples, trainingConfig.datasetOptions.maxDatasetLength, trainingConfig.datasetOptions.datasetBatchSize, getPreprocessFunctionForTask(trainingConfig.taskName, trainingConfig.customPreprocess))
    private val dataCollator = DataCollatorForSupervisedDataset(tokenizer)

    // Training state
    var trainingState : TrainingState? = null

    // Training variables tracking
    var totalSteps : Int = 0
    var globalStep : Int = 0
    var epoch : Int = 0
    var accumulatedLoss = 0f

    // Cooperative cancellation (#18): TrainingJob.cancel(...) sets this; the step/epoch loops check it and
    // break cleanly so the existing saveModel + saveTrainingState path can run. Not a loop-logic change.
    @Volatile
    var cancelRequested : Boolean = false


    // Training metrics

    data class TrainingStepMetrics(
        val step: Int,
        val epoch: Int,
        val loss: Float,
        val learningRate: Float,
        val stepDurationMs: Long,
        val memoryUsageMB: Long
    )

    data class TrainingSummary(
        val train_runtime_seconds: Float,
        val train_steps_per_second: Float,
        val train_samples_per_second: Float,
        val total_steps: Int,
        val total_samples: Int,
        val final_loss: Float,
        val peak_memory_mb: Long,
        val average_memory_mb: Float
    ) {
        /** #17: convert to the public mirror so no `ORT*` type reaches the facade surface. */
        fun toPublic(): com.martinkorelic.mobiletransformers.runtime.TrainingSummary =
            com.martinkorelic.mobiletransformers.runtime.TrainingSummary(
                trainRuntimeSeconds = train_runtime_seconds,
                trainStepsPerSecond = train_steps_per_second,
                trainSamplesPerSecond = train_samples_per_second,
                totalSteps = total_steps,
                totalSamples = total_samples,
                finalLoss = final_loss,
                peakMemoryMb = peak_memory_mb,
                averageMemoryMb = average_memory_mb,
            )
    }

    private val trainingMetrics = mutableListOf<TrainingStepMetrics>()

    /**
     * The run-level summary produced by [saveTrainingLogs], or null.
     *
     * Deliberately null unless `trainingConfig.profileMetrics` is on: the summary is computed from the
     * per-step metric samples, which are only collected under that flag. Reporting zeros instead would
     * be worse than reporting nothing.
     */
    @Volatile
    var lastSummary: TrainingSummary? = null
        private set

    init {
        val requiresGrad = loadTrainableLayerNamesJSON("$trainDir/training_config.json")

        if (requiresGrad == null) {
            Log.e(LOG_TAG, "No training config provided. Model cannot be initialized.")
        } else {
            Log.d(LOG_TAG, "Loading the training model...")
            model = createTrainingSession(checkpointPath, trainingModelPath, evalModelPath, optimizerPath, cacheDirPath, requiresGrad, trainingConfig.deviceOptions.memoryConfigId, trainingConfig.deviceOptions.coreConfigId, trainingConfig.deviceOptions.executionProvider, trainingConfig.deviceOptions.enableProfiling)
            Log.d(LOG_TAG, "Successfully created the training model. Native handle at $model")
        }

        // Load training state if available
        if (trainingConfig.loadFromState)
            trainingState = loadTrainingState("$trainDir/training_state.json")
    }

    private fun loadTrainingState(trainingStatePath: String): TrainingState? {
        return try {
            val trainingStateFile = File(trainingStatePath)
            if (trainingStateFile.exists()) {
                Log.d(LOG_TAG, "Loading training state from: $trainingStatePath")
                val jsonContent = trainingStateFile.readText()
                val gson = Gson()
                val trainingState = gson.fromJson(jsonContent, TrainingState::class.java)

                Log.d(LOG_TAG, "Training state loaded successfully:")
                Log.d(LOG_TAG, "  - Current epoch: ${trainingState.currentEpoch}")
                Log.d(LOG_TAG, "  - Current global step: ${trainingState.currentGlobalStep}")
                Log.d(LOG_TAG, "  - Scheduler current step: ${trainingState.schedulerState.currentStep}")

                trainingState
            } else {
                Log.d(LOG_TAG, "No existing training state found at: $trainingStatePath")
                Log.d(LOG_TAG, "Starting fresh training session")
                null
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error loading training state: ${e.message}")
            Log.e(LOG_TAG, "Starting fresh training session")
            null
        }
    }

    private fun saveTrainingState(globalStep: Int, epoch: Int, scheduler: LearningRateScheduler): Boolean {
        return try {
            val trainingStatePath = "$trainDir/training_state.json"
            val trainingStateFile = File(trainingStatePath)

            // Ensure the directory exists
            trainingStateFile.parentFile?.mkdirs()

            // Create training state object
            val currentTrainingState = TrainingState(
                schedulerState = scheduler.stateDict(),
                currentGlobalStep = globalStep,
                currentEpoch = epoch
            )

            // Convert to JSON with pretty printing
            val gson = GsonBuilder()
                .setPrettyPrinting()
                .create()
            val jsonContent = gson.toJson(currentTrainingState)

            // Write to file
            trainingStateFile.writeText(jsonContent)

            Log.d(LOG_TAG, "Training state saved successfully:")
            Log.d(LOG_TAG, "  - Global step: $globalStep")
            Log.d(LOG_TAG, "  - Epoch: $epoch")
            Log.d(LOG_TAG, "  - Scheduler step: ${scheduler.stateDict().currentStep}")
            Log.d(LOG_TAG, "  - File: $trainingStatePath")

            true
        } catch (e: IOException) {
            Log.e(LOG_TAG, "Failed to save training state: ${e.message}")
            false
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Unexpected error saving training state: ${e.message}")
            false
        }
    }

    fun createScheduler() : LearningRateScheduler {
        return when (trainingConfig.schedulerConfig) {
            is SchedulerConfig.Cosine -> CosineLRScheduler(
                totalSteps = totalSteps,
                warmupSteps = trainingConfig.schedulerConfig.warmupSteps,
                minLr = trainingConfig.schedulerConfig.minLearningRate,
                initialLr = trainingConfig.schedulerConfig.learningRate
            )

            is SchedulerConfig.Linear -> LinearLRScheduler(
                baseLr = trainingConfig.schedulerConfig.learningRate,
                startFactor = trainingConfig.schedulerConfig.startFactor,
                endFactor = trainingConfig.schedulerConfig.endFactor,
                totalIters = (totalSteps / trainingConfig.gradAccumSteps - 1) / trainingConfig.gradAccumSteps
            )

        }
    }

    fun startTraining(callback: TrainingCallback? = null) {
        if (model == 0L) {
            Log.e(LOG_TAG, "No native training model has been created.")
            return
        }

        try {

            callback?.onDataLoadStart()

            // Get the actual number of samples in the dataset (already calculated during initialization)
            val actualDatasetSize = dataCurator.getDatasetSize()

            // Calculate steps per epoch correctly using trainingConfig.batchSize
            // Each step processes one batch, so steps per epoch = ceil(dataset_size / batch_size)
            val stepsPerEpoch = kotlin.math.ceil(actualDatasetSize.toDouble() / trainingConfig.batchSize.toDouble()).toInt()

            // Calculate total steps
            totalSteps = trainingConfig.maxSteps ?: (trainingConfig.numTrainEpochs * stepsPerEpoch)

            callback?.onDataLoadEnd(totalSteps, stepsPerEpoch)

            scheduler = createScheduler()

            // Restore training state if available
            if (trainingState != null) {

                Log.i(LOG_TAG, "Restoring training from previous state...")
                Log.i(LOG_TAG, "  - Previous epoch: ${trainingState!!.currentEpoch}")
                Log.i(LOG_TAG, "  - Previous global step: ${trainingState!!.currentGlobalStep}")
                Log.i(LOG_TAG, "  - Previous scheduler step: ${trainingState!!.schedulerState.currentStep}")

                // Restore scheduler state
                scheduler!!.loadFromState(trainingState!!.schedulerState)

                // Restore global step and epoch
                globalStep = trainingState!!.currentGlobalStep
                epoch = trainingState!!.currentEpoch
            }

            Log.i(LOG_TAG, "Training for $totalSteps steps...")

            var totalLoss = 0f
            val trainingStartTime = System.currentTimeMillis()
            var currentTrainingTime : Long = 0

            while (epoch < trainingConfig.numTrainEpochs && globalStep < totalSteps) {

                if (cancelRequested) {
                    Log.i(LOG_TAG, "Training cancelled at epoch $epoch (global step: $globalStep)")
                    break
                }

                Log.i(LOG_TAG, "Starting epoch $epoch (global step: $globalStep)")

                val epochStartTime = System.currentTimeMillis()
                callback?.onEpochStart(
                    TrainingProgress(
                        currentStep = globalStep,
                        currentEpoch = epoch,
                        stepLoss = 0f,
                        epochLoss = totalLoss / globalStep,
                        learningRate = scheduler!!.getLR(),
                        stepDurationMs = 0,
                        epochDurationMs = 0,
                        totalDurationMs = System.currentTimeMillis() - trainingStartTime
                    )
                )

                var epochLoss = 0f

                // Reset data curator
                dataCurator.reset()
                val dataloader = dataCurator.getBatchedDataset()

                for (batch in dataloader) {

                    // Cooperative cancel (#18): break out so the outer while sees the flag and exits cleanly.
                    if (cancelRequested) {
                        break
                    }

                    // End the training if global steps have been reached
                    if (globalStep >= totalSteps) {
                        break
                    }

                    // Skip batches if we're resuming in the middle of an epoch
                    if (trainingState != null && epoch == trainingState!!.currentEpoch && globalStep < trainingState!!.currentGlobalStep) {
                        // We need to skip this batch to align with the restored state
                        // This handles cases where we resume mid-epoch
                        continue
                    }

                    callback?.onStepStart(
                        TrainingProgress(
                            currentStep = globalStep,
                            currentEpoch = epoch,
                            stepLoss = 0f,
                            epochLoss = totalLoss / globalStep,
                            learningRate = scheduler!!.getLR(),
                            stepDurationMs = 0,
                            epochDurationMs = 0,
                            totalDurationMs = System.currentTimeMillis() - trainingStartTime
                        )
                    )

                    val stepStartTime = System.currentTimeMillis()

                    // Perform training
                    val loss = performTrainStep(batch)



                    accumulatedLoss += loss
                    epochLoss += loss
                    totalLoss += loss

                    if (globalStep != 0 && globalStep % trainingConfig.gradAccumSteps == 0) {

                        // Set the model learning rate
                        setLearningRate(model, scheduler!!.getLR())

                        // Backward step + optimizer
                        optimizerStep(model)

                        val avgLoss = accumulatedLoss / trainingConfig.gradAccumSteps

                        callback?.onOptimizerStep(
                            TrainingProgress(
                                currentStep = globalStep,
                                currentEpoch = epoch,
                                stepLoss = avgLoss,
                                epochLoss = totalLoss / globalStep,
                                learningRate = scheduler!!.getLR(),
                                stepDurationMs = 0,
                                epochDurationMs = 0,
                                totalDurationMs = System.currentTimeMillis() - trainingStartTime
                            )
                        )

                        Log.i(LOG_TAG, "Optimizer gradient step")
                        Log.i(LOG_TAG, "Step $globalStep - Loss: $avgLoss")
                        Log.i(LOG_TAG, "LR: ${scheduler?.getLR()}")
                        accumulatedLoss = 0f
                    }

                    val stepEndTime = System.currentTimeMillis()

                    // Get new learning rate from scheduler and set it
                    scheduler!!.step()

                    // Save the model
                    if (globalStep != 0 && globalStep % trainingConfig.saveSteps == 0) {
                        saveModel(model, true)
                        saveTrainingState(globalStep + 1, epoch, scheduler!!)
                    }

                    currentTrainingTime = System.currentTimeMillis()

                    if (trainingConfig.profileMetrics) {
                        val memoryUsage = getTotalMemoryUsageMB()
                        val stepMetrics = TrainingStepMetrics(
                            step = globalStep,
                            epoch = epoch,
                            loss = loss,
                            learningRate = scheduler!!.getLR(),
                            stepDurationMs = stepEndTime - stepStartTime,
                            memoryUsageMB = memoryUsage
                        )
                        trainingMetrics.add(stepMetrics)
                    }

                    callback?.onStepEnd(
                        TrainingProgress(
                            currentStep = globalStep,
                            currentEpoch = epoch,
                            stepLoss = loss,
                            epochLoss = epochLoss / epoch,
                            learningRate = scheduler!!.getLR(),
                            stepDurationMs = stepEndTime - stepStartTime,
                            epochDurationMs = currentTrainingTime - epochStartTime,
                            totalDurationMs = currentTrainingTime - trainingStartTime
                        )
                    )
                    globalStep++
                }

                currentTrainingTime = System.currentTimeMillis()
                callback?.onEpochEnd(
                    TrainingProgress(
                        currentStep = globalStep,
                        currentEpoch = epoch,
                        stepLoss = 0f,
                        epochLoss = epochLoss / epoch,
                        learningRate = scheduler!!.getLR(),
                        stepDurationMs = 0,
                        epochDurationMs = currentTrainingTime - epochStartTime,
                        totalDurationMs = currentTrainingTime - trainingStartTime
                    )
                )

                epoch++
            }

            val completionTime = System.currentTimeMillis()

            // Merge weights at the end
            if (trainingConfig.mergeWeightsAtEnd) {
                val mergeTimeStart = System.currentTimeMillis()
                callback?.onMergeStart(
                    TrainingProgress(
                        currentStep = globalStep,
                        currentEpoch = epoch,
                        stepLoss = 0f,
                        epochLoss = 0f,
                        totalLoss = totalLoss / globalStep,
                        learningRate = scheduler!!.getLR(),
                        stepDurationMs = 0,
                        epochDurationMs = 0,
                        totalDurationMs = 0,
                        isCompleted = true
                    )
                )

                // Export the current weights into inference folder
                mergeExportSessionWeights()

                callback?.onMergeEnd(
                    TrainingProgress(
                        currentStep = globalStep,
                        currentEpoch = epoch,
                        stepLoss = 0f,
                        epochLoss = 0f,
                        totalLoss = totalLoss / globalStep,
                        learningRate = scheduler!!.getLR(),
                        stepDurationMs = 0,
                        epochDurationMs = 0,
                        totalDurationMs = System.currentTimeMillis() - mergeTimeStart,
                        isCompleted = true
                    )
                )
            }

            // Save and release the model if end of steps
            if (trainingConfig.saveModelAtEnd) {
                val saveModelStart = System.currentTimeMillis()
                callback?.onSaveModelStart(
                    TrainingProgress(
                        currentStep = globalStep,
                        currentEpoch = epoch,
                        stepLoss = 0f,
                        epochLoss = totalLoss / globalStep,
                        learningRate = scheduler!!.getLR(),
                        stepDurationMs = 0,
                        epochDurationMs = 0,
                        totalDurationMs = 0
                    )
                )

                // Save the checkpoint, and release the session unless the caller asked to keep it.
                // `destroySession(true)` does both; splitting them is what lets a caller read the
                // checkpoint it just trained (see keepSessionAtEnd).
                if (trainingConfig.keepSessionAtEnd) saveModel(model, true) else destroySession(true)

                // Callbacks: onSaveModelEnd
                callback?.onSaveModelEnd(
                    TrainingProgress(
                        currentStep = globalStep,
                        currentEpoch = epoch,
                        stepLoss = 0f,
                        epochLoss = totalLoss / globalStep,
                        learningRate = scheduler!!.getLR(),
                        stepDurationMs = 0,
                        epochDurationMs = 0,
                        totalDurationMs = System.currentTimeMillis() - saveModelStart
                    )
                )
            } else if (!trainingConfig.keepSessionAtEnd) {
                // Destroy training session without saving
                destroySession(false)
            }

            // Save the final metrics to training_logs.json
            if (trainingConfig.profileMetrics) {
                saveTrainingLogs(trainingStartTime, completionTime, globalStep * trainingConfig.batchSize)
            }

            // Final completion callback
            callback?.onCompletion(
                TrainingProgress(
                    currentStep = globalStep,
                    currentEpoch = epoch,
                    stepLoss = 0f,
                    epochLoss = 0f,
                    totalLoss = totalLoss / globalStep,
                    learningRate = scheduler!!.getLR(),
                    stepDurationMs = 0,
                    epochDurationMs = 0,
                    totalDurationMs = completionTime - trainingStartTime,
                    isCompleted = true
                )
            )

        } catch (e: Throwable) {
            Log.e(LOG_TAG, e.toString())
            callback?.onError(e)
        }
    }

    fun performTrainStep(batch: List<ORTDataCurator.TrainingSample>): Float {
        if (model == 0L) {
            Log.e(LOG_TAG, "No native training model has been created.")
            return 0f
        }

        // Use the data collator to properly batch and pad the data
        val collatedBatch = dataCollator.collate(batch)

        // Flatten the inputs for the native call
        val flatInputIds = collatedBatch.inputIds.flatMap { it.asIterable() }.toLongArray()
        val flatLabels = collatedBatch.labels.flatMap { it.asIterable() }.toLongArray()
        val flatAttention = collatedBatch.attentionMask.flatMap { it.asIterable() }.toLongArray()

        // Call the native training step function
        val loss = performTraining(
            model,
            flatInputIds,
            flatLabels,
            flatAttention,
            collatedBatch.batchSize,
            collatedBatch.sequenceLength
        )

        return loss
    }

    private fun getTotalMemoryUsageMB(): Long {
        try {
            // Use ActivityManager.getProcessMemoryInfo for most accurate PSS measurement
            // This is what Android Studio Memory Profiler uses internally
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val pids = intArrayOf(android.os.Process.myPid())
            val memoryInfoArray = activityManager.getProcessMemoryInfo(pids)

            if (memoryInfoArray.isNotEmpty()) {
                val memoryInfo = memoryInfoArray[0]

                // - Java heap (Dalvik heap)
                // - Native heap (malloc/free, JNI allocations)
                // - Code memory (.so, .dex, .oat files)
                // - Stack memory
                // - Graphics memory (bitmaps, textures)
                // - Other native memory
                return (memoryInfo.totalPss / 1024).toLong() // Convert KB to MB
            }
        } catch (e: Exception) {
            Log.w(LOG_TAG, "Could not get PSS memory info: ${e.message}")
        }

        // Fallback: Basic Java + Native heap (less comprehensive but still useful)
        val runtime = Runtime.getRuntime()
        val javaHeapUsed = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val nativeHeapUsed = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)

        return javaHeapUsed + nativeHeapUsed
    }

    // Function to save training logs (call this at the end of training)
    private fun saveTrainingLogs(trainingStartTime : Long, trainingEndTime : Long, totalSamples: Int) {
        if (!trainingConfig.profileMetrics || trainingMetrics.isEmpty()) {
            Log.i(LOG_TAG, "No training metrics to save")
            return
        }

        try {
            val totalRuntimeSeconds = (trainingEndTime - trainingStartTime) / 1000f
            val totalSteps = trainingMetrics.size

            // Calculate summary statistics
            val finalLoss = trainingMetrics.lastOrNull()?.loss ?: 0f
            val peakMemoryMB = trainingMetrics.maxOfOrNull { it.memoryUsageMB } ?: 0L
            val avgMemoryMB = trainingMetrics.map { it.memoryUsageMB }.average().toFloat()

            // Create training summary
            val summary = TrainingSummary(
                train_runtime_seconds = totalRuntimeSeconds,
                train_steps_per_second = totalSteps / totalRuntimeSeconds,
                train_samples_per_second = totalSamples / totalRuntimeSeconds,
                total_steps = totalSteps,
                total_samples = totalSamples,
                final_loss = finalLoss,
                peak_memory_mb = peakMemoryMB,
                average_memory_mb = avgMemoryMB
            )

            lastSummary = summary

            // Create complete training log
            val trainingLog = mapOf(
                "summary" to summary,
                "step_metrics" to trainingMetrics
            )

            // Convert to JSON and save
            val gson = Gson()
            val jsonString = gson.toJson(trainingLog)

            // Save to internal storage
            val file = File("$trainDir/training_logs.json")
            file.writeText(jsonString)

            Log.i(LOG_TAG, "Training logs saved to: ${file.absolutePath}")
            Log.i(LOG_TAG, "Training Summary:")
            Log.i(LOG_TAG, "  Runtime: ${totalRuntimeSeconds}s")
            Log.i(LOG_TAG, "  Steps/sec: ${summary.train_steps_per_second}")
            Log.i(LOG_TAG, "  Samples/sec: ${summary.train_samples_per_second}")
            Log.i(LOG_TAG, "  Peak Memory: ${peakMemoryMB}MB")
            Log.i(LOG_TAG, "  Final Loss: $finalLoss")

        } catch (e: Exception) {
            Log.e(LOG_TAG, "Failed to save training logs: ${e.message}")
        }
    }

    /**
     * Release the native training session. Idempotent.
     *
     * The handle used to be passed to `releaseTrainingSession` unconditionally and never cleared, so a
     * second call — which the train->merge->generate flow makes, once at the end of training and once on
     * teardown — freed an already-freed `TrainingSessionCache` and took the process down with SIGSEGV in
     * `Java_..._releaseTrainingSession`. Zeroing the handle under the same guard makes the second call a
     * no-op instead of a use-after-free.
     */
    @Synchronized
    fun destroySession(saveCheckpoint: Boolean) {
        if (model == 0L) {
            Log.d(LOG_TAG, "Training session already released; nothing to destroy.")
            return
        }
        Log.d(LOG_TAG, "Destroying training session and saving checkpoint...")
        val handle = model
        model = 0L
        releaseTrainingSession(handle, saveCheckpoint = saveCheckpoint)
    }

    fun mergeExportSessionWeights() {
        // #9 unified layout: the merger ONNX models + weight_handoff_map.json live in the inference
        // package dir, and merged trainable tensors overwrite their per-tensor <name>.bin in place there
        // (atomic rename + checksum on the native side). The old inference/merged subdir is retired.
        // #23: the native LOAD side (ORTGeneratorNative + session_cache.h) now consumes the handoff map
        // (map present + every externalDataLocation .bin exists + checksum valid), so these in-place
        // merges are seen fail-closed at load time.
        val inferenceDir = pkgPaths.inference.absolutePath
        val merged = mergeExportWeights(
            model,
            "$trainDir/training_config.json",
            inferenceDir,
            inferenceDir,
        )
        // #9 fail-closed: the native merge reports partial/failed writes, and discarding that left a
        // half-merged inference/ dir looking like a successful train→merge. Some tensors would carry
        // the trained values and the rest the frozen base — the model still generates, just wrongly.
        if (!merged) {
            throw MissingArtifactException(
                "weight merge failed for $inferenceDir (see logcat for the offending tensor); " +
                    "the inference package may be partially merged",
            )
        }
    }

    external fun releaseTrainingSession(session: Long, saveCheckpoint : Boolean)

    external fun saveModel(session: Long, saveOptimizer : Boolean)

    external fun createTrainingSession(checkpointPath:String, trainModelPath: String, evalModelPath: String,
                                       optimizerModelPath: String, cacheDirPath: String, requiresGrad: Array<String>, memoryConfigId: String, coreConfigId: String, executionProvider : String, enableProfiling: Boolean) : Long

    external fun performTraining(session: Long, inputIds: LongArray, labels: LongArray, attentionMask: LongArray, batchSize: Int,
                                 sequenceLength: Int) : Float

    external fun setLearningRate(session: Long, learningRate: Float)

    external fun optimizerStep(session: Long)

    external fun mergeExportWeights(
        session: Long,
        peftMappingPath: String?,
        mergerModelsDirectory: String?,
        outputDirectory: String?
    ): Boolean

    /**
     * #36: raw little-endian bytes of one ORT checkpoint parameter, or `null` if absent.
     *
     * Moves BYTES only — the federated record's format is owned by `federated/AdapterTensorCodec`,
     * which is pinned byte-for-byte against the cross-language golden. See the C++ docstring for why
     * the record is not assembled natively.
     */
    external fun nativeExportCheckpointTensor(session: Long, name: String): ByteArray?

    /** #36: writes raw little-endian bytes back into one checkpoint parameter, by name. */
    external fun nativeImportCheckpointTensor(session: Long, name: String, data: ByteArray): Boolean

    /** The live training session handle, for the federated round to read/write checkpoint tensors. */
    internal fun trainingSessionHandle(): Long = model
}