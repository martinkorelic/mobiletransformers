package com.martinkorelic.ortmobile

import android.util.Log
import org.json.JSONObject
import java.io.*
import kotlin.math.min


class ORTDataCurator(
    private val tokenizer: ORTTokenizerNative,
    private val file: String,
    val batchSize: Int = 4,
    private val maxContextLength: Int? = null,
    private val removeLongSamples: Boolean = true,
    private val maxDatasetLength  : Int? = null,
    private val datasetBatchSize : Int? = null, // If null, load all data into memory
    private val customPreprocess: TaskPreprocessor? = null
) {
    private val LOG_TAG = "ORTDataCurator"

    // Dataset info
    private var totalDatasetSamples = 0
    private var isInitialized = false

    // For streaming mode (when datasetBatchSize is provided)
    private var currentFilePosition = 0L
    private var currentSampleIndex = 0
    private var totalSamplesProcessed = 0
    private var isDatasetExhausted = false
    private var currentDataBatch: List<TrainingSample> = emptyList()
    private var currentBatchIndex = 0

    // For in-memory mode (when datasetBatchSize is null)
    private var allSamples: List<TrainingSample> = emptyList()
    private var inMemoryIndex = 0

    init {
        initialize()
    }

    /**
     * Initialize the data curator by first loading and counting samples
     */
    private fun initialize() {
        if (isInitialized) return

        Log.i(LOG_TAG, "Initializing data curator...")

        // First pass: count and optionally load all samples
        val samples = mutableListOf<TrainingSample>()
        var count = 0

        try {
            BufferedReader(FileReader("$file.jsonl")).use { reader ->
                reader.forEachLine { line ->
                    if (line.isNotBlank() && (maxDatasetLength == null || count < maxDatasetLength)) {
                        val sample = processLine(line)
                        if (sample != null) {
                            count++
                            // If streaming mode is not enabled, keep samples in memory
                            if (datasetBatchSize == null) {
                                samples.add(sample)
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error during initialization: ${e.message}")
        }

        totalDatasetSamples = count

        if (datasetBatchSize == null) {
            // In-memory mode: keep all samples
            allSamples = samples
            Log.i(LOG_TAG, "Initialized in-memory mode with $totalDatasetSamples samples")
        } else {
            // Streaming mode: samples are discarded, file will be streamed
            Log.i(LOG_TAG, "Initialized streaming mode with $totalDatasetSamples samples, batch size: $datasetBatchSize")
        }

        isInitialized = true
    }

    private fun loadNextDataBatch(): Boolean {
        // Only used in streaming mode
        if (datasetBatchSize == null) return false
        if (isDatasetExhausted || (maxDatasetLength != null && totalSamplesProcessed >= maxDatasetLength)) {
            return false
        }

        val samples = mutableListOf<TrainingSample>()
        var samplesLoaded = 0
        var linesRead = 0

        try {
            RandomAccessFile("$file.jsonl", "r").use { raf ->
                raf.seek(currentFilePosition)

                var line: String? = null

                while (samplesLoaded < datasetBatchSize!! &&
                    totalSamplesProcessed < totalDatasetSamples &&
                    raf.readLine().also { line = it } != null) {

                    linesRead++
                    line?.let { lineContent ->
                        val sample = processLine(lineContent)
                        if (sample != null) {
                            samples.add(sample)
                            samplesLoaded++
                            totalSamplesProcessed++
                        }
                    }
                }

                // Update file position for next batch
                currentFilePosition = raf.filePointer

                // Check if we've reached end of file
                if (line == null) {
                    isDatasetExhausted = true
                }
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error loading data batch: ${e.message}")
            return false
        }

        currentDataBatch = samples
        currentBatchIndex = 0

        Log.d(LOG_TAG, "Loaded batch: $samplesLoaded samples (lines read: $linesRead), " +
                "total processed: $totalSamplesProcessed/$totalDatasetSamples, file pos: $currentFilePosition")

        return samples.isNotEmpty()
    }

    /**
     * Get the next sample from the current batch, loading new batch if needed
     */
    private fun getNextSample(): TrainingSample? {
        return if (datasetBatchSize == null) {
            // In-memory mode
            if (inMemoryIndex >= allSamples.size) {
                null
            } else {
                allSamples[inMemoryIndex++]
            }
        } else {
            // Streaming mode
            if (currentBatchIndex >= currentDataBatch.size) {
                if (!loadNextDataBatch()) {
                    return null // No more data available
                }
            }

            if (currentBatchIndex < currentDataBatch.size) {
                currentDataBatch[currentBatchIndex++]
            } else {
                null
            }
        }
    }

    /**
     * Reset the data curator to start from beginning
     */
    fun reset() {
        if (datasetBatchSize == null) {
            // In-memory mode
            inMemoryIndex = 0
        } else {
            // Streaming mode
            currentFilePosition = 0L
            currentSampleIndex = 0
            totalSamplesProcessed = 0
            isDatasetExhausted = false
            currentDataBatch = emptyList()
            currentBatchIndex = 0
        }
        Log.i(LOG_TAG, "Data curator reset to beginning")
    }

    /**
     * Get total dataset size
     */
    fun getDatasetSize(): Int {
        return totalDatasetSamples
    }

    /**
     * Build method for backward compatibility - returns all samples up to maxDatasetLength
     * WARNING: This loads all data into memory and should be avoided for large datasets
     */
    @Deprecated("Use getBatchedDataset() for memory-efficient streaming")
    fun build(): List<TrainingSample> {
        val dataset = mutableListOf<TrainingSample>()
        val reader = BufferedReader(FileReader(file))
        var line: String?

        while (reader.readLine().also { line = it } != null) {
            line?.let {
                val sample = processLine(it)
                if (sample != null) {
                    dataset.add(sample)
                }
            }
        }

        reader.close()
        return dataset
    }

    /**
     * Get current progress information
     */
    fun getProgress(): DatasetProgress {
        return if (datasetBatchSize == null) {
            // In-memory mode
            DatasetProgress(
                samplesProcessed = inMemoryIndex,
                maxSamples = totalDatasetSamples,
                currentBatchSize = allSamples.size,
                isExhausted = inMemoryIndex >= allSamples.size
            )
        } else {
            // Streaming mode
            DatasetProgress(
                samplesProcessed = totalSamplesProcessed,
                maxSamples = totalDatasetSamples,
                currentBatchSize = currentDataBatch.size,
                isExhausted = isDatasetExhausted
            )
        }
    }

    fun getBatchedDataset(): Sequence<List<TrainingSample>> = sequence {
        var batch = mutableListOf<TrainingSample>()

        while (true) {
            val sample = getNextSample() ?: break

            batch.add(sample)
            if (batch.size == batchSize) {
                yield(batch.toList())
                batch.clear()
            }
        }

        // Yield remaining samples if any
        if (batch.isNotEmpty()) {
            yield(batch.toList())
        }
    }

    private fun processLine(line: String): TrainingSample? {
        try {
            val json = JSONObject(line)

            // Step 1: Get input and label text (e.g. prompt + response)
            val (inputText, labelText) = customPreprocess?.preprocess(json)
                ?: return null

            if (inputText.isBlank() || labelText.isBlank()) return null

            // Step 2: Tokenize prompt and answer separately (like Python code)
            val promptTokens = tokenizer.tokenize(inputText)
            val answerTokens = tokenizer.tokenize(
                labelText
            ) // No special tokens for answer

            // Step 3: Concatenate the token seq  uences
            val fullInputIds = (promptTokens + answerTokens).toMutableList()

            // Step 4: Create labels same as fullInputIds, then mask prompt part
            val labels = fullInputIds.toMutableList()

            // Step 5: Mask ALL prompt tokens with -100 (ignore index)
            for (i in 0 until promptTokens.size) {
                if (i < labels.size) labels[i] = -100
            }

            // Step 6: Optionally filter long sequences
            if (removeLongSamples && maxContextLength != null && fullInputIds.size >= maxContextLength) {
                return null
            }

            return TrainingSample(inputIds = fullInputIds, labels = labels)
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Invalid JSON or tokenization failed: ${e.message}")
            return null
        }
    }

    data class TrainingSample(
        val inputIds: List<Int>,
        val labels: List<Int>
    )

    data class DatasetProgress(
        val samplesProcessed: Int,
        val maxSamples: Int,
        val currentBatchSize: Int,
        val isExhausted: Boolean
    )
}
