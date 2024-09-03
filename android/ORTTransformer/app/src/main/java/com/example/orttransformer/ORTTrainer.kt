package com.example.orttransformer

// ONNX runtime training framework
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtTrainingSession


import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.LongBuffer
import java.nio.file.Path
import java.util.Collections
import kotlin.io.path.Path
import kotlin.math.exp
import kotlin.random.Random

class ORTTrainer(
    checkpointPath: String,
    trainModelPath: String,
    evalModelPath: String,
    optimizerModelPath: String
) {

    private var ortEnv: OrtEnvironment? = null
    private var ortTrainingSession: OrtTrainingSession? = null
    private var ortSession: OrtSession? = null

    init {
        ortEnv = OrtEnvironment.getEnvironment()

        val ortoption = OrtSession.SessionOptions()
        // Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance.
        // e.g. ORT explicitly assigns shape related ops to CPU to improve performance
        ortoption.addNnapi()
        ortTrainingSession = ortEnv?.createTrainingSession(checkpointPath, trainModelPath, evalModelPath, optimizerModelPath, ortoption)
    }

    fun performTrainingDummy() {
        ortSession = null

        val batchSize : Long = 2
        val d1 = dummyInfData()
        val l1 = dummyInfData()

        val d2 = generateLongBuffers()
        val l2 = generateLongBuffers()

        var loss = -1.0f
        Log.d("ORTTrainer", "Start training...")
        ortEnv.use {
            val dataShape = longArrayOf(batchSize, 23)
            val labelsShape = longArrayOf(batchSize, 23)

            val inputTensor = OnnxTensor.createTensor(ortEnv, d2, dataShape)
            val labelsTensor = OnnxTensor.createTensor(ortEnv, l2, labelsShape)
            inputTensor.use {
                labelsTensor.use {
                    val ortInputMap: MutableMap<String, OnnxTensor> = HashMap<String, OnnxTensor>()
                    ortInputMap["input_ids"] = inputTensor
                    ortInputMap["labels"] = labelsTensor
                    val output = ortTrainingSession?.trainStep(ortInputMap)
                    output.use {
                        Log.d("ORTTrainer", "${output?.get(0)}")
                        Log.d("ORTTrainer", "${output?.get(0)?.value}")
                        loss = ((output?.get(0)?.value) as Float)

                        }
                    output?.close()
                }
            }
            Log.d("ORTTrainer", "Done training, final loss: $loss")
            ortTrainingSession?.optimizerStep()
            ortTrainingSession?.lazyResetGrad()

        }
    }

    fun dummyData(): Pair<LongBuffer, LongBuffer> {
        val vocabSize = 32000

        val inputSize = 10
        val logitsSize = vocabSize

        // Create ByteBuffers with the appropriate size
        val inputBuffer = ByteBuffer.allocate(Long.SIZE_BYTES * inputSize).asLongBuffer()
        val logitsBuffer = ByteBuffer.allocate(Long.SIZE_BYTES * logitsSize).asLongBuffer()

        // Populate the buffers with random long values
        val random = Random.Default

        for (i in 0 until inputSize) {
            Log.d("T", "${random.nextLong(vocabSize.toLong())}")
            inputBuffer.put(random.nextLong(vocabSize.toLong())) // Random long in range [0, vocabSize)
        }

        for (i in 0 until logitsSize) {
            logitsBuffer.put(random.nextLong(vocabSize.toLong())) // Random long in range [0, vocabSize)
            Log.d("T", "${random.nextLong(vocabSize.toLong())}")
        }

        // Reset the buffers to prepare for reading
        inputBuffer.rewind()
        logitsBuffer.rewind()

        return Pair(inputBuffer, logitsBuffer)
    }

    fun dummyInfData(): LongBuffer? {
        val array = intArrayOf(
            1, 15043, 920, 338, 596, 2462, 29973, 7963, 366, 526, 2599, 1532,
            29889, 1724, 338, 278, 931, 29973, 3529, 1246, 592, 3339, 3301
        )
        // Convert IntArray to LongArray
        val longArray = array.map { it.toLong() }.toLongArray()

        // Allocate a LongBuffer with the same size as the LongArray
        val longBuffer = LongBuffer.allocate(longArray.size)

        // Put the LongArray values into the LongBuffer
        longBuffer.put(longArray)

        // Reset the buffer position to 0 for reading
        longBuffer.rewind()

        return longBuffer
    }

    fun generateLongBuffers(): LongBuffer? {
        // Define two sequences
        val sequence1 = intArrayOf(
            1, 15043, 920, 338, 596, 2462, 29973, 7963, 366, 526, 2599, 1532,
            29889, 1724, 338, 278, 931, 29973, 3529, 1246, 592, 3339, 3301
        )
        val sequence2 = intArrayOf(
            1, 15043, 920, 338, 596, 2462, 29973, 7963, 366, 526, 2599, 1532,
            29889, 1724, 338, 278, 931, 29973, 3529, 1246, 592, 3339, 3301
        )

        // Combine the sequences into one LongArray
        val combinedArray = (sequence1 + sequence2).map { it.toLong() }.toLongArray()

        // Allocate a LongBuffer with the size of the combined array
        val longBuffer = LongBuffer.allocate(combinedArray.size)

        // Put the combined LongArray values into the LongBuffer
        longBuffer.put(combinedArray)

        // Reset the buffer position to 0 for reading
        longBuffer.rewind()

        return longBuffer
    }

    fun performInferenceDummy(cacheDir: File, exportModel: Boolean): Int {
        val data = dummyInfData()
        if (ortSession == null) {

            val inferenceModelPath: Path = Path(cacheDir.toString(), "inference_model_new_1.onnx")

            if (exportModel) {
                val graphOutput: Array<String> = arrayOf("logits")
                ortTrainingSession?.exportModelForInference(inferenceModelPath, graphOutput)
            }
            ortTrainingSession?.close()

            val ortoption = OrtSession.SessionOptions()
            ortoption.addNnapi()
            ortSession = ortEnv?.createSession(inferenceModelPath.toString(), ortoption)

        }
        var maxIdx = -1
        Log.d("ORTTrainer", "Starting inference...")
        ortEnv.use {
            val shape = longArrayOf(1, 23)
            val tensor = OnnxTensor.createTensor(ortEnv, data, shape)
            tensor.use {
                val output = ortSession?.run(Collections.singletonMap("input_ids", tensor))
                output.use {
                    @Suppress("UNCHECKED_CAST")
                    Log.d("ORTTrainer", "${output?.get(0)}")
                    Log.d("ORTTrainer", "${output?.get(0)?.value}")
                    val rawOutput = ((output?.get(0)?.value) as Array<Array<FloatArray>>)[0]
                    val probabilities = softMax(rawOutput[1])

                    maxIdx = probabilities.indices.maxBy { probabilities[it] } ?: -1
                }
            }
        }
        Log.d("ORTTrainer", "Inference finished.")

        check(maxIdx >= 0) { "Index is < 0" }

        Log.d("ORTTrainer", "$maxIdx")
        return maxIdx
    }

    private fun softMax(modelResult: FloatArray): FloatArray {
        val labelVals = modelResult.copyOf()
        val max = labelVals.max()
        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            Log.d("ORTTrainer", "${labelVals[i]}")
            labelVals[i] = exp(labelVals[i] - max)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }
}