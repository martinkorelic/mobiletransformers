package com.example.llmfinetuning
import org.tensorflow.lite.Interpreter;
import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer


class TFliteHelper(context: Context) {
    private var tfliteModel: Interpreter? = null

    init {
        try {
            //val assetManager = context.assets
            //val inputStream = assetManager.open("gemma2-lora4-inference+training.tflite")
            //val file = File(context.filesDir, "gemma2-lora4-inference+training.tflite")
            val modelPath = "/data/local/tmp/gemma2-lora4-fix-inference+training.tflite"

            val modelFile = File(modelPath)
            if (!modelFile.exists()) {
                throw FileNotFoundException("Model file not found at path: $modelPath")
            }

            tfliteModel = Interpreter(modelFile)
            Log.d("LLMFinetune", "Signature keys: ${tfliteModel?.signatureKeys?.joinToString(separator = ", ")}")
            tfliteModel?.getInputTensorFromSignature("padding_mask", "call_inference")?.let { Log.d("LLMFinetune", "Input bytes: ${it.numBytes()}") }
            tfliteModel?.getInputTensorFromSignature("token_ids", "call_inference")?.let { Log.d("LLMFinetune", "Output bytes: ${it.numBytes()}") }
            tfliteModel?.getOutputTensorFromSignature("loss", "call_training")?.let { Log.d("LLMFinetune", "Output bytes loss: ${it.numBytes()}") }
            tfliteModel?.getOutputTensorFromSignature("sparse_categorical_accuracy", "call_training")?.let { Log.d("LLMFinetune", "Output bytes loss: ${it.numBytes()}") }
            //print(tfliteModel?.signatureKeys)
            //print(tfliteModel?.)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun runInference(): FloatArray {
        // Convert input array to ByteBuffer
        val inputs: MutableMap<String, Any> = HashMap()
        val outputs: MutableMap<String, Any> = HashMap()

        val out_padding_mask = ByteBuffer.allocate(256)
        out_padding_mask.order(ByteOrder.nativeOrder())
        outputs["padding_mask"] = out_padding_mask

        val out_token_ids = ByteBuffer.allocate(1024)
        out_token_ids.order(ByteOrder.nativeOrder())
        outputs["token_ids"] = out_token_ids

        inputs["token_ids"] = intArrayToByteBuffer(arrayOf(intArrayOf(2, 214064, 603, 5271, 6044, 9581, 1, 0)), 256)
        inputs["padding_mask"] = intArrayToByteBuffer(arrayOf(intArrayOf(1, 1, 1, 1, 1, 1, 1, 0)), 256)

        // Run inference
        Log.d("LLMFinetune", "Running inference...")
        tfliteModel?.runSignature(inputs, outputs, "call_inference")

        Log.d("LLMFinetune", printByteBufferAsInt32Array(outputs["token_ids"] as ByteBuffer))
        return FloatArray(1)
        // Get output data
        //return outputTensorBuffer.floatArray
    }

    fun runTraining(): FloatArray {
        // Convert input array to ByteBuffer
        val inputs: MutableMap<String, Any> = HashMap()
        val outputs: MutableMap<String, Any> = HashMap()

        val loss = ByteBuffer.allocate(4)
        loss.order(ByteOrder.nativeOrder())
        outputs["loss"] = loss

        val sca = ByteBuffer.allocate(4)
        sca.order(ByteOrder.nativeOrder())
        outputs["sparse_categorical_accuracy"] = sca

        // Token ids for "<bos> Keras is deep learning library<eos>"
        inputs["x_token_ids"] = intArrayToByteBuffer(arrayOf(intArrayOf(2, 214064, 603, 5271, 6044, 9581, 1, 0)), 256)
        inputs["x_padding_mask"] = intArrayToByteBuffer(arrayOf(intArrayOf(1, 1, 1, 1, 1, 1, 1, 0)), 256)
        inputs["y_token_ids"] = intArrayToByteBuffer(arrayOf(intArrayOf(214064, 603, 5271, 6044, 9581, 3, 0, 0)), 256)
        inputs["sample_weights"] = intArrayToByteBuffer(arrayOf(intArrayOf(1, 1, 1, 1, 1, 1, 0, 0)), 256)

        // Run inference
        Log.d("LLMFinetune", "Running training...")
        tfliteModel?.runSignature(inputs, outputs, "call_training")

        Log.d("LLMFinetune", (outputs["loss"] as ByteBuffer).getFloat().toString())
        Log.d("LLMFinetune", (outputs["sparse_categorical_accuracy"] as ByteBuffer).getFloat().toString())
        return FloatArray(1)
        // Get output data
        //return outputTensorBuffer.floatArray
    }

    // Function to convert int array to ByteBuffer with padding
    private fun intArrayToByteBuffer(data: Array<IntArray>, maxLength: Int): ByteBuffer {
        val paddedData = data.map { row ->
            row.copyOf(maxLength)
        }.toTypedArray()

        val flatData = paddedData.flatMap { it.asList() }.toIntArray()
        val byteBuffer = ByteBuffer.allocateDirect(flatData.size * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        for (value in flatData) {
            byteBuffer.putInt(value)
        }
        byteBuffer.rewind()
        return byteBuffer
    }


    fun printByteBufferAsInt32Array(buffer: ByteBuffer) : String {
        buffer.rewind() // Ensure the buffer is at the beginning
        buffer.order(ByteOrder.nativeOrder()) // Set the byte order to native order

        // Ensure the buffer's remaining bytes can be divided into 4-byte int32 values
        val intArray = IntArray(buffer.remaining() / 4) {
            buffer.int
        }

        // Print the array in joined format
        return intArray.joinToString(", ", "[", "]")
    }
}