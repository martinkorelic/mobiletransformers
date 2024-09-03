package com.example.orttransformer

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.example.orttransformer.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private val LOG_TAG = "MainActivity"

    private lateinit var binding: ActivityMainBinding
    //private var ortTrainer: ORTTrainer? = null
    private var ortGeneratorNative: ORTGeneratorNative? = null
    private var ortTrainerNative : ORTTrainerNative? = null
    private var ortTokenizer : ORTTokenizer? = null

    private var artifactDir : String = "/data/local/tmp/artifacts"
    private var genAiConfigPath : String = "/data/local/tmp/genaitest"

    private var inferenceModelPath : String = ""


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ortTokenizer = ORTTokenizer(genAiConfigPath)

        inferenceModelPath = "${applicationContext.cacheDir}/inference_model.onnx"

        // Test training
        ortTrainerNative = makeOrtTrainer()
        //performTestTraining()

        // Test inference
        ortGeneratorNative = makeOrtInference()
        performTestInference()
    }

    private fun makeOrtTrainer() : ORTTrainerNative? {

        if (ortTokenizer == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer.")
            return null
        }

        return ORTTrainerNative(artifactDir, ortTokenizer!!, applicationContext.cacheDir.toString())
    }

    private fun makeOrtInference() : ORTGeneratorNative? {
        if (ortTokenizer == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer.")
            return null
        }

        ortGeneratorNative = ORTGeneratorNative(ortTokenizer!!)

        // If ORTTrainer instance already exists, we create a inference model from it
        if (ortTrainerNative != null) {
            Log.d(LOG_TAG, "Loading the model from training session for inference...")
            ortGeneratorNative?.createInferenceModelFromTraining(ortTrainerNative!!.model, inferenceModelPath)
        } else {
            // If ORTTrainer instance does not exist, we load the inference model along with the data from the checkpoint
            Log.d(LOG_TAG, "Loading the model from checkpoint for inference...")
            ortGeneratorNative?.createInferenceModelFromCheckpoint(inferenceModelPath, "$artifactDir/checkpoint", "$artifactDir/training_config.json")
        }

        return ortGeneratorNative
    }

    private fun performTestInference() {
        if (ortTrainerNative == null) {
            Log.e(LOG_TAG, "No inference model defined.")
            return
        }
        val runtime = Runtime.getRuntime()
        val usedMemBefore = runtime.totalMemory() - runtime.freeMemory()
        val startTime = System.nanoTime()

        Log.d(LOG_TAG, "Starting test inference generation...")
        ortGeneratorNative?.generate("Hello, this is a message for the world.", 100)
        val endTime = System.nanoTime()
        val usedMemAfter = runtime.totalMemory() - runtime.freeMemory()
        val durationInSeconds = (endTime - startTime) / 1_000_000_000.0
        val memoryUsedInMB = (usedMemAfter - usedMemBefore) / (1024)

        Log.d(LOG_TAG, "Time elapsed: $durationInSeconds s")
        Log.d(LOG_TAG, "Memory used for training: $memoryUsedInMB")
    }

    private fun performTestTraining() {
        if (ortTrainerNative == null) {
            Log.e(LOG_TAG, "No trainer model defined.")
            return
        }

        val trainData = arrayOf("Hello, this is a message for the world. How is your day?", "I am fine thank you.", "Today is a wonderful day.", "Hello to world and anyone!")

        // Profiling
        val runtime = Runtime.getRuntime()
        val usedMemBefore = runtime.totalMemory() - runtime.freeMemory()
        val startTime = System.nanoTime()
        Log.d(LOG_TAG, "Starting test training...")
        val loss = ortTrainerNative?.performTrainStep(trainData)

        val endTime = System.nanoTime()
        val usedMemAfter = runtime.totalMemory() - runtime.freeMemory()
        val durationInSeconds = (endTime - startTime) / 1_000_000_000.0
        val memoryUsedInMB = (usedMemAfter - usedMemBefore) / (1024)

        Log.d(LOG_TAG, "Loss from training: $loss")
        Log.d(LOG_TAG, "Time elapsed: $durationInSeconds s")
        Log.d(LOG_TAG, "Memory used for training: $memoryUsedInMB")
    }


    /**
     * A native method that is implemented by the 'orttransformer' native library,
     * which is packaged with this application.
    */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'orttransformer' library on application startup.
        init {
            System.loadLibrary("orttransformer")
        }
    }


}