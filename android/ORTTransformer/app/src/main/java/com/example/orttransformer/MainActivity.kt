package com.example.orttransformer

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBars
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.orttransformer.databinding.ActivityMainBinding
import com.example.orttransformer.repository.InferenceRepository
import com.example.orttransformer.repository.LLMRepository
import com.example.orttransformer.repository.TrainingRepository
import com.example.orttransformer.ui.theme.ORTTransformerTheme
import com.example.orttransformer.viewmodels.InferenceViewModel
import com.example.orttransformer.viewmodels.TrainingViewModel
import com.example.orttransformer.views.InferenceScreen
import com.example.orttransformer.views.TrainingScreen

class MainActivity : ComponentActivity() {

    private val LOG_TAG = "MainActivity"

    private lateinit var binding: ActivityMainBinding
    //private var ortTrainer: ORTTrainer? = null
    private var ortGeneratorNative: ORTGeneratorNative? = null
    private var ortTrainerNative : ORTTrainerNative? = null
    private var ortTokenizer : ORTTokenizer? = null
    private var ortGenAiNative : ORTGenAINative? = null

    private var artifactTrainDir : String = "/data/local/tmp/tinyllama_int16/train"
    private var tokenizerConfigPath : String = "/data/local/tmp/genaitest"
    private var genAiConfigPath : String = "/data/local/tmp/tinyllama_int16/inference"

    private var inferenceModelPath : String = ""

    private var llmRepository = LLMRepository(artifactTrainDir,
        ""
    )
    private var inferenceRepository = InferenceRepository(llmRepository)
    private var trainingRepository = TrainingRepository(llmRepository)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        //ortTokenizer = ORTTokenizer(tokenizerConfigPath)

        //inferenceModelPath = "${applicationContext.cacheDir}/inference_model.onnx"

        // Test training
        //ortTrainerNative = makeOrtTrainer()
        //performTestTraining()

        // Test GenAI
        //ortGenAiNative = makeOrtGenAI()

        //var prompt = "Is there an answer to the end of the universe? Will it ever end and what will"

        //ortGenAiNative?.generate(prompt)

        //prompt = "Hello, this is a message for the world. How is your day?"
        //ortGenAiNative?.generate(prompt)

        // Test inference
        //ortGeneratorNative = makeOrtInference()
        //performTestInference()
        setContent {
            ORTTransformerTheme {
                Surface (
                    modifier = Modifier.fillMaxSize().padding(WindowInsets.systemBars.asPaddingValues()),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainApp()
                }
            }
        }


    }


    private fun makeOrtGenAI() : ORTGenAINative? {
        if (ortTrainerNative == null) {
            Log.e(LOG_TAG, "Could not find the train model. Make sure it is initialized before GenAI inference.")
            return null
        }

        val genAiNative = ORTGenAINative(artifactTrainDir, genAiConfigPath)
        genAiNative.createGenAISessionFromTraining(ortTrainerNative!!.model)

        return genAiNative
    }

    private fun makeOrtTrainer() : ORTTrainerNative? {

        if (ortTokenizer == null) {
            Log.e(LOG_TAG, "Could not find the tokenizer.")
            return null
        }

        return ORTTrainerNative(artifactTrainDir, ortTokenizer!!, applicationContext.cacheDir.toString())
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
            ortGeneratorNative?.createInferenceModelFromCheckpoint(inferenceModelPath, "$artifactTrainDir/checkpoint", "$artifactTrainDir/training_config.json")
        }

        return ortGeneratorNative
    }


    private fun performTestTraining() {
        if (ortTrainerNative == null) {
            Log.e(LOG_TAG, "No trainer model defined.")
            return
        }

        val trainData = listOf("Hello, this is a message for the world. How is your day?", "I am fine thank you.", "Today is a wonderful day.", "Hello to world and anyone!")

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


    companion object {
        // Used to load the 'orttransformer' library on application startup.
        init {
            System.loadLibrary("orttransformer")
        }
    }

    @Composable
    fun MainApp() {
        var selectedTab by remember { mutableStateOf(0) }
        val tabs = listOf("Inference", "Training")

        Column(modifier = Modifier.fillMaxSize().padding(top = 16.dp)) {
            TabRow(selectedTabIndex = selectedTab) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(title) }
                    )
                }
            }
            when (selectedTab) {
                0 -> InferenceScreen(viewModel = InferenceViewModel(inferenceRepository = inferenceRepository))
                1 -> TrainingScreen(viewModel = TrainingViewModel(trainingRepository = trainingRepository))
            }
        }
    }
}