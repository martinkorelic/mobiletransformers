package com.example.llmfinetuning

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.llmfinetuning.ui.theme.LLMFinetuningTheme
import com.example.llmfinetuning.TFliteHelper

class MainActivity : ComponentActivity() {

    private lateinit var model: TFliteHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("LLMFinetune", "Starting the load...")
        // Create TFLite instance
        model = TFliteHelper(this)

        // Single example inference call
        //model.runInference()

        // Single example training call
        model.runTraining()

        enableEdgeToEdge()
        setContent {
            LLMFinetuningTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    LLMFinetuningTheme {
        Greeting("Android")
    }
}