package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import com.martinkorelic.mobiletransformers.app.viewmodels.TrainingUiState
import com.martinkorelic.mobiletransformers.app.viewmodels.TrainingViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun TrainingScreen(
    viewModel: TrainingViewModel
) {
    val trainingState by viewModel.trainingState.collectAsState()
    val trainLoss by viewModel.trainLoss.collectAsState()
    val currentStepDuration by viewModel.currentStepDuration.collectAsState()
    val averageStepDuration by viewModel.averageStepDuration.collectAsState()
    val learningRate by viewModel.learningRate.collectAsState()
    val currentStep by viewModel.currentStep.collectAsState()
    val currentEpoch by viewModel.currentEpoch.collectAsState()

    var trainingLog by remember { mutableStateOf("Training Loss Log:\n") }

    // Update loss log when trainLoss changes
    LaunchedEffect(trainLoss, currentStepDuration) {
        if (currentStepDuration > 0f) {
            val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
            val stepDurationSec = currentStepDuration / 1000.0
            trainingLog += "[$timestamp] Step $currentStep (Epoch $currentEpoch)\n"
            trainingLog += "  Loss: ${String.format("%.6f", trainLoss)}\n"
            trainingLog += "  Step Time: ${String.format("%.2f", stepDurationSec)}s\n"
            trainingLog += "  LR: ${String.format("%.9f", learningRate)}\n"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Training State Display
        Text(
            text = "Training Status: ${trainingState.name}",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        // Loss Log Text Area
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(12.dp)
            ) {
                val scrollState = rememberScrollState()

                // Auto-scroll to bottom when new content is added
                LaunchedEffect(trainingLog) {
                    scrollState.animateScrollTo(scrollState.maxValue)
                }

                Text(
                    text = trainingLog,
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(scrollState),
                    style = MaterialTheme.typography.bodyMedium.copy(
                        fontFamily = FontFamily.Monospace
                    ),
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }

        // Current Loss Display
        if (trainLoss > 0f) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(12.dp)
                ) {
                    Text(
                        text = "Current Loss: ${String.format("%.6f", trainLoss)}",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                    if (currentStepDuration > 0 && averageStepDuration > 0) {
                        val currentStepSec = currentStepDuration / 1000.0
                        val avgStepSec = averageStepDuration / 1000.0
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Last Step: ${String.format("%.2f", currentStepSec)}s",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                            Text(
                                text = "Avg Time: ${String.format("%.2f", avgStepSec)}s/step",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                        }
                    }
                }
            }
        }

        // Action Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Start Training Button
            Button(
                onClick = { viewModel.startTraining() },
                enabled = trainingState == TrainingUiState.ReadyTrain,
                modifier = Modifier.weight(1f)
            ) {
                when (trainingState) {
                    TrainingUiState.Training -> {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Training...")
                    }
                    else -> {
                        Text("Start Training")
                    }
                }
            }

            // Save Model Button
            Button(
                onClick = { viewModel.endTraining(true) },
                enabled = trainingState == TrainingUiState.ReadyTrain,
                modifier = Modifier.weight(1f)
            ) {
                when (trainingState) {
                    TrainingUiState.SavingModel -> {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Saving...")
                    }
                    else -> {
                        Text("Save Model")
                    }
                }
            }
        }

        // Clear Log Button
        OutlinedButton(
            onClick = { trainingLog = "Training Loss Log:\n" },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Clear Log")
        }
    }
}