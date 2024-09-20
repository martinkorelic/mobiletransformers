package com.example.orttransformer.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.orttransformer.ui.theme.ORTTransformerTheme
import com.example.orttransformer.viewmodels.TrainingViewModel


@Composable
fun TrainingScreen(viewModel : TrainingViewModel) {
    val trainingData by viewModel.trainingData.collectAsState()
    var newTrainingInput by remember { mutableStateOf(TextFieldValue("")) }
    val isTraining by viewModel.isTraining.collectAsState()

    val keyboardController = LocalSoftwareKeyboardController.current

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        TrainingStatusModal(isTraining = isTraining)
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(bottom = 16.dp)
        ) {
            items(trainingData) { item ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(text = item, modifier = Modifier.padding(8.dp))
                    IconButton(onClick = {
                        viewModel.removeTrainingData(item)
                    }) {
                        Icon(Icons.Default.Delete, contentDescription = "Delete")
                    }
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TextField(
                enabled = !isTraining,
                value = newTrainingInput,
                onValueChange = { newTrainingInput = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Enter training data") }
            )
            Button(
                enabled = !isTraining,
                onClick = {
                    if (newTrainingInput.text.isNotEmpty()) {
                        viewModel.addTrainingData(newTrainingInput.text)
                        newTrainingInput = TextFieldValue("") // Clear input
                        keyboardController?.hide() // Hide keyboard
                    }
                },
                modifier = Modifier.padding(start = 8.dp)
            ) {
                Text("Add")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            enabled = !isTraining,
            onClick = {
                viewModel.startTraining()
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Train")
        }
    }
}

@Composable
fun TrainingStatusModal(isTraining: Boolean) {
    if (isTraining) {
        // Show training modal
        AlertDialog(
            onDismissRequest = { /* Do nothing on dismiss */ },
            title = { Text("Training in progress") },
            text = { Text("Training... Please wait.") },
            confirmButton = {
                Button(onClick = { /* Handle button click */ }) {
                    Text("OK")
                }
            }
        )
    }
}