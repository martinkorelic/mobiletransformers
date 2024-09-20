package com.example.orttransformer.views

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.orttransformer.ui.theme.ORTTransformerTheme
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.TextField
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.orttransformer.repository.ChatMessage
import com.example.orttransformer.viewmodels.InferenceViewModel
import kotlinx.coroutines.launch


@Composable
fun InferenceScreen(viewModel: InferenceViewModel) {
    var chatInput by remember { mutableStateOf(TextFieldValue("")) }
    val chatHistory by viewModel.chatHistory.collectAsState()
    val chatStream by viewModel.chatStream.collectAsState()
    val isStreaming by viewModel.isStreaming.collectAsState()

    val keyboardController = LocalSoftwareKeyboardController.current

    Column(modifier = Modifier
        .fillMaxSize()
        .padding(16.dp)) {
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(bottom = 16.dp)
        ) {
            items(chatHistory) { message ->
                ChatBubble(message)
            }
            // Show the current streaming message if available
            if (isStreaming) {
                chatStream.let { streamingMessage ->
                    item {
                        ChatBubble(message = ChatMessage(message = streamingMessage.joinToString(separator = ""), isUserMessage = false))
                    }
                }
            }

        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TextField(
                enabled = !isStreaming,
                value = chatInput,
                onValueChange = { chatInput = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Enter message") }
            )
            Button(
                enabled = !isStreaming,
                onClick = {
                    keyboardController?.hide()
                    viewModel.sendMessage(chatInput.text)
                    chatInput = TextFieldValue("")
                },
                modifier = Modifier.padding(start = 8.dp)
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
fun ChatBubble(message: ChatMessage) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            //.align(if (message.isUserMessage) Alignment.End else Alignment.Start)
    ) {
        Surface(
            modifier = Modifier
                //.background(if (message.isUserMessage) Color.Blue else Color.Gray)
                .padding(8.dp), // This padding applies to the inside of the bubble
            color = if (message.isUserMessage) Color.Blue else Color.Gray,
            shape = MaterialTheme.shapes.medium
        ) {
            Text(
                text = message.message,
                color = Color.White,
                fontSize = 16.sp,
                textAlign = TextAlign.Start,
                modifier = Modifier.padding(16.dp) // This padding applies to the text within the bubble
            )
        }
    }
}

