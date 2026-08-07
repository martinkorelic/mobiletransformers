package com.martinkorelic.mobiletransformers.app.views

import android.widget.Toast
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.martinkorelic.mobiletransformers.app.viewmodels.ChatMessage
import com.martinkorelic.mobiletransformers.app.viewmodels.ConfigurationViewModel
import com.martinkorelic.mobiletransformers.app.viewmodels.InferenceUiState
import com.martinkorelic.mobiletransformers.app.viewmodels.InferenceViewModel
import com.martinkorelic.mobiletransformers.app.viewmodels.RagMessage


@Composable
fun InferenceScreen(viewModel: InferenceViewModel, configurationViewModel: ConfigurationViewModel) {
    var chatInput by remember { mutableStateOf(TextFieldValue("")) }
    val chatHistory by viewModel.chatHistory.collectAsState()
    val chatStream by viewModel.chatStream.collectAsState()
    val isStreaming by viewModel.isStreaming.collectAsState()
    val inferenceState by viewModel.inferenceState.collectAsState()

    val ttlmTime by viewModel.ttlmTime.collectAsState()
    val prefillTime by viewModel.prefillTime.collectAsState()
    val generationTime by viewModel.generationTime.collectAsState()
    val queryTime by viewModel.queryTime.collectAsState()
    val embeddingTime by viewModel.embeddingTime.collectAsState()

    val ragEnabled = configurationViewModel.ragEnabled.value

    val keyboardController = LocalSoftwareKeyboardController.current
    val context = LocalContext.current

    Column(modifier = Modifier
        .fillMaxSize()
        .background(Color.White)
        .padding(16.dp)
        ) {

        Column (
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        ) {
            // Compact metrics display
            CompactMetricsCard(
                inferenceState = inferenceState,
                ttlmTime = ttlmTime,
                prefillTime = prefillTime,
                generationTime = generationTime,
                queryTime = queryTime,
                embeddingTime = embeddingTime,
                showRagMetrics = ragEnabled,
                modifier = Modifier.padding(bottom = 16.dp)
            )
        }

        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(bottom = 16.dp)
        ) {
            items(chatHistory) { message ->
                when (message) {
                    is ChatMessage -> ChatBubble(message = message)
                    is RagMessage -> RagSourcesCard(ragMessage = message)
                }
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
                enabled = inferenceState == InferenceUiState.ReadyGenerate,
                value = chatInput,
                onValueChange = { chatInput = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Enter message", color = MaterialTheme.colorScheme.primary) },
                colors = TextFieldDefaults.colors(
                    focusedTextColor = MaterialTheme.colorScheme.secondary,
                    unfocusedTextColor = MaterialTheme.colorScheme.secondary,
                    disabledTextColor = Color.Gray,
                    focusedContainerColor = Color.White,
                    unfocusedContainerColor = Color.White,
                    disabledContainerColor = Color.White.copy(alpha = 0.7f),
                    focusedIndicatorColor = MaterialTheme.colorScheme.primary,
                    unfocusedIndicatorColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
                    disabledIndicatorColor = Color.Gray,
                    cursorColor = MaterialTheme.colorScheme.primary
                )
            )

            Button(
                enabled = inferenceState == InferenceUiState.ReadyGenerate,
                onClick = {
                    keyboardController?.hide()
                    viewModel.sendMessage(chatInput.text, ragEnabled)
                    chatInput = TextFieldValue("")
                },
                modifier = Modifier.padding(start = 8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = Color.White
                )
            ) {
                Text("Send", color = Color.White)
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Second Row: Reload Button
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround
        ) {
            // RAG Toggle Button
            Button(
                enabled = inferenceState == InferenceUiState.ReadyGenerate && configurationViewModel.isRagAvailable.value,
                onClick = {
                    configurationViewModel.updateRagEnabled(!ragEnabled)
                },
                modifier = Modifier.padding(start = 8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (ragEnabled && configurationViewModel.isRagAvailable.value) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.outline
                    },
                    contentColor = if (ragEnabled && configurationViewModel.isRagAvailable.value) {
                        Color.White
                    } else {
                        MaterialTheme.colorScheme.onSurface
                    },
                    disabledContainerColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.12f),
                    disabledContentColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
                )
            ) {
                Text(
                    text = if (ragEnabled) "RAG ON" else "RAG OFF",
                    fontSize = 12.sp
                )
            }
            Button(
                enabled = inferenceState == InferenceUiState.ReadyGenerate,
                onClick = {
                    keyboardController?.hide()
                    Toast.makeText(context, "Reloading model...", Toast.LENGTH_SHORT).show()
                    viewModel.reloadInferenceSession()
                    chatInput = TextFieldValue("")
                },
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = Color.White
                )
            ) {
                Icon(
                    Icons.Default.Refresh,
                    contentDescription = "Reload",
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text("Reload")
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
                .padding(8.dp)
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.primary, // Change to your desired border color
                    shape = MaterialTheme.shapes.medium
                ),
            color = if (message.isUserMessage) Color.White else MaterialTheme.colorScheme.primary,

            shape = MaterialTheme.shapes.medium
        ) {
            Text(
                text = message.message,
                color = if (message.isUserMessage) MaterialTheme.colorScheme.primary else Color.White,
                fontSize = 16.sp,
                textAlign = TextAlign.Start,
                modifier = Modifier.padding(16.dp) // This padding applies to the text within the bubble
            )
        }
    }
}

// Compact metrics card
@Composable
fun CompactMetricsCard(
    inferenceState: InferenceUiState,
    ttlmTime: Double,
    prefillTime: Double,
    generationTime: Double,
    queryTime: Double,
    embeddingTime: Double,
    showRagMetrics: Boolean,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
        shape = RoundedCornerShape(8.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            // Status row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Status:",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                StatusChip(inferenceState = inferenceState)
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Metrics in separate rows
            CompactMetricItem(
                label = "Model Load",
                value = "%.2f s".format(ttlmTime)
            )

            CompactMetricItem(
                label = "Prefill",
                value = "%.2f s".format(prefillTime)
            )

            CompactMetricItem(
                label = "Generation",
                value = "%.2f tokens/s".format(generationTime)
            )

            // RAG metrics - only show when enabled
            if (showRagMetrics) {
                CompactMetricItem(
                    label = "Embedding",
                    value = "%.3f s".format(embeddingTime)
                )
                CompactMetricItem(
                    label = "DB Query",
                    value = "%.3f s".format(queryTime)
                )
            }
        }
    }
}

@Composable
fun StatusChip(inferenceState: InferenceUiState) {
    val (text, color) = when (inferenceState) {
        InferenceUiState.LoadingModel -> "Loading Model" to MaterialTheme.colorScheme.tertiary
        InferenceUiState.LoadingRetriever -> "Loading RAG" to MaterialTheme.colorScheme.tertiary
        InferenceUiState.FinishedLoadingModel -> "Model Ready" to MaterialTheme.colorScheme.primary
        InferenceUiState.Querying -> "DB Query" to MaterialTheme.colorScheme.secondary
        InferenceUiState.ReadyGenerate -> "Ready" to Color(0xFF4CAF50)
        InferenceUiState.Generating -> "Generating" to MaterialTheme.colorScheme.primary
        InferenceUiState.Error -> "Error" to MaterialTheme.colorScheme.error
    }

    Surface(
        color = color.copy(alpha = 0.1f),
        shape = RoundedCornerShape(12.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
        )
    }
}

@Composable
fun CompactMetricItem(
    label: String,
    value: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun RagSourcesCard(
    ragMessage: RagMessage,
    modifier: Modifier = Modifier
) {
    var expandedItems by remember { mutableStateOf(setOf<Int>()) }

    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            // Header with icon
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(6.dp)
            ) {
                Icon(
                    Icons.Default.Search,
                    contentDescription = "RAG Sources",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(16.dp)
                )

                Spacer(modifier = Modifier.width(8.dp))

                Text(
                    text = "Reading from ${ragMessage.documents.size} sources...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    fontWeight = FontWeight.Medium
                )
            }

            // Sources list
            ragMessage.documents.forEachIndexed { index, chunk ->
                val isExpanded = expandedItems.contains(index)

                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 6.dp)
                        .clickable {
                            expandedItems = if (isExpanded) {
                                expandedItems - index
                            } else {
                                expandedItems + index
                            }
                        },
                    color = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(8.dp)
                    ) {
                        // File name and score
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.weight(1f)
                            ) {
                                Text(
                                    text = "• ",
                                    color = MaterialTheme.colorScheme.primary,
                                    style = MaterialTheme.typography.bodyMedium
                                )

                                Text(
                                    text = chunk.file,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurface,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    modifier = Modifier.weight(1f)
                                )
                            }

                            // Score badge
                            Surface(
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Text(
                                    text = "${(chunk.score * 100).toInt()}%",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                )
                            }

                            // Expand/collapse icon
                            Icon(
                                if (isExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = if (isExpanded) "Collapse" else "Expand",
                                tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                                modifier = Modifier.size(16.dp)
                            )
                        }

                        // Expanded content
                        if (isExpanded) {
                            Spacer(modifier = Modifier.height(8.dp))

                            Surface(
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Column(
                                    modifier = Modifier.padding(8.dp)
                                ) {
                                    Text(
                                        text = "Content Preview:",
                                        style = MaterialTheme.typography.labelMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.padding(bottom = 4.dp)
                                    )

                                    Text(
                                        text = chunk.content.take(200) + if (chunk.content.length > 200) "..." else "",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        lineHeight = 16.sp
                                    )

                                    if (chunk.content.length > 200) {
                                        Text(
                                            text = "Tap to see full content",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.padding(top = 4.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                if (index < ragMessage.documents.size - 1) {
                    Spacer(modifier = Modifier.height(4.dp))
                }
            }
        }
    }
}