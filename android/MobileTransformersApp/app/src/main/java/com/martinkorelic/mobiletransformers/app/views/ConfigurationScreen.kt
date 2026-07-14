package com.martinkorelic.mobiletransformers.app.views

import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.DeviceOptions
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.SamplingOptions
import com.martinkorelic.mobiletransformers.SchedulerConfig
import com.martinkorelic.mobiletransformers.app.viewmodels.ConfigurationViewModel


// ConfigurationScreen.kt
@Composable
fun ConfigurationScreen(viewModel: ConfigurationViewModel) {
    var selectedConfigTab by remember { mutableStateOf(0) }
    val configTabs = listOf("Generation Config", "Training Config")

    Column(modifier = Modifier
        .fillMaxSize()
        .padding(16.dp)) {
        TabRow(selectedTabIndex = selectedConfigTab) {
            configTabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedConfigTab == index,
                    onClick = { selectedConfigTab = index },
                    text = { Text(title) }
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        when (selectedConfigTab) {
            0 -> GenerationConfigScreen(viewModel)
            1 -> TrainingConfigScreen(viewModel)
        }
    }
}

@Composable
fun GenerationConfigScreen(viewModel: ConfigurationViewModel) {
    val config = viewModel.generationConfig.value
    val ragConfig = viewModel.ragConfig.value
    val availableModels = viewModel.availableModels
    val ragEnabled = viewModel.ragEnabled.value
    val isRagAvailable = viewModel.isRagAvailable.value

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Model Configuration",
                        style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )

                    // Model Name Dropdown
                    ModelDropdown(
                        selectedModel = config.repoName,
                        availableModels = availableModels,
                        onModelSelected = { viewModel.onGenerationModelChanged(it) },
                        label = "Model Name"
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Type Dropdown
                    TypeDropdown(
                        selectedType = config.type,
                        onTypeSelected = {
                            viewModel.updateGenerationConfig(config.copy(type = it))
                        }
                    )
                }
            }
        }

        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Generation Settings",
                        style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )

                    // Max Sequence Length
                    IntegerField(
                        value = config.maxSequenceLength,
                        onValueChange = {
                            viewModel.updateGenerationConfig(config.copy(maxSequenceLength = it))
                        },
                        label = "Max Sequence Length"
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Time Step Update
                    IntegerField(
                        value = config.timeStepUpdate,
                        onValueChange = {
                            viewModel.updateGenerationConfig(config.copy(timeStepUpdate = it))
                        },
                        label = "Time Step Update"
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // System Prompt
                    OutlinedTextField(
                        value = config.systemPrompt ?: "",
                        onValueChange = {
                            viewModel.updateGenerationConfig(
                                config.copy(systemPrompt = it.takeIf { it.isNotBlank() })
                            )
                        },
                        label = { Text("System Prompt") },
                        modifier = Modifier.fillMaxWidth(),
                        minLines = 3
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    // Checkboxes
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        CheckboxWithLabel(
                            checked = config.trackMetrics,
                            onCheckedChange = {
                                viewModel.updateGenerationConfig(config.copy(trackMetrics = it))
                            },
                            label = "Track Metrics"
                        )

                        CheckboxWithLabel(
                            checked = config.loadMergedWeights,
                            onCheckedChange = {
                                viewModel.updateGenerationConfig(config.copy(loadMergedWeights = it))
                            },
                            label = "Load Merged Weights"
                        )
                    }
                }
            }
        }
        
        item { 
           RagConfigurationCard(ragConfig = ragConfig, ragEnabled = ragEnabled && isRagAvailable, onRagEnabledChanged = { enabled ->
               viewModel.updateRagEnabled(enabled)
           }, onRagConfigChanged = { newConfig ->
               viewModel.updateRagConfig(newConfig)
           })
        }

        item {
            SamplingOptionsCard(
                sampling = config.sampling,
                onSamplingChanged = {
                    viewModel.updateGenerationConfig(config.copy(sampling = it))
                }
            )
        }

        item {
            DeviceOptionsCard(
                deviceOptions = config.deviceOptions,
                onDeviceOptionsChanged = {
                    viewModel.updateGenerationConfig(config.copy(deviceOptions = it))
                },
                isInference = true
            )
        }
    }
}

@Composable
fun TrainingConfigScreen(viewModel: ConfigurationViewModel) {
    val config = viewModel.trainingConfig.value
    val availableModels = viewModel.availableModels

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Model Configuration",
                        style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )

                    // Model Name Dropdown
                    ModelDropdown(
                        selectedModel = config.repoName,
                        availableModels = availableModels,
                        onModelSelected = { viewModel.onTrainingModelChanged(it) },
                        label = "Model Name"
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Task Name
                    OutlinedTextField(
                        value = config.taskName,
                        onValueChange = {
                            viewModel.updateTrainingConfig(config.copy(taskName = it))
                        },
                        label = { Text("Task Name") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Train File
                    OutlinedTextField(
                        value = config.datasetOptions.trainFile,
                        onValueChange = {
                            viewModel.updateTrainingConfig(
                                config.copy(
                                    datasetOptions = config.datasetOptions.copy(trainFile = it)
                                )
                            )
                        },
                        label = { Text("Train File") },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }

        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Training Parameters",
                        style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )

                    Row(modifier = Modifier.fillMaxWidth()) {
                        IntegerField(
                            value = config.batchSize,
                            onValueChange = {
                                viewModel.updateTrainingConfig(config.copy(batchSize = it))
                            },
                            label = "Batch Size",
                            modifier = Modifier.weight(1f)
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        IntegerField(
                            value = config.numTrainEpochs,
                            onValueChange = {
                                viewModel.updateTrainingConfig(config.copy(numTrainEpochs = it))
                            },
                            label = "Epochs",
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    Row(modifier = Modifier.fillMaxWidth()) {
                        IntegerField(
                            value = config.datasetOptions.maxSequenceLength ?: 0,
                            onValueChange = {
                                viewModel.updateTrainingConfig(
                                    config.copy(
                                        datasetOptions = config.datasetOptions.copy(maxSequenceLength = it)
                                    )
                                )
                            },
                            label = "Max Seq Length",
                            modifier = Modifier.weight(1f)
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        NullableIntegerField(
                            value = config.maxSteps,
                            onValueChange = {
                                viewModel.updateTrainingConfig(config.copy(maxSteps = it))
                            },
                            label = "Max Steps",
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    Row(modifier = Modifier.fillMaxWidth()) {
                        IntegerField(
                            value = config.saveSteps,
                            onValueChange = {
                                viewModel.updateTrainingConfig(config.copy(saveSteps = it))
                            },
                            label = "Save Steps",
                            modifier = Modifier.weight(1f)
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        IntegerField(
                            value = config.gradAccumSteps,
                            onValueChange = {
                                viewModel.updateTrainingConfig(config.copy(gradAccumSteps = it))
                            },
                            label = "Grad Accum Steps",
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }

        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Dataset Configuration",
                        style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )

                    Row(modifier = Modifier.fillMaxWidth()) {
                        IntegerField(
                            value = config.datasetOptions.maxDatasetLength ?: 0,
                            onValueChange = {
                                viewModel.updateTrainingConfig(
                                    config.copy(
                                        datasetOptions = config.datasetOptions.copy(maxDatasetLength = it)
                                    )
                                )
                            },
                            label = "Max Dataset Length",
                            modifier = Modifier.weight(1f)
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        IntegerField(
                            value = config.datasetOptions.datasetBatchSize ?: 0,
                            onValueChange = {
                                viewModel.updateTrainingConfig(
                                    config.copy(
                                        datasetOptions = config.datasetOptions.copy(datasetBatchSize = it)
                                    )
                                )
                            },
                            label = "Dataset Batch Size",
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    // Checkboxes
                    Column {
                        CheckboxWithLabel(
                            checked = config.datasetOptions.removeLongSamples,
                            onCheckedChange = {
                                viewModel.updateTrainingConfig(
                                    config.copy(
                                        datasetOptions = config.datasetOptions.copy(removeLongSamples = it)
                                    )
                                )
                            },
                            label = "Remove Long Samples"
                        )

                        CheckboxWithLabel(
                            checked = config.mergeWeightsAtEnd,
                            onCheckedChange = {
                                viewModel.updateTrainingConfig(config.copy(mergeWeightsAtEnd = it))
                            },
                            label = "Merge Weights at End"
                        )

                        CheckboxWithLabel(
                            checked = config.saveModelAtEnd,
                            onCheckedChange = {
                                viewModel.updateTrainingConfig(config.copy(saveModelAtEnd = it))
                            },
                            label = "Save Model at End"
                        )
                    }
                }
            }
        }

        item {
            SchedulerConfigCard(
                schedulerType = config.schedulerType,
                schedulerConfig = config.schedulerConfig,
                onSchedulerChanged = { type, schedulerConfig ->
                    viewModel.updateTrainingConfig(
                        config.copy(
                            schedulerType = type,
                            schedulerConfig = schedulerConfig
                        )
                    )
                }
            )
        }

        item {
            DeviceOptionsCard(
                deviceOptions = config.deviceOptions,
                onDeviceOptionsChanged = {
                    viewModel.updateTrainingConfig(config.copy(deviceOptions = it))
                }
            )
        }
    }
}

// Helper Composables
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDropdown(
    selectedModel: String,
    availableModels: List<String>,
    onModelSelected: (String) -> Unit,
    label: String,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = !expanded },
        modifier = modifier
    ) {
        OutlinedTextField(
            value = selectedModel,
            onValueChange = {},
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor()
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            availableModels.forEach { model ->
                DropdownMenuItem(
                    text = { Text(model) },
                    onClick = {
                        onModelSelected(model)
                        expanded = false
                    }
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TypeDropdown(
    selectedType: String,
    onTypeSelected: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    val types = listOf("native")

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = !expanded }
    ) {
        OutlinedTextField(
            value = selectedType,
            onValueChange = {},
            readOnly = true,
            label = { Text("Type") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor()
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            types.forEach { type ->
                DropdownMenuItem(
                    text = { Text(type) },
                    onClick = {
                        onTypeSelected(type)
                        expanded = false
                    }
                )
            }
        }
    }
}

@Composable
fun IntegerField(
    value: Int,
    onValueChange: (Int) -> Unit,
    label: String,
    modifier: Modifier = Modifier,
    enabled : Boolean = true
) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = { newValue ->
            newValue.toIntOrNull()?.let { onValueChange(it) }
        },
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = modifier,
        enabled = enabled
    )
}

@Composable
fun NullableIntegerField(
    value: Int?,
    onValueChange: (Int?) -> Unit,
    label: String,
    modifier: Modifier = Modifier
) {
    OutlinedTextField(
        value = value?.toString() ?: "",
        onValueChange = { newValue ->
            if (newValue.isBlank()) {
                onValueChange(null)
            } else {
                newValue.toIntOrNull()?.let { onValueChange(it) }
            }
        },
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = modifier
    )
}

@Composable
fun FloatField(
    value: Float,
    onValueChange: (Float) -> Unit,
    label: String,
    modifier: Modifier = Modifier
) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = { newValue ->
            newValue.toFloatOrNull()?.let { onValueChange(it) }
        },
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        modifier = modifier
    )
}

@Composable
fun CheckboxWithLabel(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    label: String,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Checkbox(
            checked = checked,
            onCheckedChange = onCheckedChange
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(text = label)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SamplingOptionsCard(
    sampling: SamplingOptions,
    onSamplingChanged: (SamplingOptions) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Sampling Options",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(bottom = 12.dp)
            )

            // Method Dropdown
            var methodExpanded by remember { mutableStateOf(false) }
            val methods = listOf("greedy", "top_p", "top_k")

            ExposedDropdownMenuBox(
                expanded = methodExpanded,
                onExpandedChange = { methodExpanded = !methodExpanded }
            ) {
                OutlinedTextField(
                    value = sampling.method,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Sampling Method") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = methodExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = methodExpanded,
                    onDismissRequest = { methodExpanded = false }
                ) {
                    methods.forEach { method ->
                        DropdownMenuItem(
                            text = { Text(method) },
                            onClick = {
                                onSamplingChanged(sampling.copy(method = method))
                                methodExpanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(modifier = Modifier.fillMaxWidth()) {
                FloatField(
                    value = sampling.temperature,
                    onValueChange = { onSamplingChanged(sampling.copy(temperature = it)) },
                    label = "Temperature",
                    modifier = Modifier.weight(1f)
                )

                Spacer(modifier = Modifier.width(8.dp))

                FloatField(
                    value = sampling.topP,
                    onValueChange = { onSamplingChanged(sampling.copy(topP = it)) },
                    label = "Top P",
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(modifier = Modifier.fillMaxWidth()) {
                IntegerField(
                    value = sampling.topK,
                    onValueChange = { onSamplingChanged(sampling.copy(topK = it)) },
                    label = "Top K",
                    modifier = Modifier.weight(1f)
                )

                Spacer(modifier = Modifier.width(8.dp))

                IntegerField(
                    value = sampling.seed,
                    onValueChange = { onSamplingChanged(sampling.copy(seed = it)) },
                    label = "Seed",
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DeviceOptionsCard(
    deviceOptions: DeviceOptions,
    onDeviceOptionsChanged: (DeviceOptions) -> Unit,
    isInference: Boolean = false
) {
    val context = LocalContext.current

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Device Options",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(bottom = 12.dp)
            )

            // Execution Provider Dropdown
            var providerExpanded by remember { mutableStateOf(false) }
            val providers = listOf("cpu", "nnapi", "xnnpack")

            ExposedDropdownMenuBox(
                expanded = providerExpanded,
                onExpandedChange = { providerExpanded = !providerExpanded }
            ) {
                OutlinedTextField(
                    value = deviceOptions.executionProvider,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Execution Provider") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = providerExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = providerExpanded,
                    onDismissRequest = { providerExpanded = false }
                ) {
                    providers.forEach { provider ->
                        DropdownMenuItem(
                            text = { Text(provider) },
                            onClick = {
                                onDeviceOptionsChanged(deviceOptions.copy(executionProvider = provider))
                                providerExpanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Core Config ID Dropdown (opt1, opt2, opt3)
            var coreConfigExpanded by remember { mutableStateOf(false) }
            val coreConfigs = listOf("opt1", "opt2", "opt3")

            ExposedDropdownMenuBox(
                expanded = coreConfigExpanded,
                onExpandedChange = { coreConfigExpanded = !coreConfigExpanded }
            ) {
                OutlinedTextField(
                    value = deviceOptions.coreConfigId,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Core Config ID") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = coreConfigExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = coreConfigExpanded,
                    onDismissRequest = { coreConfigExpanded = false }
                ) {
                    coreConfigs.forEach { config ->
                        DropdownMenuItem(
                            text = { Text(config) },
                            onClick = {
                                onDeviceOptionsChanged(deviceOptions.copy(coreConfigId = config))
                                coreConfigExpanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Memory Config ID Dropdown (high_perf, low_mem)
            var memoryConfigExpanded by remember { mutableStateOf(false) }
            val memoryConfigs = listOf("high_perf", "low_mem")

            ExposedDropdownMenuBox(
                expanded = memoryConfigExpanded,
                onExpandedChange = { memoryConfigExpanded = !memoryConfigExpanded }
            ) {
                OutlinedTextField(
                    value = deviceOptions.memoryConfigId,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Memory Config ID") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = memoryConfigExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = memoryConfigExpanded,
                    onDismissRequest = { memoryConfigExpanded = false }
                ) {
                    memoryConfigs.forEach { config ->
                        DropdownMenuItem(
                            text = { Text(config) },
                            onClick = {
                                // Show warning toast if high_perf is selected in inference mode
                                if (isInference && config == "low_mem") {
                                    Toast.makeText(
                                        context,
                                        "Warning: Low memory option is very likely to cause a crash with inference!",
                                        Toast.LENGTH_LONG
                                    ).show()
                                }
                                onDeviceOptionsChanged(deviceOptions.copy(memoryConfigId = config))
                                memoryConfigExpanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            CheckboxWithLabel(
                checked = deviceOptions.enableProfiling,
                onCheckedChange = {
                    onDeviceOptionsChanged(deviceOptions.copy(enableProfiling = it))
                },
                label = "Enable Profiling"
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RagConfigurationCard(
    ragConfig: ORTRagConfig,
    ragEnabled: Boolean,
    onRagEnabledChanged: (Boolean) -> Unit,
    onRagConfigChanged: (ORTRagConfig) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "RAG Configuration",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(bottom = 12.dp)
            )

            // Enable RAG Switch
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Enable RAG",
                    style = MaterialTheme.typography.bodyLarge
                )
                Switch(
                    checked = ragEnabled,
                    onCheckedChange = onRagEnabledChanged
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Search Type Dropdown
            var searchTypeExpanded by remember { mutableStateOf(false) }
            val searchTypes = listOf("semantic", "text")

            ExposedDropdownMenuBox(
                expanded = searchTypeExpanded,
                onExpandedChange = {
                    if (ragEnabled) searchTypeExpanded = !searchTypeExpanded
                }
            ) {
                OutlinedTextField(
                    value = ragConfig.searchType,
                    onValueChange = {},
                    readOnly = true,
                    enabled = ragEnabled,
                    label = { Text("Search Type") },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(
                            expanded = searchTypeExpanded
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor(),
                    colors = OutlinedTextFieldDefaults.colors(
                        disabledTextColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f),
                        disabledBorderColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f),
                        disabledLabelColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f)
                    )
                )
                ExposedDropdownMenu(
                    expanded = searchTypeExpanded,
                    onDismissRequest = { searchTypeExpanded = false }
                ) {
                    searchTypes.forEach { searchType ->
                        DropdownMenuItem(
                            text = { Text(searchType) },
                            onClick = {
                                onRagConfigChanged(ragConfig.copy(searchType = searchType))
                                searchTypeExpanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Top K Field
            IntegerField(
                value = ragConfig.topK,
                onValueChange = { onRagConfigChanged(ragConfig.copy(topK = it)) },
                label = "Top K Results",
                enabled = ragEnabled,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SchedulerConfigCard(
    schedulerType: String,
    schedulerConfig: SchedulerConfig,
    onSchedulerChanged: (String, SchedulerConfig) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Scheduler Configuration",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(bottom = 12.dp)
            )

            // Scheduler Type Dropdown
            var typeExpanded by remember { mutableStateOf(false) }
            val types = listOf("linear", "cosine")

            ExposedDropdownMenuBox(
                expanded = typeExpanded,
                onExpandedChange = { typeExpanded = !typeExpanded }
            ) {
                OutlinedTextField(
                    value = schedulerType,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Scheduler Type") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = typeExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = typeExpanded,
                    onDismissRequest = { typeExpanded = false }
                ) {
                    types.forEach { type ->
                        DropdownMenuItem(
                            text = { Text(type) },
                            onClick = {
                                val newConfig = when (type) {
                                    "linear" -> SchedulerConfig.Linear()
                                    "cosine" -> SchedulerConfig.Cosine()
                                    else -> schedulerConfig
                                }
                                onSchedulerChanged(type, newConfig)
                                typeExpanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            when (schedulerConfig) {
                is SchedulerConfig.Linear -> {
                    Text(
                        text = "Linear Scheduler",
                        style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )

                    FloatField(
                        value = schedulerConfig.learningRate,
                        onValueChange = {
                            onSchedulerChanged(schedulerType, schedulerConfig.copy(learningRate = it))
                        },
                        label = "Learning Rate",
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Row(modifier = Modifier.fillMaxWidth()) {
                        FloatField(
                            value = schedulerConfig.startFactor,
                            onValueChange = {
                                onSchedulerChanged(schedulerType, schedulerConfig.copy(startFactor = it))
                            },
                            label = "Start Factor",
                            modifier = Modifier.weight(1f)
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        FloatField(
                            value = schedulerConfig.endFactor,
                            onValueChange = {
                                onSchedulerChanged(schedulerType, schedulerConfig.copy(endFactor = it))
                            },
                            label = "End Factor",
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
                is SchedulerConfig.Cosine -> {
                    Text(
                        text = "Cosine Scheduler",
                        style = MaterialTheme.typography.headlineSmall,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )

                    FloatField(
                        value = schedulerConfig.learningRate,
                        onValueChange = {
                            onSchedulerChanged(schedulerType, schedulerConfig.copy(learningRate = it))
                        },
                        label = "Learning Rate",
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Row(modifier = Modifier.fillMaxWidth()) {
                        FloatField(
                            value = schedulerConfig.minLearningRate,
                            onValueChange = {
                                onSchedulerChanged(schedulerType, schedulerConfig.copy(minLearningRate = it))
                            },
                            label = "Min Learning Rate",
                            modifier = Modifier.weight(1f)
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        IntegerField(
                            value = schedulerConfig.warmupSteps,
                            onValueChange = {
                                onSchedulerChanged(schedulerType, schedulerConfig.copy(warmupSteps = it))
                            },
                            label = "Warmup Steps",
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }
    }
}