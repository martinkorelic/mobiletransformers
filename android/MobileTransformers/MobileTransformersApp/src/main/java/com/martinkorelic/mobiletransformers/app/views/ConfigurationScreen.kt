package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.app.viewmodels.ConfigurationViewModel
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.constants.SearchType

/**
 * The knobs, expressed through the public config types.
 *
 * If a setting here needed an `ORT*` type to express, that would be a facade gap for #17/#19. None
 * did — which is the result this screen reports.
 */
@Composable
fun ConfigurationScreen(vm: ConfigurationViewModel) {
    val gen by vm.generation.collectAsState()
    val train by vm.train.collectAsState()
    val rag by vm.rag.collectAsState()
    val dataset by vm.dataset.collectAsState()

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Section("Generation") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                IntField("maxNewTokens", gen.maxNewTokens, vm::setMaxNewTokens)
                Text("sampling method", style = MaterialTheme.typography.labelMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SamplingMethod.entries.forEach { m ->
                        FilterChip(
                            selected = gen.sampling.method == m,
                            onClick = { vm.setSamplingMethod(m) },
                            label = { Text(m.name) },
                        )
                    }
                }
                FloatField("temperature", gen.sampling.temperature, vm::setTemperature)
                IntField("topK", gen.sampling.topK, vm::setTopK)
                FloatField("topP", gen.sampling.topP, vm::setTopP)
                IntField("seed", gen.sampling.seed, vm::setSeed)
                TextField("systemPrompt", gen.systemPrompt ?: "", vm::setSystemPrompt)
                LabeledSwitch("loadMerged", gen.loadMerged, vm::setLoadMerged)
            }
        }

        Section("Training") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                IntField("epochs", train.epochs, vm::setEpochs)
                IntField("batchSize", train.batchSize, vm::setBatchSize)
                IntField("maxSteps", train.maxSteps ?: 0) { vm.setMaxSteps(it.takeIf { v -> v > 0 }) }
                Text(
                    "maxSteps is an upper bound — training also stops at the end of the epoch, so " +
                        "rows / batchSize wins when it is smaller.",
                    style = MaterialTheme.typography.bodySmall,
                )
                IntField(
                    "gradientAccumulationSteps",
                    train.gradientAccumulationSteps,
                    vm::setGradientAccumulationSteps,
                )
                Text(
                    "The optimizer steps on globalStep % gradAccumSteps == 0. At the default of 4 a " +
                        "short bounded run can finish, report success on every callback, and apply no " +
                        "update at all.",
                    style = MaterialTheme.typography.bodySmall,
                )
                FloatField("learningRate", train.learningRate, vm::setLearningRate)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SchedulerType.entries.forEach { s ->
                        FilterChip(
                            selected = train.scheduler == s,
                            onClick = { vm.setScheduler(s) },
                            label = { Text(s.name) },
                        )
                    }
                }
                IntField("warmupSteps", train.warmupSteps, vm::setWarmupSteps)
                LabeledSwitch("mergeAtEnd", train.mergeAtEnd, vm::setMergeAtEnd)
                LabeledSwitch("resumeFromState", train.resumeFromState, vm::setResumeFromState)
            }
        }

        Section("Dataset") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                TextField("trainFile", dataset.trainFile, vm::setTrainFile)
                TextField("task", dataset.task ?: "", vm::setTask)
                Text(
                    "Registered tasks: logiqa, boolq, mini_personalqa, mini_recommendation, cola, " +
                        "cola_cls, mobile_actions. Empty means whatever the package declares.",
                    style = MaterialTheme.typography.bodySmall,
                )
                IntField("maxSequenceLength", dataset.maxSequenceLength, vm::setMaxSequenceLength)
                IntField("maxDatasetLength", dataset.maxDatasetLength, vm::setMaxDatasetLength)
            }
        }

        Section("Retrieval") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                IntField("topK", rag.topK, vm::setTopKRag)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SearchType.entries.forEach { s ->
                        FilterChip(
                            selected = rag.searchType == s,
                            onClick = { vm.setSearchType(s) },
                            label = { Text(s.name) },
                        )
                    }
                }
                FloatField("minScore", rag.minScore.toFloat()) { vm.setMinScore(it.toDouble()) }
                IntField("chunkSize", rag.chunkSize, vm::setChunkSize)
                IntField("chunkOverlap", rag.chunkOverlap, vm::setChunkOverlap)
                Text(
                    "similarityMetric is ${rag.similarityMetric} and read-only — the on-device vector " +
                        "store uses cosine similarity.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "The embedding identity (repo, file, dimension) defaults to whatever the package " +
                        "declares in embedding/rag_config.json. Hardcoding it here would point the " +
                        "retriever at a vector width the package need not have.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        OutlinedButton(onClick = vm::reset, modifier = Modifier.padding(16.dp)) {
            Text("Reset to SDK defaults")
        }
    }
}
