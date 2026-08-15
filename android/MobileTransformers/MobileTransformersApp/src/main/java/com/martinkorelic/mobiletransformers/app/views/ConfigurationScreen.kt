package com.martinkorelic.mobiletransformers.app.views

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.martinkorelic.mobiletransformers.Tasks
import com.martinkorelic.mobiletransformers.app.viewmodels.ConfigurationViewModel
import com.martinkorelic.mobiletransformers.app.viewmodels.label
import com.martinkorelic.mobiletransformers.app.viewmodels.peftDescription
import com.martinkorelic.mobiletransformers.app.viewmodels.peftOf
import com.martinkorelic.mobiletransformers.app.viewmodels.peftOptions
import com.martinkorelic.mobiletransformers.constants.CoreConfigId
import com.martinkorelic.mobiletransformers.constants.ExecutionProvider
import com.martinkorelic.mobiletransformers.constants.IndexingMode
import com.martinkorelic.mobiletransformers.constants.MemoryConfigId
import com.martinkorelic.mobiletransformers.constants.SamplingMethod
import com.martinkorelic.mobiletransformers.constants.SchedulerType
import com.martinkorelic.mobiletransformers.constants.SearchType

/**
 * The knobs, expressed through the public config types.
 *
 * If a setting here needed an `ORT*` type to express, that would be a facade gap for #17/#19. None
 * did — which is the result this screen reports.
 *
 * ### Why it is tabbed, and why so much of it is now pickers
 *
 * It was one scroll of ~25 fields with no grouping, and several of them were **text fields over
 * closed sets**: `task` had to be spelled exactly right from a list in a help paragraph, and the
 * device options were not editable at all despite being on every public config. A mistyped task is
 * accepted here and reported as `Unsupported task: …` minutes into a training run, on a different
 * screen — which is the worst possible place to learn about a typo. Anything with a knowable set of
 * values is a [Dropdown] or a [ChipPicker] now.
 */
@Composable
fun ConfigurationScreen(vm: ConfigurationViewModel) {
    var tab by remember { mutableIntStateOf(0) }

    Column(Modifier.fillMaxSize()) {
        SubTabs(listOf("Generation", "Training", "Dataset", "Retrieval", "Device"), tab) { tab = it }

        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            when (tab) {
                0 -> GenerationTab(vm)
                1 -> TrainingTab(vm)
                2 -> DatasetTab(vm)
                3 -> RetrievalTab(vm)
                else -> DeviceTab(vm)
            }

            OutlinedButton(onClick = vm::reset, modifier = Modifier.padding(16.dp)) {
                Text("Reset every section to SDK defaults")
            }
        }
    }
}

@Composable
private fun GenerationTab(vm: ConfigurationViewModel) {
    val gen by vm.generation.collectAsState()

    Section("Length and sampling") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField(
                "maxNewTokens", gen.maxNewTokens, range = 1..4096,
                hint = "how many tokens to generate before stopping",
                onChange = vm::setMaxNewTokens,
            )
            ChipPicker(
                "sampling method",
                SamplingMethod.entries,
                gen.sampling.method,
                { it.wire },
                vm::setSamplingMethod,
            )
            // Only the knobs the chosen method actually reads. Showing topK beside GREEDY invites the
            // reasonable conclusion that changing it will do something.
            when (gen.sampling.method) {
                SamplingMethod.GREEDY -> Text(
                    "Greedy takes the highest-probability token every step, so temperature, topK, " +
                        "topP and seed have no effect — the output is deterministic.",
                    style = MaterialTheme.typography.bodySmall,
                )
                SamplingMethod.TOP_K -> {
                    FloatField("temperature", gen.sampling.temperature, 0.01f..5f, onChange = vm::setTemperature)
                    IntField("topK", gen.sampling.topK, 1..1000, onChange = vm::setTopK)
                    IntField("seed", gen.sampling.seed, onChange = vm::setSeed)
                }
                SamplingMethod.TOP_P -> {
                    FloatField("temperature", gen.sampling.temperature, 0.01f..5f, onChange = vm::setTemperature)
                    FloatField("topP", gen.sampling.topP, 0f..1f, onChange = vm::setTopP)
                    IntField("seed", gen.sampling.seed, onChange = vm::setSeed)
                }
            }
        }
    }

    Section("Prompt and weights") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            TextField("systemPrompt", gen.systemPrompt ?: "", vm::setSystemPrompt)
            LabeledSwitch("loadMerged", gen.loadMerged, vm::setLoadMerged)
            Text(
                "loadMerged generates from the merged weights a training run wrote back into the " +
                    "inference graph. Off, you are generating from the package as it was pulled — " +
                    "which is how to compare before and after fine-tuning.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun TrainingTab(vm: ConfigurationViewModel) {
    val train by vm.train.collectAsState()
    val peft by vm.peft.collectAsState()

    Section("Run length") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField("epochs", train.epochs, 1..100, onChange = vm::setEpochs)
            IntField("batchSize", train.batchSize, 1..64, onChange = vm::setBatchSize)
            IntField(
                "maxSteps (0 = unbounded)", train.maxSteps ?: 0, 0..100_000,
                hint = "an upper bound, not a target",
            ) { vm.setMaxSteps(it.takeIf { v -> v > 0 }) }
            Text(
                "maxSteps is an upper bound — training also stops at the end of the epoch, so " +
                    "rows / batchSize wins when it is smaller. Measured: a run asking for 120 steps " +
                    "took 54 because the dataset held 108 rows.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }

    Section("Optimizer") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField(
                "gradientAccumulationSteps", train.gradientAccumulationSteps, 1..64,
                onChange = vm::setGradientAccumulationSteps,
            )
            Text(
                "The optimizer steps on globalStep % gradAccumSteps == 0. At the default of 4 a short " +
                    "bounded run can finish, report success on every callback, and apply no update " +
                    "at all — set it to 1 for a quick demo run.",
                style = MaterialTheme.typography.bodySmall,
            )
            FloatField("learningRate", train.learningRate, 0f..1f, onChange = vm::setLearningRate)
            ChipPicker("scheduler", SchedulerType.entries, train.scheduler, { it.wire }, vm::setScheduler)
            IntField("warmupSteps", train.warmupSteps, 0..10_000, onChange = vm::setWarmupSteps)
        }
    }

    Section("PEFT method") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Dropdown(
                label = "method",
                options = peftOptions,
                selected = peft.label,
                optionLabel = { it },
                describe = { peftDescription(it) },
                onSelect = { vm.setPeft(peftOf(it, peft.rank, peft.alpha)) },
            )
            IntField("rank", peft.rank, 1..256, onChange = vm::setPeftRank)
            IntField("alpha", peft.alpha, 1..256, onChange = vm::setPeftAlpha)
            Text(
                "PEFT topology is fixed at export time, so this selects and validates against what " +
                    "the installed package was built with rather than rewriting the graph. A " +
                    "mismatch is reported here, at selection, instead of failing a training run.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }

    Section("After the run") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            LabeledSwitch("mergeAtEnd", train.mergeAtEnd, vm::setMergeAtEnd)
            LabeledSwitch("resumeFromState", train.resumeFromState, vm::setResumeFromState)
        }
    }
}

@Composable
private fun DatasetTab(vm: ConfigurationViewModel) {
    val dataset by vm.dataset.collectAsState()
    val context = LocalContext.current
    val modelState by vm.modelState.collectAsState()
    // Re-read when the model changes: the files live inside the loaded package's train/ stage.
    val trainFiles = remember(modelState) { vm.availableTrainFiles(context) }

    Section("Which file, and how to read it") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (trainFiles.isEmpty()) {
                Text(
                    "No .jsonl files in the loaded package's train/ stage. Model packages ship no " +
                        "training data by design — use 'Install sample dataset' on the Training " +
                        "screen, or push your own into that directory.",
                    style = MaterialTheme.typography.bodySmall,
                )
                TextField("trainFile", dataset.trainFile, vm::setTrainFile)
            } else {
                Dropdown(
                    label = "trainFile",
                    options = trainFiles,
                    selected = dataset.trainFile.takeIf { it in trainFiles },
                    optionLabel = { it },
                    onSelect = vm::setTrainFile,
                )
            }

            Dropdown(
                label = "task (the preprocessor that parses it)",
                options = vm.taskOptions,
                selected = vm.taskOptions.firstOrNull { it.name == dataset.task },
                optionLabel = { it.name },
                describe = { it.description },
                onSelect = { vm.setTask(it.name) },
            )
            Text(
                "The task names come from the trainer's own dispatch, so this list cannot drift from " +
                    "what it accepts. Leaving it unset uses whatever the package declares; a package " +
                    "declaring nothing fails closed rather than guessing how to parse your rows.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }

    Section("Shape") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField(
                "maxSequenceLength", dataset.maxSequenceLength, 8..4096,
                hint = "tokens per example; longer costs memory quadratically in attention",
                onChange = vm::setMaxSequenceLength,
            )
            IntField(
                "maxDatasetLength", dataset.maxDatasetLength, 1..100_000,
                hint = "rows to read; caps a long run on a phone",
                onChange = vm::setMaxDatasetLength,
            )
        }
    }
}

@Composable
private fun RetrievalTab(vm: ConfigurationViewModel) {
    val rag by vm.rag.collectAsState()

    Section("Search") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField("topK", rag.topK, 1..100, hint = "chunks retrieved per query", onChange = vm::setTopKRag)
            ChipPicker("searchType", SearchType.entries, rag.searchType, { it.wire }, vm::setSearchType)
            FloatField(
                "minScore", rag.minScore.toFloat(), 0f..1f,
                hint = "drop matches below this cosine similarity",
            ) { vm.setMinScore(it.toDouble()) }
            Text(
                "similarityMetric is ${rag.similarityMetric} and read-only — the on-device vector " +
                    "store uses cosine similarity.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }

    Section("Chunking") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            IntField("chunkSize", rag.chunkSize, 32..4096, onChange = vm::setChunkSize)
            IntField(
                "chunkOverlap", rag.chunkOverlap, 0..1024,
                hint = "characters repeated between adjacent chunks",
                onChange = vm::setChunkOverlap,
            )
            ChipPicker(
                "indexingMode",
                IndexingMode.entries,
                rag.indexingMode,
                { it.wire },
                // DYNAMIC is a fail-closed stub in v1, so selecting it would produce a refusal. It is
                // shown because the enum has it, and the note says what it does.
                { },
            )
            Text(
                "indexingMode is ${rag.indexingMode.wire}. DYNAMIC is declared by the enum but fails " +
                    "closed in v1, so it is not selectable here.",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "The embedding identity (repo, file, dimension) defaults to whatever the package " +
                    "declares in embedding/rag_config.json. Hardcoding it would point the retriever " +
                    "at a vector width the package need not have.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun DeviceTab(vm: ConfigurationViewModel) {
    val device by vm.device.collectAsState()

    Section("Execution") {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                "These apply to generation, training and retrieval together. They were on every " +
                    "public config and editable from nowhere until this tab existed.",
                style = MaterialTheme.typography.bodySmall,
            )
            Dropdown(
                label = "executionProvider",
                options = ExecutionProvider.entries,
                selected = device.executionProvider,
                optionLabel = { it.wire },
                describe = {
                    when (it) {
                        ExecutionProvider.CPU -> "the guaranteed floor; works on every device"
                        ExecutionProvider.XNNPACK -> "optimized CPU kernels; usually the fastest here"
                        ExecutionProvider.NNAPI -> "vendor accelerator, where the device offers one"
                    }
                },
                onSelect = vm::setExecutionProvider,
            )
            Dropdown(
                label = "coreConfigId",
                options = CoreConfigId.entries,
                selected = device.coreConfigId,
                optionLabel = { it.wire },
                onSelect = vm::setCoreConfig,
            )
            Dropdown(
                label = "memoryConfigId",
                options = MemoryConfigId.entries,
                selected = device.memoryConfigId,
                optionLabel = { it.wire },
                describe = {
                    when (it) {
                        MemoryConfigId.LOW_MEM -> "smaller arenas; for devices near their ceiling"
                        MemoryConfigId.HIGH_PERF -> "larger arenas; the default"
                    }
                },
                onSelect = vm::setMemoryConfig,
            )
            LabeledSwitch("enableProfiling", device.enableProfiling, vm::setProfiling)
            Text(
                "Profiling writes an ONNX Runtime trace beside the model. Useful once, expensive " +
                    "every time — it slows the session it measures.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }

    Section("When these take effect") {
        Text(
            "A session reads its device options when it is created, so changing them applies to the " +
                "next load, the next training run or the next ingest — not to the session already " +
                "open. Reload from Models to apply them to generation now.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
