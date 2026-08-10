package com.martinkorelic.mobiletransformers

import org.json.JSONObject

/**
 * Pads a batch into the tensors a training step binds.
 *
 * Takes the pad token rather than the whole tokenizer: that is all it ever used, and depending on
 * [ORTTokenizerNative] made the collator un-unit-testable (its `init` loads the native library). The
 * label-padding rule below is the half that decides the exported graph's label rank, so it needs to be
 * pinned on the host — the same reason `GenerationInputs`, `training_inputs.h` and `layer_name.h` are
 * shaped as pure seams.
 */
class DataCollatorForSupervisedDataset(private val padToken: Int?) {

    constructor(tokenizer: ORTTokenizerNative) : this(tokenizer.padToken)

    fun collate(batch: List<ORTDataCurator.TrainingSample>): CollatedBatch {
        val padToken = this.padToken
        val padLabel = -100

        val maxLength = batch.maxOf { it.inputIds.size }

        val inputIdsPadded = batch.map { sample ->
            val padded = sample.inputIds + List(maxLength - sample.inputIds.size) { padToken }
            padded.map { it?.toLong() ?: 0 }.toLongArray()
        }

        // Per-SEQUENCE labels must NOT be padded to the sequence length.
        //
        // Padding them would turn one label per example back into one per token, which is precisely
        // what the native side uses to infer the label rank: `training_inputs.h::labels_shape` reads
        // `batch*seq` as `[batch, seq]` and `batch` as `[batch]`. Padding here would make a
        // classification graph receive a per-token label tensor and either throw inside ORT or, worse,
        // bind a wrongly-shaped tensor. The two halves have to agree, and this is the half that decides.
        val labelsPadded = batch.map { sample ->
            if (sample.perSequenceLabel) {
                sample.labels.map { it.toLong() }.toLongArray()
            } else {
                val padded = sample.labels + List(maxLength - sample.labels.size) { padLabel }
                padded.map { it.toLong() }.toLongArray()
            }
        }

        // Create attention mask: 1 for non-pad tokens, 0 for pad tokens
        val attentionMaskPadded = batch.map { sample ->
            // Original tokens get attention (1), padded tokens don't (0)
            val originalLength = sample.inputIds.size
            val paddingLength = maxLength - originalLength

            val attentionMask = List(originalLength) { 1L } + List(paddingLength) { 0L }
            attentionMask.toLongArray()
        }

        return CollatedBatch(
            inputIds = inputIdsPadded.toTypedArray(),
            labels = labelsPadded.toTypedArray(),
            attentionMask = attentionMaskPadded.toTypedArray(),
            sequenceLength = maxLength,
            batchSize = batch.size
        )
    }

    data class CollatedBatch(
        val inputIds: Array<LongArray>,
        val labels: Array<LongArray>,
        val attentionMask: Array<LongArray>,
        val sequenceLength: Int,
        val batchSize: Int
    )
}

// Interface for preprocessing functions
interface TaskPreprocessor {
    fun preprocess(json: JSONObject): Pair<String, String>

    /**
     * The per-SEQUENCE class label for this example, or `null` when the objective is token-level.
     *
     * #33: a sequence-classification graph declares `labels[batch]` — one label per example, not one
     * per token — which is the axis that separates it from every decoder objective
     * (`TaskSpec.label_shape`). The native binder already handles both: `training_inputs.h` derives the
     * label rank from how many label elements the caller actually supplies.
     *
     * Additive with a default rather than a changed signature, because [TaskPreprocessor] is public
     * API — callers pass their own through `DatasetConfig.customPreprocess`. Returning `null` (the
     * default) keeps every existing preprocessor on the token-level path unchanged.
     *
     * When this returns non-null, [preprocess] is not consulted for the label; only its input text is.
     */
    fun classLabel(json: JSONObject): Int? = null
}

// Factory function using the interface
fun getPreprocessFunctionForTask(
    taskName: String,
    customPreprocess: TaskPreprocessor? = null
): TaskPreprocessor {
    if (customPreprocess != null) {
        return customPreprocess
    }

    return when (taskName.lowercase()) {
        "logiqa" -> LogiqaPreprocessor
        "boolq" -> BoolqPreprocessor
        "mini_personalqa" -> MiniPersonalQAPreprocessor
        "mini_recommendation" -> MiniRecommendationPreprocessor
        "cola" -> CoLAPreprocessor
        "cola_cls" -> CoLAClassificationPreprocessor
        else -> throw IllegalArgumentException("Unsupported task: $taskName. Please provide a customPreprocess function.")
    }
}

// Concrete implementations
object LogiqaPreprocessor : TaskPreprocessor {
    override fun preprocess(json: JSONObject): Pair<String, String> {
        val labelMap = mapOf(0 to "A", 1 to "B", 2 to "C", 3 to "D")

        val article = json.optString("text", "")
        val question = json.optString("question", "")
        val optionsArray = json.optJSONArray("options")
        val answer = json.optInt("answer", -1)

        var options = ""
        if (optionsArray != null) {
            for (i in 0 until optionsArray.length()) {
                val option = optionsArray.optString(i, "")
                options += "${labelMap[i]} $option\n"
            }
        }

        val input = "Write a multi-choice question for the following article:\n" +
                "Article: $article\n" +
                "Question: $question\n" +
                "Options: $options" +
                "Answer: \n\n "

        val label = labelMap[answer] ?: ""

        return input to label
    }
}

object BoolqPreprocessor : TaskPreprocessor {
    override fun preprocess(json: JSONObject): Pair<String, String> {
        val question = json.optString("question", "")
        val passage = json.optString("passage", "")
        val answer = json.optString("answer", "")

        val input = "Q: $question?\nP: $passage\nA: \n\n "

        return input to answer
    }
}

object MiniPersonalQAPreprocessor : TaskPreprocessor {
    override fun preprocess(json: JSONObject): Pair<String, String> {
        val questionText = json.optString("question", "")
        val choicesObject = json.optJSONObject("choices")
        val correctAnswer = json.optString("correct_answer", "")

        // Build the formatted question
        var formatted = "Question: $questionText\n\n"

        if (choicesObject != null) {
            val keys = choicesObject.keys()
            while (keys.hasNext()) {
                val choiceKey = keys.next()
                val choiceValue = choicesObject.optString(choiceKey, "")
                formatted += "$choiceKey: $choiceValue\n"
            }
        }

        val input = formatted + "\n\nAnswer: "
        val label = correctAnswer

        return input to label
    }
}

object MiniRecommendationPreprocessor : TaskPreprocessor {
    override fun preprocess(json: JSONObject): Pair<String, String> {
        val userQuery = json.optString("prompt", "")
        val recommendation = json.optString("recommendation", "")

        val formatted = "Recommend best actions based on this user query: $userQuery"

        val input = formatted + "\n\nAnswer: "

        val label = recommendation

        return input to label
    }
}

object CoLAPreprocessor : TaskPreprocessor {
    override fun preprocess(json: JSONObject): Pair<String, String> {
        val sentence = json.optString("sentence", "")
        val label = json.optInt("label", 0)

        val input = "Is this sentence grammatically acceptable? $sentence\nA: "
        val output = if (label == 1) "acceptable" else "unacceptable"

        return input to output
    }
}

/**
 * The same CoLA data as a real sequence-classification objective (#33).
 *
 * [CoLAPreprocessor] stringifies the class index into `"acceptable"`/`"unacceptable"` so it fits the
 * decoder's text-to-text contract — a workaround for having only one label shape. This one keeps the
 * label a class index and lets the graph's `labels[batch]` input receive it directly.
 *
 * Registered as `cola_cls` rather than replacing `cola`, because the two are genuinely different
 * objectives over the same file and a package declares which one it trains.
 */
object CoLAClassificationPreprocessor : TaskPreprocessor {
    override fun preprocess(json: JSONObject): Pair<String, String> =
        json.optString("sentence", "") to ""

    override fun classLabel(json: JSONObject): Int = json.optInt("label", 0)
}