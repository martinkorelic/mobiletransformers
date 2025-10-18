package com.martinkorelic.ortmobile

import org.json.JSONObject

class DataCollatorForSupervisedDataset(private val tokenizer: ORTTokenizerNative) {

    fun collate(batch: List<ORTDataCurator.TrainingSample>): CollatedBatch {
        val padToken = tokenizer.padToken
        val padLabel = -100

        val maxLength = batch.maxOf { it.inputIds.size }

        val inputIdsPadded = batch.map { sample ->
            val padded = sample.inputIds + List(maxLength - sample.inputIds.size) { padToken }
            padded.map { it?.toLong() ?: 0 }.toLongArray()
        }

        val labelsPadded = batch.map { sample ->
            val padded = sample.labels + List(maxLength - sample.labels.size) { padLabel }
            padded.map { it.toLong() }.toLongArray()
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