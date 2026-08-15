package com.martinkorelic.mobiletransformers.app

import android.content.Context
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.martinkorelic.mobiletransformers.packages.ModelFeature

/**
 * The curated list of models the app offers, loaded from `assets/model_catalog.json`.
 *
 * ### Why a catalog exists at all
 *
 * The only way in was a free-text repo id field pre-filled with one example. That asks a new user to
 * already know the answer to the question the app exists to answer, and it fails in a particularly
 * unhelpful way: a plain Hugging Face model id looks exactly like a valid entry, is accepted, and then
 * fails on the manifest request — because `fromPretrained` pulls an **exported package**, not a
 * base model. Typing `google/gemma-3-270m` produces a 404 that says nothing about that distinction.
 *
 * A catalog turns the first step into a choice among things known to work, and carries the size and
 * the feature groups next to each one so the cost of a multi-gigabyte pull is visible before it starts.
 *
 * ### Why a bundled asset rather than a Hub query
 *
 * The catalog must render with no network and no credentials — it is the first screen, and a user
 * with neither still needs to see what the app is for. Listing an organisation over the Hub API also
 * returns nothing for private or gated repos, which is what these are, so the live version of this
 * would be reliably empty exactly where it matters. Editing one JSON file to add a model is the
 * feature, not a limitation.
 */
object ModelCatalog {

    private const val ASSET = "model_catalog.json"

    private data class Wire(@SerializedName("models") val models: List<Entry> = emptyList())

    data class Entry(
        @SerializedName("repoId") val repoId: String = "",
        @SerializedName("displayName") val displayName: String = "",
        @SerializedName("description") val description: String = "",
        /** The upstream model this package was exported from — provenance, never a load key. */
        @SerializedName("baseModel") val baseModel: String = "",
        @SerializedName("task") val task: String = "",
        /** Inference group only; requesting train or rag adds to it. */
        @SerializedName("approxSizeMb") val approxSizeMb: Int = 0,
        @SerializedName("features") val features: List<String> = emptyList(),
        @SerializedName("requiresToken") val requiresToken: Boolean = false,
        /**
         * Whether this package is actually on the Hub yet.
         *
         * A catalog entry that 404s on tap is worse than no entry: the user cannot tell "not
         * published" from "your token is wrong" from "the app is broken". Unpublished entries stay
         * visible — they describe what the project supports — with Install disabled and the reason
         * on the card.
         */
        @SerializedName("published") val published: Boolean = false,
        @SerializedName("recommendedFor") val recommendedFor: String = "",
    ) {
        /** The feature groups to request when installing this entry, mapped to the SDK's enum. */
        val modelFeatures: Set<ModelFeature>
            get() = buildSet {
                add(ModelFeature.Inference)
                if ("train" in features) add(ModelFeature.Training)
                if ("rag" in features) add(ModelFeature.Rag)
            }

        val supportsTraining: Boolean get() = "train" in features
        val supportsRag: Boolean get() = "rag" in features

        val sizeLabel: String
            get() = if (approxSizeMb >= 1024) {
                "~%.1f GB".format(approxSizeMb / 1024.0)
            } else {
                "~$approxSizeMb MB"
            }
    }

    /**
     * Read the bundled catalog.
     *
     * A malformed or missing asset yields an empty list rather than a crash: the free-text "Pull by
     * id" tab is always available, so a broken catalog degrades the first screen instead of removing
     * the app's only entry point.
     */
    fun load(context: Context): List<Entry> =
        runCatching {
            context.assets.open(ASSET).bufferedReader(Charsets.UTF_8).use { reader ->
                Gson().fromJson(reader, Wire::class.java)?.models.orEmpty()
            }
        }.getOrDefault(emptyList())
            .filter { it.repoId.isNotBlank() }
}
