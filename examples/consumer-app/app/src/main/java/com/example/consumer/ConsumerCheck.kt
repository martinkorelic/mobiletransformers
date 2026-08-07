package com.example.consumer

import android.content.Context
import com.martinkorelic.mobiletransformers.MobileTransformerModel
import com.martinkorelic.mobiletransformers.MobileTransformers
import com.martinkorelic.mobiletransformers.config.GenerationConfig
import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine

/**
 * Compile-time proof that the published AAR exposes a usable public surface to an outside module.
 *
 * If any symbol here stops resolving, something public became `internal` (or moved) — a SemVer break
 * that no in-repo test would catch, because in-repo callers can see `internal` declarations.
 *
 * Nothing here runs: loading a model needs a package on a device.
 */
object ConsumerCheck {

    suspend fun load(context: Context, repoId: String): MobileTransformerModel =
        MobileTransformers.fromPretrained(
            context = context,
            repoId = repoId,
            features = setOf(ModelFeature.Inference),
            engine = InferenceEngine.NATIVE,
        )

    suspend fun generate(model: MobileTransformerModel, prompt: String): String =
        model.generate(prompt, GenerationConfig(maxNewTokens = 32)).text

    fun describe(model: MobileTransformerModel): String =
        "engine=${model.engine} features=${model.installedFeatures} repo=${model.repoId}"
}
