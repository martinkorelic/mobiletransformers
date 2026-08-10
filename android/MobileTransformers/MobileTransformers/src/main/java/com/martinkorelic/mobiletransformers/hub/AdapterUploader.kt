package com.martinkorelic.mobiletransformers.hub

import com.google.gson.Gson
import com.martinkorelic.mobiletransformers.BuildConfig
import com.martinkorelic.mobiletransformers.MissingArtifactException
import com.martinkorelic.mobiletransformers.MobileTransformersException
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap
import java.io.File
import com.martinkorelic.mobiletransformers.packages.PackagePaths

/**
 * #22: on-device adapter push-back — the Kotlin mirror of the Python `adapter/{export,convert,model_card}`
 * flow. **Disabled by default** ([BuildConfig.ADAPTER_UPLOAD_ENABLED] = false, privacy-gated); the product
 * path is device → desktop sync → Python `push-adapter`. The metadata read, Mode-1/Mode-2 gate, and card
 * (with the mandatory bold privacy warning + base-model license) are pure and JVM-tested; the actual
 * authenticated Hub upload + ORT `CheckpointState` factor read are the device legs.
 */

/** Adapter metadata recovered from the on-device cache (`train/training_config.json` + handoff map). */
data class AdapterMetadata(
    val peftMethod: String,
    val rank: Int?,
    val alpha: Int?,
    val peftTarget: List<String>,
    val trainableParameterCount: Int?,
    val tensorNames: List<String>,
)

/** Emitted layout mode (mirrors Python `convert.to_peft_layout`). */
enum class AdapterMode { PEFT, NATIVE }

private data class TrainingConfigJson(
    val peftMethod: String? = null,
    val peft_method: String? = null,
    val rank: Int? = null,
    val alpha: Int? = null,
    val peft_target: List<String>? = null,
    val trainable_parameter_count: Int? = null,
)

object AdapterPackageBuilder {
    private val gson = Gson()

    /** Read `train/training_config.json` + `train/weight_handoff_map.json` from the installed package. */
    fun build(cacheDir: File, repoId: String): AdapterMetadata {
        val sanitized = PackageFormat.sanitizeRepoId(repoId)
        val trainDir = PackagePaths.forCache(cacheDir, sanitized).train
        val cfgFile = File(trainDir, "training_config.json")
        if (!cfgFile.isFile) {
            throw MissingArtifactException("adapter export: $cfgFile not found (train the model first)")
        }
        val cfg = gson.fromJson(cfgFile.readText(), TrainingConfigJson::class.java) ?: TrainingConfigJson()
        val handoffFile = File(trainDir, WeightHandoffMap.FILENAME)
        val tensorNames =
            if (handoffFile.isFile) {
                WeightHandoffMap.load(handoffFile).entries.flatMap { it.inferenceInitializerNames.values }
            } else {
                emptyList()
            }
        return AdapterMetadata(
            peftMethod = (cfg.peftMethod ?: cfg.peft_method ?: "lora").lowercase(),
            rank = cfg.rank,
            alpha = cfg.alpha,
            peftTarget = cfg.peft_target ?: emptyList(),
            trainableParameterCount = cfg.trainable_parameter_count,
            tensorNames = tensorNames,
        )
    }
}

object AdapterModeGate {
    /**
     * Mirror of `convert.to_peft_layout`: a drop-in PEFT layout (Mode 1) is emitted only for a clean
     * LoRA adapter; MARS and factor-less LoRA fall to the MobileTransformers-native layout (Mode 2).
     */
    fun decide(meta: AdapterMetadata): AdapterMode =
        if (meta.peftMethod == "lora" && meta.rank != null && meta.alpha != null) AdapterMode.PEFT else AdapterMode.NATIVE
}

object AdapterCard {
    const val PRIVACY_WARNING =
        "**⚠️ Privacy warning:** this adapter was fine-tuned on-device and its weights may encode private " +
            "information from your training data. Review before sharing."

    fun render(meta: AdapterMetadata, mode: AdapterMode, baseModelLicense: String = "see upstream"): String =
        buildString {
            appendLine(PRIVACY_WARNING)
            appendLine()
            appendLine("## Adapter")
            appendLine("- PEFT method: ${meta.peftMethod}")
            appendLine("- Mode: $mode")
            meta.rank?.let { appendLine("- rank: $it") }
            meta.alpha?.let { appendLine("- alpha: $it") }
            if (meta.peftTarget.isNotEmpty()) appendLine("- target modules: ${meta.peftTarget.joinToString(", ")}")
            appendLine()
            appendLine("## Licenses")
            appendLine("- Base model weights: $baseModelLicense")
            appendLine()
            appendLine("## Re-apply")
            appendLine("Load onto the same base model this adapter was trained from.")
        }

    /** Fail closed if a mandatory disclosure is missing (mirror of Python `assert_required_sections`). */
    fun assertRequiredSections(card: String) {
        val missing = buildList {
            if (!card.contains("Privacy warning")) add("privacy warning")
            if (!card.contains("## Licenses")) add("licenses section")
        }
        if (missing.isNotEmpty()) {
            throw MissingArtifactException("adapter card missing required section(s): ${missing.joinToString(", ")}")
        }
    }
}

/** Raised when device upload is attempted while disabled by default (the privacy gate). */
class AdapterUploadDisabledException :
    MobileTransformersException(
        "On-device adapter upload is disabled (BuildConfig.ADAPTER_UPLOAD_ENABLED=false). Sync the " +
            "package to a desktop and run `mobiletransformers push-adapter` instead.",
    )

object AdapterUploader {
    /** True only when the build flag is set (default false). Kept as a function for test overridability. */
    fun uploadEnabled(): Boolean = BuildConfig.ADAPTER_UPLOAD_ENABLED

    /**
     * Prepare (always) and — only when [uploadEnabled] — upload the adapter. Preparation is pure
     * (metadata → gate → card + fail-closed section check); the authenticated Hub POST is the device leg.
     */
    fun prepareCard(cacheDir: File, repoId: String, baseModelLicense: String = "see upstream"): String {
        val meta = AdapterPackageBuilder.build(cacheDir, repoId)
        val mode = AdapterModeGate.decide(meta)
        val card = AdapterCard.render(meta, mode, baseModelLicense)
        AdapterCard.assertRequiredSections(card)
        return card
    }
}
