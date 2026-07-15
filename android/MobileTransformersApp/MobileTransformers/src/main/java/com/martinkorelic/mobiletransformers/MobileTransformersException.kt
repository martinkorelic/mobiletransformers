package com.martinkorelic.mobiletransformers

import com.martinkorelic.mobiletransformers.packages.ModelFeature
import com.martinkorelic.mobiletransformers.runtime.InferenceEngine

/**
 * Public exception hierarchy for the SDK facade (#19 owns the canonical set).
 *
 * Every fail-closed path in the facade raises through this sealed hierarchy — never a bare `Exception`,
 * never a silent fallback. Friendly messages name the exact missing artifact/path so a caller can act.
 *
 * The Python `exceptions.py` mirrors this hierarchy at the ROOT + INTENT level, not 1:1 by subclass name:
 * the Python set is export/hub-shaped (`ConfigValidationError`, `ExportError`, `ManifestError`,
 * `NoCompatibleVariant`, `HandoffError`, `MergeError`, `UnsupportedModelError`, `HubError`) while this
 * Kotlin set is device/facade-shaped. Both roots mean "anything the library raises"; do not force a false
 * 1:1 rename across the language boundary.
 */
// `open` (not `sealed`): subclasses live across packages now (e.g. `hub.AdapterUploadDisabledException`),
// which a sealed base — constrained to one package — would forbid.
open class MobileTransformersException(message: String, cause: Throwable? = null) :
    Exception(message, cause)

/** The requested model/package is not installed in the cache (remote pull is #21). */
class ModelNotInstalledException(message: String) : MobileTransformersException(message) {
    constructor(repoId: String, cacheDir: String) : this(
        "Model '$repoId' is not installed at $cacheDir. Pull it first (fromPretrained downloads are #21).",
    )
}

/** A required artifact (manifest, weight handoff, an `inference/`/`train/` config, …) is missing. */
class MissingArtifactException(message: String) : MobileTransformersException(message) {
    constructor(feature: ModelFeature, expectedPath: String) : this(
        "$feature is not available: expected '$expectedPath' was not found in the installed package.",
    )
}

/** The requested PEFT method/parameters do not match what the installed package was exported with. */
class PeftMismatchException(requested: String, supported: List<String>) :
    MobileTransformersException(
        "Requested PEFT '$requested' is not supported by this package. Exported with: " +
            "${supported.joinToString(", ").ifEmpty { "<none declared>" }}. On-device training can only " +
            "re-run within the exported PEFT topology, not change it.",
    )

/** A feature was requested that the installed package does not carry. */
class FeatureNotInstalledException(feature: ModelFeature, installed: Set<ModelFeature>) :
    MobileTransformersException(
        "Feature '$feature' is not installed for this package. Installed features: " +
            "${installed.joinToString(", ").ifEmpty { "<none>" }}.",
    )

/** The requested inference engine cannot run against this package (e.g. GenAI without a genai config). */
class EngineUnavailableException(engine: InferenceEngine, reason: String) :
    MobileTransformersException("Engine '$engine' is unavailable: $reason")

/** A public API surface exists but its behavior is not implemented in this tier (e.g. pushAdapter). */
class NotImplementedFeatureException(name: String) :
    MobileTransformersException("'$name' is not implemented in this version.")
