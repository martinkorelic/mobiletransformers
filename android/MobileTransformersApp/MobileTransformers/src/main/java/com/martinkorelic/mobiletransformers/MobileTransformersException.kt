package com.martinkorelic.mobiletransformers

/**
 * Public exception hierarchy for the SDK facade.
 *
 * DECOMPOSE(#19): `02_code_plans/01` (#19) owns the full hierarchy (adds `PeftMismatchException`,
 * `FeatureNotInstalledException`, `EngineUnavailableException`, `NotImplementedFeatureException`). This phase
 * stubs the base + the two construction-time exceptions #17 raises. The Python `exceptions.py` mirrors these
 * names. Do not fall back silently — every fail-closed path raises through this hierarchy.
 */
open class MobileTransformersException(message: String, cause: Throwable? = null) :
    Exception(message, cause)

/** The requested model/package is not installed in the cache. */
class ModelNotInstalledException(message: String) : MobileTransformersException(message)

/** A required artifact (manifest, weight handoff, `inference/` group, …) is missing from the package. */
class MissingArtifactException(message: String) : MobileTransformersException(message)
