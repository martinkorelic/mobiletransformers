package com.martinkorelic.mobiletransformers.internal.config

import com.google.gson.JsonParser
import com.martinkorelic.mobiletransformers.PeftMismatchException
import com.martinkorelic.mobiletransformers.config.PeftConfig

/**
 * Pure PEFT taxonomy mapping + validation (#19). On-device `applyPeft` reads the exported method from the
 * package's `train/training_config.json` and validates the requested [PeftConfig] against it. This mirrors
 * the Python export taxonomy (`export/training_export.py` `train_method`, via `config/registry/peft.py`,
 * + `MarsConfig.optimization_level`); keep
 * the two in sync. Everything here is pure (Gson, no Android framework) so it is JVM-unit-testable.
 */
data class PeftTaxonomy(val trainMethod: String, val optimizationLevel: Int?)

object PeftSupport {

    /** Map a requested [PeftConfig] onto the Python `(train_method, optimization_level)` taxonomy. */
    fun taxonomy(peft: PeftConfig): PeftTaxonomy =
        when (peft) {
            is PeftConfig.Lora -> PeftTaxonomy("lora", null)
            is PeftConfig.MarsOpt0 -> PeftTaxonomy("mars", 0)
            is PeftConfig.MarsOpt1 -> PeftTaxonomy("mars", 1)
            is PeftConfig.MarsQuantized -> PeftTaxonomy("mars", peft.optimizationLevel)
        }

    /**
     * Parse `(train_method, optimization_level)` from a `training_config.json` string (tolerating a
     * `train_config` wrapper). Returns null if the package declares no method — i.e. it cannot be
     * validated and [validate] accepts.
     */
    fun packageTaxonomy(trainingConfigJson: String): PeftTaxonomy? {
        val root = JsonParser.parseString(trainingConfigJson).asJsonObject
        val cfg = if (root.has("train_config")) root.getAsJsonObject("train_config") else root
        val method = if (cfg.has("train_method")) cfg.get("train_method").asString else return null
        val opt = if (cfg.has("optimization_level")) cfg.get("optimization_level").asInt else null
        return PeftTaxonomy(method, opt)
    }

    /** Fail closed with [PeftMismatchException] if [requested] doesn't match the package's export [pkg]. */
    fun validate(requested: PeftConfig, pkg: PeftTaxonomy?) {
        pkg ?: return // package declares no method — nothing to validate against
        val want = taxonomy(requested)
        val matches = want.trainMethod == pkg.trainMethod &&
            (want.optimizationLevel == null || pkg.optimizationLevel == null ||
                want.optimizationLevel == pkg.optimizationLevel)
        if (!matches) {
            throw PeftMismatchException(requested = describe(want), supported = listOf(describe(pkg)))
        }
    }

    private fun describe(t: PeftTaxonomy): String =
        t.trainMethod + (t.optimizationLevel?.let { " (optimization_level=$it)" } ?: "")
}
