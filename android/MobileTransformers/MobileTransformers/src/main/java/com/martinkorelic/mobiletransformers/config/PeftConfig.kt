package com.martinkorelic.mobiletransformers.config

/**
 * PEFT selection surface (#19), a sealed class over the Python export taxonomy.
 *
 * **On-device semantics:** PEFT topology is baked in at export time (`trainer/builder.py`
 * `train_method` + `peft_models/mars/config.py` `MarsConfig.optimization_level`), so a package's
 * `train/training_config.json` already fixes the method. `MobileTransformerModel.applyPeft` is therefore
 * a *selection/validation* step (against what the installed package supports) plus rank/alpha overrides —
 * never a graph rewrite. The mapping to the Python taxonomy lives in
 * `internal/config/PeftSupport.kt`.
 */
sealed class PeftConfig {
    abstract val rank: Int
    abstract val alpha: Int
    open val targetModules: List<String>? = null

    /** LoRA (`train_method = "lora"`). */
    data class Lora(
        override val rank: Int = 16,
        override val alpha: Int = 32,
        override val targetModules: List<String>? = null,
    ) : PeftConfig()

    /** MARS optimization level 0 — fully trainable, no quantization (`train_method = "mars"`). */
    data class MarsOpt0(
        override val rank: Int = 8,
        override val alpha: Int = 8,
        override val targetModules: List<String>? = null,
    ) : PeftConfig()

    /** MARS optimization level 1 — partial trainable (frozen + fused down-proj), no quantization. */
    data class MarsOpt1(
        override val rank: Int = 8,
        override val alpha: Int = 8,
        override val targetModules: List<String>? = null,
    ) : PeftConfig()

    /** MARS quantized — optimization levels 2/3/4 with 8- or 4-bit weights. */
    data class MarsQuantized(
        override val rank: Int = 8,
        override val alpha: Int = 8,
        val optimizationLevel: Int = 4, // 2, 3, or 4 (MarsConfig.optimization_level)
        val quantNBits: Int = 8, // 8 or 4 (MarsConfig.quant_n_bits)
        override val targetModules: List<String>? = null,
    ) : PeftConfig()
}
