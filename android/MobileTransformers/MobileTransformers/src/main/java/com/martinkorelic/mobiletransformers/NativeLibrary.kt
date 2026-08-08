package com.martinkorelic.mobiletransformers

/**
 * Single owner of `System.loadLibrary("mobiletransformers")`.
 *
 * Every class declaring `external fun` must touch this before its first native call. Previously only
 * [ORTGeneratorGenAI] and [GenAISpike] loaded the library, so the whole Native path — tokenizer,
 * generator, trainer, retriever — worked *only* if a GenAI class happened to be constructed first. The
 * sample app got away with it; `MobileTransformers.fromPretrained` did not, and failed with
 * `UnsatisfiedLinkError: No implementation found for ... createTokenizerSession` on a real device. JVM
 * tests never touch JNI, so nothing host-side could catch it.
 *
 * Loading happens in the object's static initializer, so the JVM guarantees it runs exactly once and
 * that concurrent callers block until it completes.
 */
internal object NativeLibrary {
    init {
        System.loadLibrary("mobiletransformers")
    }

    /** Touch the object to force its static initializer. */
    fun ensureLoaded() = Unit
}
