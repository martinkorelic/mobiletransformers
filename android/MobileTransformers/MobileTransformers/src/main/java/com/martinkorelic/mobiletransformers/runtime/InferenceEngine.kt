package com.martinkorelic.mobiletransformers.runtime

/**
 * The single inference-engine selector over one shared package: Native (guaranteed default) vs the
 * opt-in ONNX Runtime GenAI engine (#11). This is the canonical declaration; #17/#19/#24 reuse it verbatim.
 * The engine is a *selection over one `inference/` package*, never a separate package or build. GenAI runs
 * on a genai-paired stock ORT shipped as `libort_gen.so` (see spikes/genai_external_swap/README.md — ORT
 * separation); Native runs on the source-built ORT-training `libonnxruntime.so`.
 */
enum class InferenceEngine {
    /** Always available; consumes the shared `inference/` package via the native ORT runtime. */
    NATIVE,

    /** Opt-in; consumes the SAME package via the GenAI engine. Availability is gated by #11 (Gate 0.1). */
    GENAI,
}
