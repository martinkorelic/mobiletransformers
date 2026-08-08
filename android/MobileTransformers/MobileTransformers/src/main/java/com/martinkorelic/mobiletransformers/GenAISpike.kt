package com.martinkorelic.mobiletransformers

/**
 * JNI binding for the GenAI external-data-swap spike (#10, Gate 0.1). Backed by `cpp/genai_spike.cpp`.
 *
 * [runOneToken] loads `<dir>` with the stable `OgaCreateModel`, generates one greedy token, and returns
 * `"token=<id>;fp=<logits fingerprint>;rssPre=..;rssLoaded=..;rssTok=.."`. The device test drives it before
 * and after overwriting one external weight `.bin` and asserts the fingerprint changes — proving GenAI reads
 * the package's external data at construction (F2), with no graph rewrite and no fork.
 */
object GenAISpike {
    init {
        NativeLibrary.ensureLoaded()
    }

    external fun runOneToken(dir: String, prompt: String): String

    /** Parse the `key=value;...` metrics string returned by [runOneToken]. */
    fun parse(result: String): Map<String, String> =
        result.split(";").mapNotNull {
            val kv = it.split("=", limit = 2)
            if (kv.size == 2) kv[0] to kv[1] else null
        }.toMap()
}
