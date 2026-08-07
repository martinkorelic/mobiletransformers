package com.martinkorelic.mobiletransformers.runtime

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #23 regression guards (source greps): the retired `inference/merged/` handoff must not reappear in the
 * load path, and the dead GenAI stubs (`ORTGenAINative.kt` / `onnx-genai.cpp`, deleted by #11) must stay
 * deleted rather than be resurrected half-implemented.
 */
class NativeLoadRegressionTest {

    private fun moduleRoot(): File {
        var dir: File? = File("").absoluteFile
        while (dir != null) {
            if (File(dir, "src/main/cpp/session_cache.h").isFile) return dir
            val nested = File(dir, "android/MobileTransformersApp/MobileTransformers")
            if (File(nested, "src/main/cpp/session_cache.h").isFile) return nested
            dir = dir.parentFile
        }
        error("could not locate the MobileTransformers module root from ${File("").absolutePath}")
    }

    private fun read(rel: String): String = File(moduleRoot(), rel).readText()

    @Test
    fun noMergedSubdirLiteralInNativeLoadPath() {
        val kotlin = read("src/main/java/com/martinkorelic/mobiletransformers/ORTGeneratorNative.kt")
        assertFalse("ORTGeneratorNative still builds an inference/merged path", kotlin.contains("\"/merged\""))
        assertFalse("ORTGeneratorNative still probes a 'merged' subdir", kotlin.contains(", \"merged\")"))

        val cache = read("src/main/cpp/session_cache.h")
        assertFalse("session_cache still reads from /merged", cache.contains("\"/merged\""))
        assertFalse("session_cache still reads from inference/merged", cache.contains("inference_model_path + \"/merged\""))
    }

    @Test
    fun handoffLoadIsMapDriven() {
        val cache = read("src/main/cpp/session_cache.h")
        assertTrue("session_cache must load via the shared handoff reader", cache.contains("load_handoff_entries"))
    }

    @Test
    fun deadGenAiStubsStayDeleted() {
        val root = moduleRoot()
        assertFalse(
            "ORTGenAINative.kt was resurrected",
            File(root, "src/main/java/com/martinkorelic/mobiletransformers/ORTGenAINative.kt").exists(),
        )
        assertFalse("onnx-genai.cpp was resurrected", File(root, "src/main/cpp/onnx-genai.cpp").exists())
    }
}
