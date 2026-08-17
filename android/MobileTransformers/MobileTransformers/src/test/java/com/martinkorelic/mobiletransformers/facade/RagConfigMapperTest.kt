package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.NotImplementedFeatureException
import com.martinkorelic.mobiletransformers.ORTRagConfig
import com.martinkorelic.mobiletransformers.config.RagConfig
import com.martinkorelic.mobiletransformers.constants.IndexingMode
import com.martinkorelic.mobiletransformers.constants.SearchType
import com.martinkorelic.mobiletransformers.internal.config.toOrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

/** #27: public RagConfig → internal ORTRagConfig mapping, defaults, read-only metric, F7 fail-closed. */
class RagConfigMapperTest {

    @Test
    fun everyFieldMapsToOrtTarget() {
        val ort = RagConfig(
            topK = 5,
            searchType = SearchType.TEXT,
            embeddingDimension = 384,
            minScore = 0.25,
            embeddingRepoId = "org__model",
            embeddingModelFile = "embed.onnx",
            chunkSize = 256,
            chunkOverlap = 32,
            maxTextLength = 2048,
        ).toOrt()
        assertEquals("org__model", ort.repoName)
        assertEquals("embed.onnx", ort.onnxName)
        assertEquals(384, ort.embeddingDimension)
        assertEquals(5, ort.topK)
        assertEquals("text", ort.searchType)
        assertEquals(0.25, ort.minScore, 0.0)
        assertEquals("precompute", ort.indexingMode)
        assertEquals(256, ort.chunkSize)
        assertEquals(32, ort.chunkOverlap)
        assertEquals(2048, ort.maxTextLength)
    }

    @Test
    fun defaults() {
        val ort = RagConfig().toOrt()
        assertEquals("semantic", ort.searchType)
        assertEquals(0.0, ort.minScore, 0.0)
        assertEquals("precompute", ort.indexingMode)
    }

    /**
     * The encoder the package shipped survives a default-constructed public config. Before this, a
     * `RagConfig()` overwrote `repoName`/`onnxName`/`embeddingDimension` with library defaults, so a
     * real package's retriever looked under `<cacheDir>/model/embedding/` for a 256-wide store.
     */
    @Test
    fun packageEncoderIdentitySurvivesDefaultConfig() {
        val fromPackage = ORTRagConfig(
            repoName = "HuggingFaceTB__SmolLM2-135M-Instruct",
            onnxName = "embedding_model",
            embeddingDimension = 384,
        )
        val ort = RagConfig(topK = 3).toOrt(fromPackage)
        assertEquals("HuggingFaceTB__SmolLM2-135M-Instruct", ort.repoName)
        assertEquals("embedding_model", ort.onnxName)
        assertEquals(384, ort.embeddingDimension)
        // Query shaping still comes from the caller.
        assertEquals(3, ort.topK)
    }

    @Test
    fun explicitEncoderIdentityOverridesThePackage() {
        val fromPackage = ORTRagConfig(repoName = "pkg", onnxName = "a", embeddingDimension = 384)
        val ort = RagConfig(
            embeddingRepoId = "other",
            embeddingModelFile = "b",
            embeddingDimension = 768,
        ).toOrt(fromPackage)
        assertEquals("other", ort.repoName)
        assertEquals("b", ort.onnxName)
        assertEquals(768, ort.embeddingDimension)
    }

    @Test
    fun similarityMetricIsReadOnlyCosine() {
        assertEquals("COSINE", RagConfig().similarityMetric)
    }

    @Test
    fun dynamicIndexingModeFailsClosed() {
        val ex = assertThrows(NotImplementedFeatureException::class.java) {
            RagConfig(indexingMode = IndexingMode.DYNAMIC).toOrt()
        }
        assertEquals(true, ex.message!!.contains("dynamic"))
    }
}
