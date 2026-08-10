package com.martinkorelic.mobiletransformers.packages

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Kotlin half of the path-resolver parity. Mirrors `tests/unit/test_package_paths.py` case for case —
 * the same package is read by Python, Kotlin and C++, so a divergence here is a real defect.
 */
class PackagePathsTest {

    private fun hubVariant(
        paths: Map<String, String> = mapOf(
            "inference" to "variants/cpu-int4/inference",
            "train" to "variants/cpu-int4/train",
            "embedding" to "variants/cpu-int4/embedding",
            "tokenizer" to "shared/tokenizer",
        ),
    ) = MobileTransformersManifest.Variant(id = "cpu-int4", paths = paths)

    @Test
    fun hubLayoutUsesTheManifestsDeclaredPaths() {
        val paths = PackagePaths.forHub(File("/pkg"), hubVariant())

        assertEquals(File("/pkg/variants/cpu-int4/train"), paths.train)
        assertEquals(File("/pkg/variants/cpu-int4/inference"), paths.inference)
        // Shared across variants — NOT under variants/<id>/. Re-deriving the convention gets this wrong.
        assertEquals(File("/pkg/shared/tokenizer"), paths.tokenizer)
    }

    @Test
    fun hubLayoutHonoursAVariantThatPlacesAStageUnusually() {
        val odd = hubVariant(
            mapOf(
                "inference" to "variants/cpu-int4/inference",
                "train" to "somewhere/else/train",
                "tokenizer" to "shared/tokenizer",
            ),
        )

        assertEquals(File("/pkg/somewhere/else/train"), PackagePaths.forHub(File("/pkg"), odd).train)
    }

    @Test
    fun cacheLayoutIsFlatAndDeclaresEveryStage() {
        val paths = PackagePaths.forCache("/cache", "org__model")

        assertEquals(File("/cache/org__model/train"), paths.train)
        assertEquals(File("/cache/org__model/inference"), paths.inference)
        assertEquals(File("/cache/org__model/embedding"), paths.embedding)
        // Flat: a sibling, not shared/. This is the difference the #35 client tripped over.
        assertEquals(File("/cache/org__model/tokenizer"), paths.tokenizer)
        assertTrue(PackagePaths.STAGES.all { paths.has(it) })
    }

    @Test
    fun theTwoLayoutsDisagreeWhichIsTheWholePoint() {
        val hub = PackagePaths.forHub(File("/pkg"), hubVariant())
        val cache = PackagePaths.forCache("/pkg", "model")

        assertNotEquals(hub.train, cache.train)
    }

    @Test
    fun weightHandoffSitsInsideInferenceInBothLayouts() {
        assertEquals(
            File("/pkg/variants/cpu-int4/inference/weight_handoff_map.json"),
            PackagePaths.forHub(File("/pkg"), hubVariant()).weightHandoff,
        )
        assertEquals(
            File("/cache/model/inference/weight_handoff_map.json"),
            PackagePaths.forCache("/cache", "model").weightHandoff,
        )
    }

    @Test
    fun anUndeclaredStageFailsClosedNamingWhatExists() {
        val paths = PackagePaths.forHub(
            File("/pkg"),
            hubVariant(mapOf("inference" to "variants/v/inference")),
        )

        assertFalse(paths.has("train"))
        val error = assertThrows(IllegalArgumentException::class.java) { paths.train }
        assertTrue(error.message!!.contains("train"))
        assertTrue(error.message!!.contains("inference"))
    }

    @Test
    fun anUnknownStageNameIsRejectedRatherThanSilentlyMissing() {
        val paths = PackagePaths.forCache("/cache", "model")

        val error = assertThrows(IllegalArgumentException::class.java) { paths.stage("trian") }
        assertTrue(error.message!!.contains("unknown stage"))
    }

    @Test
    fun aVariantWithoutPathsFailsClosedTellingYouToReExport() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            PackagePaths.forHub(File("/pkg"), hubVariant(emptyMap()))
        }
        assertTrue(error.message!!.contains("re-export"))
    }
}
