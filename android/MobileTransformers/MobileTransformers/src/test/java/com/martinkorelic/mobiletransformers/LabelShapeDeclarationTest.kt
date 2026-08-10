package com.martinkorelic.mobiletransformers

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * #33: how a preprocessor DECLARES its label shape.
 *
 * Robolectric because these parse real `org.json.JSONObject`s, which the plain unit-test classpath
 * stubs — and `isReturnDefaultValues = false` (build.gradle.kts) makes that stub throw rather than
 * silently return 0/null, so a test could otherwise "pass" against a method that never ran.
 */
@RunWith(RobolectricTestRunner::class)
class LabelShapeDeclarationTest {

    @Test
    fun aTokenLevelPreprocessorDeclaresNoClassLabelSoNothingElseChanges() {
        // The default keeps every existing preprocessor — including users' own `customPreprocess`,
        // which is public API — on the token-level path without any change.
        val json = JSONObject("""{"sentence": "The cat sat.", "label": 1}""")

        assertNull(CoLAPreprocessor.classLabel(json))
        assertEquals(1, CoLAClassificationPreprocessor.classLabel(json))
    }

    @Test
    fun theClassificationPreprocessorKeepsTheLabelAnIndexRatherThanStringifyingIt() {
        val json = JSONObject("""{"sentence": "The cat sat.", "label": 0}""")

        // CoLAPreprocessor works around having only one label shape by turning the class into words.
        assertEquals("unacceptable", CoLAPreprocessor.preprocess(json).second)
        // The classification objective does not need that: the index goes straight to `labels[batch]`.
        assertEquals(0, CoLAClassificationPreprocessor.classLabel(json))
        assertEquals("The cat sat.", CoLAClassificationPreprocessor.preprocess(json).first)
    }

    @Test
    fun theRegistryResolvesBothObjectivesOverTheSameDatasetFile() {
        assertTrue(getPreprocessFunctionForTask("cola") === CoLAPreprocessor)
        assertTrue(getPreprocessFunctionForTask("cola_cls") === CoLAClassificationPreprocessor)
        assertFalse(getPreprocessFunctionForTask("cola") === getPreprocessFunctionForTask("cola_cls"))
    }
}
