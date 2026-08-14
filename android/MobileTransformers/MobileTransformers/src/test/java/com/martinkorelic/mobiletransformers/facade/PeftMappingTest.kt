package com.martinkorelic.mobiletransformers.facade

import com.martinkorelic.mobiletransformers.PeftMismatchException
import com.martinkorelic.mobiletransformers.config.PeftConfig
import com.martinkorelic.mobiletransformers.internal.config.PeftSupport
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #19: the on-device PEFT taxonomy mapping + validation (pure; mirrors the Python export taxonomy —
 * `export/training_export.py` `train_method` + `MarsConfig.optimization_level`).
 */
class PeftMappingTest {

    @Test
    fun variantsMapToPythonTaxonomy() {
        assertEquals("lora", PeftSupport.taxonomy(PeftConfig.Lora()).trainMethod)
        assertNull(PeftSupport.taxonomy(PeftConfig.Lora()).optimizationLevel)
        assertEquals("mars" to 0, PeftSupport.taxonomy(PeftConfig.MarsOpt0()).let { it.trainMethod to it.optimizationLevel })
        assertEquals("mars" to 1, PeftSupport.taxonomy(PeftConfig.MarsOpt1()).let { it.trainMethod to it.optimizationLevel })
        assertEquals(4, PeftSupport.taxonomy(PeftConfig.MarsQuantized(optimizationLevel = 4)).optimizationLevel)
    }

    @Test
    fun packageTaxonomyParsesTrainMethodAndLevel() {
        val json = """{"train_method":"mars","optimization_level":1}"""
        val pkg = PeftSupport.packageTaxonomy(json)
        assertEquals(PeftSupport.taxonomy(PeftConfig.MarsOpt1()), pkg)
    }

    @Test
    fun packageTaxonomyHonorsTrainConfigWrapper() {
        val json = """{"train_config":{"train_method":"lora"}}"""
        assertEquals("lora", PeftSupport.packageTaxonomy(json)!!.trainMethod)
    }

    @Test
    fun packageTaxonomyNullWhenNoMethodDeclared() {
        assertNull(PeftSupport.packageTaxonomy("""{"batchSize":4}"""))
    }

    @Test
    fun validateAcceptsMatchingMethod() {
        // Should not throw.
        PeftSupport.validate(PeftConfig.MarsOpt1(), PeftSupport.taxonomy(PeftConfig.MarsOpt1()))
    }

    @Test
    fun validateAcceptsWhenPackageDeclaresNoMethod() {
        PeftSupport.validate(PeftConfig.Lora(), null)
    }

    @Test
    fun validateThrowsOnMethodMismatch() {
        val ex = assertThrows(PeftMismatchException::class.java) {
            PeftSupport.validate(PeftConfig.Lora(), PeftSupport.taxonomy(PeftConfig.MarsOpt0()))
        }
        assertTrue(ex.message!!.contains("mars"))
        assertTrue(ex.message!!.contains("lora"))
    }

    @Test
    fun validateThrowsOnOptimizationLevelMismatch() {
        assertThrows(PeftMismatchException::class.java) {
            PeftSupport.validate(PeftConfig.MarsQuantized(optimizationLevel = 4), PeftSupport.taxonomy(PeftConfig.MarsOpt1()))
        }
    }
}
