package com.martinkorelic.mobiletransformers.packages

import com.google.gson.JsonParser
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests (plain JUnit, no device) for the #13 cache-bridge. Cross-language parity is pinned by
 * the SAME shared JSON oracles the Python tests use (under tests/fixtures), located by walking up to the
 * repo root, so the Kotlin and Python implementations cannot drift.
 */
class PackagesTest {
    private fun repoRoot(): File {
        var dir: File? = File("").absoluteFile
        while (dir != null) {
            if (File(dir, "tests/fixtures/sanitize_repo_id_cases.json").isFile) return dir
            dir = dir.parentFile
        }
        error("could not locate repo root (tests/fixtures not found from ${File("").absolutePath})")
    }

    private fun fixtures() = File(repoRoot(), "tests/fixtures")
    private fun tinyPackage() = File(fixtures(), "tiny_package")

    // --- cross-language parity ------------------------------------------------

    @Test
    fun sanitizeRepoIdMatchesSharedOracle() {
        val json = JsonParser.parseString(File(fixtures(), "sanitize_repo_id_cases.json").readText())
        for (case in json.asJsonObject.getAsJsonArray("cases")) {
            val obj = case.asJsonObject
            val input = obj.get("input").asString
            val expected = obj.get("expected").asString
            assertEquals("sanitize($input)", expected, PackageFormat.sanitizeRepoId(input))
        }
    }

    @Test
    fun checkCompatMatchesSharedOracle() {
        val json = JsonParser.parseString(File(fixtures(), "check_compat_cases.json").readText())
        for (case in json.asJsonObject.getAsJsonArray("cases")) {
            val o = case.asJsonObject
            val accept = o.get("expect").asString == "accept"
            val got = PackageFormat.checkCompat(
                o.get("doc").asString, o.get("minReader").asString, o.get("reader").asString,
            )
            assertEquals("checkCompat(${o.get("doc").asString}, ${o.get("minReader").asString}, ${o.get("reader").asString})", accept, got)
        }
    }

    // --- manifest + validator -------------------------------------------------

    @Test
    fun validFixtureManifestValidates() {
        val pkg = tinyPackage()
        val manifest = MobileTransformersManifest.load(File(pkg, PackageFormat.MANIFEST_FILENAME))
        ManifestValidator.validate(manifest, pkg) // no throw
        assertEquals("cpu-int4", manifest.defaultVariant)
    }

    @Test
    fun badDefaultVariantRejected() {
        val pkg = tinyPackage()
        val manifest = MobileTransformersManifest.load(File(pkg, PackageFormat.MANIFEST_FILENAME))
            .copy(defaultVariant = "ghost")
        assertThrows(ManifestException::class.java) { ManifestValidator.validate(manifest, pkg) }
    }

    // --- variant selection ----------------------------------------------------

    @Test
    fun selectPrefersSmallestMemoryThenDefault() {
        val m = MobileTransformersManifest.load(File(tinyPackage(), PackageFormat.MANIFEST_FILENAME))
        val v = VariantSelector.select(m, abis = listOf("arm64-v8a"), requestedFeatures = listOf("core", "inference"))
        assertEquals("cpu-int4", v.id)
    }

    @Test
    fun selectGenaiFiltersNativeOnly() {
        val m = MobileTransformersManifest.load(File(tinyPackage(), PackageFormat.MANIFEST_FILENAME))
        val v = VariantSelector.select(m, abis = listOf("arm64-v8a"), requestedEngine = "genai", requestedFeatures = listOf("genai"))
        assertEquals("cpu-int4", v.id)
    }

    @Test
    fun selectNoMatchThrows() {
        val m = MobileTransformersManifest.load(File(tinyPackage(), PackageFormat.MANIFEST_FILENAME))
        assertThrows(NoCompatibleVariantException::class.java) {
            VariantSelector.select(m, abis = listOf("arm64-v8a"), totalMemMb = 1024, requestedFeatures = listOf("core"))
        }
    }

    // --- checksum -------------------------------------------------------------

    @Test
    fun checksumVerifyDetectsCorruption() {
        val pkg = tinyPackage()
        val rel = "variants/cpu-int4/inference/model.onnx"
        val manifest = MobileTransformersManifest.load(File(pkg, PackageFormat.MANIFEST_FILENAME))
        val good = mapOf(rel to manifest.sha256[rel]!!)
        assertTrue(ChecksumVerifier.verify(pkg, good))
        val bad = mapOf(rel to "0".repeat(64))
        assertFalse(ChecksumVerifier.verify(pkg, bad))
    }

    // --- installer + cache index ---------------------------------------------

    /**
     * #21 crash safety: reinstalling over an existing package must never leave the cache empty.
     * The installer used to `deleteRecursively()` the live tree BEFORE renaming the new one in, so a
     * crash or a failed rename in that window destroyed the model — including local training state.
     */
    @Test
    fun reinstallOverExistingPackagePreservesTheCacheAndLeavesNoRetiredDir() {
        val cacheDir = File(createTempDir(), "cache").apply { mkdirs() }
        val first = ModelPackageInstaller.install(tinyPackage(), cacheDir, "org/Tiny-Model", "cpu-int4")
        // Something the user produced locally, which a delete-first install would destroy.
        File(first.repoDir, "train").mkdirs()
        File(first.repoDir, "train/local_marker.txt").writeText("trained")

        val second = ModelPackageInstaller.install(tinyPackage(), cacheDir, "org/Tiny-Model", "cpu-int4")
        assertEquals(first.repoDir, second.repoDir)
        assertTrue(File(second.repoDir, "inference/model.onnx").isFile)
        // The old tree is moved aside then removed — never left behind as cache litter.
        assertTrue(cacheDir.listFiles()!!.none { it.name.startsWith(".retired-") })
    }

    @Test
    fun installMaterializesCacheShapeAtomically() {
        val cacheDir = File(createTempDir(), "cache").apply { mkdirs() }
        val installed = ModelPackageInstaller.install(tinyPackage(), cacheDir, "org/Tiny-Model", "cpu-int4")
        assertEquals("org__Tiny-Model", installed.sanitizedRepoId)
        // Conventional layout LLMRepository probes:
        assertTrue(File(installed.repoDir, "inference/model.onnx").isFile)
        assertTrue(File(installed.repoDir, "train/training_config.json").isFile)
        assertTrue(File(installed.repoDir, "tokenizer/tokenizer.json").isFile)
        assertTrue(File(installed.repoDir, PackageFormat.MANIFEST_FILENAME).isFile)
        // No leftover staging dir.
        assertFalse(File(cacheDir, ".staging/org__Tiny-Model").exists())

        val index = CacheIndex.list(cacheDir)
        val entry = index.first { it.sanitizedRepoId == "org__Tiny-Model" }
        assertEquals("MobileTransformers/Tiny-0.1B", entry.baseModelId)
        assertTrue(entry.hasManifest)
        assertTrue(entry.sizeBytes > 0)
    }

    @Test
    fun cacheIndexToleratesLegacyDir() {
        val cacheDir = File(createTempDir(), "cache").apply { mkdirs() }
        File(cacheDir, "legacy-model/inference").apply { mkdirs() }
        File(cacheDir, "legacy-model/inference/model.onnx").writeText("x")
        val entry = CacheIndex.list(cacheDir).first { it.sanitizedRepoId == "legacy-model" }
        assertFalse(entry.hasManifest)
        assertNull(entry.baseModelId)
    }

    // --- supportedEngines reaches the engine selector (#13) --------------------

    @Test
    fun supportedEnginesComesFromTheNamedVariant() {
        val m = MobileTransformersManifest.load(
            File(tinyPackage(), PackageFormat.MANIFEST_FILENAME),
        )
        assertEquals(setOf("native", "genai"), m.supportedEnginesFor("cpu-int4"))
        // The native-only variant must NOT offer genai — this is the whole point: `create` was called
        // with a hard-coded setOf("native","genai") regardless of what the package declared.
        assertEquals(setOf("native"), m.supportedEnginesFor("cpu-fp16"))
    }

    @Test
    fun supportedEnginesFallsBackToTheDefaultVariant() {
        val m = MobileTransformersManifest.load(
            File(tinyPackage(), PackageFormat.MANIFEST_FILENAME),
        )
        // No variant named -> the manifest's defaultVariant (cpu-int4).
        assertEquals(setOf("native", "genai"), m.supportedEnginesFor())
    }

    @Test
    fun supportedEnginesIsNullWhenThePackageDeclaresNone() {
        // An older export, or a variant with an empty list: null means "unknown", and the caller keeps
        // its permissive default rather than silently narrowing to native-only.
        val undeclared = MobileTransformersManifest.parse(
            """{"defaultVariant":"v","variants":[{"id":"v","supportedEngines":[]}]}""",
        )
        assertNull(undeclared.supportedEnginesFor())
        assertNull(MobileTransformersManifest.parse("""{"variants":[]}""").supportedEnginesFor())
    }
}
