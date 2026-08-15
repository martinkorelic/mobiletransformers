package com.martinkorelic.mobiletransformers

import java.io.File

/**
 * Reads a file out of this module's own source tree, for the handful of guards that can only be
 * expressed over source.
 *
 * Some invariants have no runtime handle. A Kotlin `when` does not expose its branches, so "every
 * dispatch branch is also listed in the public registry" can be checked only by reading the `when`.
 * That is the same reasoning as `tests/unit/test_guards.py`, which greps the Kotlin sources from the
 * Python side — this is its in-module equivalent, so a Kotlin-only invariant does not have to be
 * enforced from another language's test suite.
 *
 * Walks up to the module root rather than assuming a working directory: Gradle's test JVM starts in
 * the module directory, but the same tests are run from the repo root by `make test-jvm`.
 */
object TestSources {

    private const val MODULE = "android/MobileTransformers/MobileTransformers"

    private fun moduleRoot(): File {
        var dir: File? = File("").absoluteFile
        while (dir != null) {
            // Either we are already inside the module, or we can see it from the repo root.
            if (File(dir, "src/main/java/com/martinkorelic/mobiletransformers").isDirectory) return dir
            val fromRepoRoot = File(dir, MODULE)
            if (File(fromRepoRoot, "src/main/java/com/martinkorelic/mobiletransformers").isDirectory) {
                return fromRepoRoot
            }
            dir = dir.parentFile
        }
        error("could not locate the SDK module from ${File("").absolutePath}")
    }

    /** @param relativePath path under `src/`, e.g. `main/java/.../DataUtil.kt`. */
    fun read(relativePath: String): String {
        val file = File(moduleRoot(), "src/$relativePath")
        check(file.isFile) { "source not found: ${file.path}" }
        return file.readText(Charsets.UTF_8)
    }
}
