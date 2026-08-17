package com.martinkorelic.mobiletransformers.packages

import java.io.File
import java.security.MessageDigest

/** SHA-256 integrity verification of package files against a `{relativePath: sha256hex}` map (#13). */
object ChecksumVerifier {
    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    /** True iff every entry in [checksums] exists under [baseDir] and hashes to the expected digest. */
    fun verify(baseDir: File, checksums: Map<String, String>): Boolean {
        for ((rel, expected) in checksums) {
            val f = File(baseDir, rel)
            if (!f.isFile || sha256(f) != expected) return false
        }
        return true
    }

    /** Verify the [requiredFiles] subset of a manifest's `sha256` map; returns the first bad path or null. */
    fun firstMismatch(baseDir: File, manifest: MobileTransformersManifest): String? {
        for (rel in manifest.requiredFiles) {
            val expected = manifest.sha256[rel] ?: continue
            val f = File(baseDir, rel)
            if (!f.isFile || sha256(f) != expected) return rel
        }
        return null
    }
}
