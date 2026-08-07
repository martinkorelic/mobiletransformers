package com.martinkorelic.mobiletransformers.internal.runtime

import com.martinkorelic.mobiletransformers.MissingArtifactException
import com.martinkorelic.mobiletransformers.packages.ChecksumVerifier
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap
import java.io.File

/**
 * #23: fail-closed precondition for loading merged trained weights as flat per-tensor external
 * initializers, replacing the retired `inference/merged/` directory probe.
 *
 * Mirrors the on-disk contract #9 writes into `<cacheDir>/<repo>/inference/`:
 *  - `weight_handoff_map.json` (schema-gated via [PackageFormat.checkCompat]),
 *  - one flat `<name>.bin` per `externalDataLocation[role]`,
 *  - a sibling `<name>.bin.sha256` (hex + newline) written atomically by the merger
 *    (`weight_merger.cpp::write_raw_tensor_atomic`), and/or a per-role `sha256` in the map itself.
 *
 * **Checksum precedence: sidecar over map.** The sidecar is the *live* digest (rewritten by the device
 * merger); the map's `sha256` is the *shipped* digest (stamped by the exporter over the pre-merge base
 * bytes, never refreshed on device). See `docs/MODEL_FORMAT.md`.
 *
 * This is the PRIMARY gate: it runs BEFORE the native session is created. A missing map means there is
 * nothing merged to load (the base graph is used — not an error). A map that is present but broken
 * (missing `.bin`, checksum mismatch, absent checksum) throws [MissingArtifactException] naming the
 * offending tensor rather than silently downgrading to base weights. The C++ loader
 * (`session_cache.h`) re-derives initializer names from the map and additionally validates dtype/shape
 * against each loaded `TensorProto` (which requires parsing the protobuf, hence C++-side).
 */
object HandoffPrecondition {

    /**
     * True iff [inferenceDir] carries a valid, fully-materialized merged-weight set.
     *
     * @param verifyChecksums when true (the load gate), every `.bin` is hashed and matched against its
     *   declared checksum; when false (the cheap capability query), only presence + schema are checked.
     * @throws MissingArtifactException if the map is present but any file/checksum check fails.
     */
    fun loadMergedWeightsReady(inferenceDir: File, verifyChecksums: Boolean = true): Boolean {
        val mapFile = File(inferenceDir, WeightHandoffMap.FILENAME)
        if (!mapFile.isFile) return false

        val map = WeightHandoffMap.load(mapFile)
        if (!PackageFormat.checkCompat(map.schemaVersion, map.minReaderVersion, WeightHandoffMap.READER_VERSION)) {
            throw MissingArtifactException(
                "weight_handoff_map.json schema ${map.schemaVersion} (minReader ${map.minReaderVersion}) " +
                    "incompatible with reader ${WeightHandoffMap.READER_VERSION}",
            )
        }
        if (map.entries.isEmpty()) return false

        for (entry in map.entries) {
            val where = entry.trainingBaseLayerName.ifEmpty { "<unnamed handoff entry>" }
            for ((role, binName) in entry.externalDataLocation) {
                val bin = File(inferenceDir, binName)
                if (!bin.isFile) {
                    throw MissingArtifactException(
                        "$where: merged weight file '$binName' (role '$role') missing from inference/",
                    )
                }
                if (!verifyChecksums) continue

                val actual = ChecksumVerifier.sha256(bin)
                // #9/#23 precedence: the SIDECAR wins. `<name>.bin.sha256` is the LIVE digest —
                // `weight_merger.cpp::write_raw_tensor_atomic` rewrites it on every on-device merge.
                // `entry.sha256[role]` is the SHIPPED digest, stamped once by the exporter over the
                // pre-merge base bytes and never updated by the device (C++ only reads the map).
                // Preferring the map made a *correct* merge throw a checksum mismatch on the next load.
                val expected = readSidecar(File(inferenceDir, "$binName.sha256"))
                    ?: entry.sha256[role]?.takeIf { it.isNotBlank() }
                    ?: throw MissingArtifactException(
                        "$where: no checksum for role '$role' (neither '$binName.sha256' nor map sha256)",
                    )
                if (!actual.equals(expected, ignoreCase = true)) {
                    throw MissingArtifactException(
                        "$where: checksum mismatch for '$binName' (role '$role'): expected $expected, got $actual",
                    )
                }
            }
        }
        return true
    }

    /** Non-throwing capability query: presence + schema + file existence only (no hashing). */
    fun mergedWeightsPresent(inferenceDir: File): Boolean =
        try {
            loadMergedWeightsReady(inferenceDir, verifyChecksums = false)
        } catch (_: Exception) {
            false
        }

    private fun readSidecar(f: File): String? =
        if (f.isFile) f.readText().trim().substringBefore('\n').trim().ifBlank { null } else null
}
