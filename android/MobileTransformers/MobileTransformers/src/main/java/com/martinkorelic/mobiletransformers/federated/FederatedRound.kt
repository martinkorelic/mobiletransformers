package com.martinkorelic.mobiletransformers.federated

import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Reads and writes ORT checkpoint parameters by name. Implemented over JNI on device; substitutable in
 * tests, which is the point — the round's logic (clipping, name matching, consent) is then host-testable
 * without a training session.
 */
interface CheckpointTensorStore {
    /** Raw little-endian bytes for [name], or `null` when the checkpoint has no such parameter. */
    fun read(name: String): ByteArray?

    /** Writes [data] back to [name]. Returns false on any mismatch; never truncates or pads. */
    fun write(name: String, data: ByteArray): Boolean
}

/**
 * The two directions a federated round moves adapter factors in.
 *
 * An interface for the same reason [CheckpointTensorStore] is one: the *ordering* of a round
 * (import → train → export) is a correctness property that must be pinned on the host, and the real
 * implementation refuses to run at all in a build where `FEDERATION_ENABLED` is false — which is every
 * unit-test build, deliberately. Substituting this seam tests the sequence without weakening the gate.
 */
interface AdapterExchange {
    /** Builds the record this device would upload. */
    fun exportUpdate(
        baseModelId: String,
        packageRevision: String,
        peftMethod: String,
        round: Int,
        metrics: Map<String, Double> = emptyMap(),
    ): ByteArray

    /** Writes an aggregated record into the local checkpoint; returns the number of tensors written. */
    fun importAggregate(blob: ByteArray): Int
}

/**
 * One federated round's device-side half: **export** the local adapter factors, and **import** the
 * aggregated ones.
 *
 * Deliberately does not perform the training itself or the HTTP — those are
 * `repository/TrainingRepository.performTraining` and the gateway client respectively. What lives here
 * is everything that decides *what leaves the device* and *what is allowed back in*, because that is
 * the part where a mistake is a privacy incident rather than a bug.
 *
 * Ordering, naming and dtype all come from `weight_handoff_map.json` via
 * [WeightHandoffMap.adapterTensorSpecs]; nothing here re-derives a tensor identity.
 */
class FederatedRound(
    private val config: FederatedConfig,
    private val handoff: WeightHandoffMap,
    private val store: CheckpointTensorStore,
) : AdapterExchange {

    /**
     * Builds the record this device would send, applying the configured clipping first.
     *
     * @throws FederatedConsentException before reading anything at all when a precondition fails. The
     *   gate runs first on purpose: a round that reads the user's adapters and only then discovers it
     *   lacks consent has already done the thing consent governs.
     */
    override fun exportUpdate(
        baseModelId: String,
        packageRevision: String,
        peftMethod: String,
        round: Int,
        metrics: Map<String, Double>,
    ): ByteArray {
        config.requireRoundIsPermitted()

        val record = AdapterTensorCodec.build(
            handoff = handoff,
            baseModelId = baseModelId,
            packageRevision = packageRevision,
            peftMethod = peftMethod,
            round = round,
            metrics = metrics,
        ) { spec -> store.read(spec.name)?.let { clipToNorm(it, config.clipNorm) } }

        return AdapterTensorCodec.serialize(record)
    }

    /**
     * Writes an aggregated record back into the local checkpoint, **by name**.
     *
     * @return the number of tensors written.
     * @throws FederatedRecordException if the record describes a different schema than this package,
     *   or if any tensor cannot be written. Partial application is the dangerous outcome — half the
     *   layers updated and half not is a model that is neither the local one nor the global one — so a
     *   failure is reported rather than swallowed per tensor.
     */
    override fun importAggregate(blob: ByteArray): Int {
        config.requireRoundIsPermitted()

        val record = AdapterTensorCodec.deserialize(blob)
        AdapterTensorCodec.checkFormat(record, handoff)

        // Only names this package actually declares may be written. A record naming a tensor we do not
        // have is a mismatch, not something to apply optimistically.
        val declared = handoff.adapterTensorSpecs().associateBy { it.name }
        var written = 0
        for (tensor in record.tensors) {
            val spec = declared[tensor.name]
                ?: throw FederatedRecordException(
                    "aggregated record carries '${tensor.name}', which this package does not declare"
                )
            if (tensor.shape != spec.shape) {
                throw FederatedRecordException(
                    "aggregated '${tensor.name}' has shape ${tensor.shape} but this package declares " +
                        "${spec.shape}"
                )
            }
            if (!store.write(tensor.name, tensor.payload)) {
                throw FederatedRecordException(
                    "failed to write aggregated tensor '${tensor.name}' into the local checkpoint"
                )
            }
            written++
        }
        return written
    }

    /**
     * Scales a float32 tensor so its L2 norm is at most [maxNorm], leaving it untouched when already
     * within bound.
     *
     * Clipping is what bounds any single example's influence on what leaves the device, and it is the
     * precondition for local DP noise to mean anything: without a bound on the update, there is no
     * sensitivity to calibrate noise against.
     */
    internal fun clipToNorm(bytes: ByteArray, maxNorm: Double): ByteArray {
        val buffer = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val count = bytes.size / 4
        val values = FloatArray(count) { buffer.getFloat(it * 4) }

        var sumSquares = 0.0
        for (v in values) sumSquares += v.toDouble() * v.toDouble()
        val norm = sqrt(sumSquares)
        if (norm <= maxNorm || norm == 0.0) return bytes

        val scale = min(1.0, maxNorm / norm)
        val out = java.nio.ByteBuffer.allocate(bytes.size).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (v in values) out.putFloat((v * scale).toFloat())
        return out.array()
    }
}
