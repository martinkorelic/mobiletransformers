package com.martinkorelic.mobiletransformers.federated

import android.util.Log
import com.martinkorelic.mobiletransformers.ORTTrainerNative
import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap

/**
 * The bounded local training one federated round performs between import and export.
 *
 * A seam rather than a direct call into `TrainingRepository` for two reasons: the round must be
 * host-testable without a native session, and *what* "bounded" means (steps, dataset, scheduler
 * constraints) belongs to #34's scheduler and the caller's `ORTTrainingConfig`, not to the federated
 * layer. This layer only guarantees that training happens **between** the import and the export.
 */
fun interface LocalRoundTraining {
    /** Runs one round's worth of local training. [round] is passed for logging/telemetry only. */
    suspend fun trainOneRound(round: Int)
}

/** What one round did, including the number that decides whether federation is affordable at all. */
class FederatedRoundResult(
    val round: Int,
    /** Tensors written into the local checkpoint from the global record; 0 for the first round. */
    val importedTensors: Int,
    /** The serialized record this device would upload. */
    val update: ByteArray,
    /** True when local training ran between the import and the export. */
    val trainedLocally: Boolean,
) {
    /** On-device communication size for this round's adapter payload — the #36 DoD measurement. */
    val payloadBytes: Int get() = update.size

    fun describe(): String =
        "round $round: imported $importedTensors tensor(s), trained=$trainedLocally, " +
            "upload payload $payloadBytes B"
}

/**
 * One federated round on the device: **import** the global adapter, train locally under the caller's
 * bounds, **export** the updated adapter (#36 step 2).
 *
 * ```
 * global record ──import──▶ local checkpoint ──train──▶ local checkpoint ──export──▶ update record
 * ```
 *
 * Everything that decides what leaves the device — consent, TLS, auth, clipping, tensor identity —
 * lives in [FederatedRound] and [FederatedConfig]; this class owns only the **ordering**, which is
 * itself a correctness property: exporting before training would upload the global adapter back
 * unchanged, and a round that trains without importing first diverges from the cohort silently. Both
 * are the "two halves each verified alone" failure this project keeps paying for, so the order is
 * asserted here rather than left to the caller's call sequence.
 *
 * The transport is deliberately absent. This returns bytes and accepts bytes; whether they travel over
 * HTTPS to `federated serve`, or over `adb` in a device test, is the caller's problem — which is what
 * makes the round runnable end to end without a server.
 */
class FederatedTrainingRepository(
    private val round: AdapterExchange,
    private val localTraining: LocalRoundTraining,
    private val baseModelId: String,
    private val packageRevision: String = "",
    private val peftMethod: String = "lora",
) {

    /**
     * Runs one round.
     *
     * @param globalRecord the aggregate from the previous round, or `null` for the very first round
     *   (there is nothing to import yet — a device must be able to join a cohort that has not published
     *   an aggregate, and refusing would make round 0 impossible).
     * @param roundNumber stamped on the exported record so the gateway can reject a stale submission.
     * @param metrics local metrics to report alongside the update. Loss and example counts only —
     *   never anything derived from the raw examples themselves.
     */
    suspend fun runRound(
        globalRecord: ByteArray?,
        roundNumber: Int,
        metrics: Map<String, Double> = emptyMap(),
        train: Boolean = true,
    ): FederatedRoundResult {
        val imported = if (globalRecord == null) {
            Log.i(LOG_TAG, "round $roundNumber: no global record, starting from the local adapter")
            0
        } else {
            val count = round.importAggregate(globalRecord)
            Log.i(LOG_TAG, "round $roundNumber: imported $count tensor(s) from the global record")
            count
        }

        if (train) localTraining.trainOneRound(roundNumber)

        val update = round.exportUpdate(
            baseModelId = baseModelId,
            packageRevision = packageRevision,
            peftMethod = peftMethod,
            round = roundNumber,
            metrics = metrics,
        )

        val result = FederatedRoundResult(roundNumber, imported, update, train)
        Log.i(LOG_TAG, result.describe())
        return result
    }

    companion object {
        private const val LOG_TAG = "FederatedTraining"

        /**
         * Builds a repository over a live training session.
         *
         * The only place [NativeCheckpointTensorStore] is constructed outside a test — callers get the
         * JNI binding by asking for a round, not by reaching for the native methods themselves.
         */
        internal fun forSession(
            config: FederatedConfig,
            handoff: WeightHandoffMap,
            trainer: ORTTrainerNative,
            localTraining: LocalRoundTraining,
            baseModelId: String,
            packageRevision: String = "",
            peftMethod: String = "lora",
        ): FederatedTrainingRepository = FederatedTrainingRepository(
            round = FederatedRound(config, handoff, NativeCheckpointTensorStore(trainer)),
            localTraining = localTraining,
            baseModelId = baseModelId,
            packageRevision = packageRevision,
            peftMethod = peftMethod,
        )
    }
}
