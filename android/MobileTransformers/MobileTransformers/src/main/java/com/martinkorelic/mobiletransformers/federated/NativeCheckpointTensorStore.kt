package com.martinkorelic.mobiletransformers.federated

import com.martinkorelic.mobiletransformers.ORTTrainerNative

/**
 * The one place [CheckpointTensorStore] is bound to a live ORT training session (#36).
 *
 * `nativeExportCheckpointTensor` / `nativeImportCheckpointTensor` move **bytes only** — the record
 * format is owned by [AdapterTensorCodec], which is pinned against the cross-language golden. This
 * class is the whole of the glue, deliberately: everything that decides *what* is read or written
 * (order, names, dtypes, clipping, consent) lives in [FederatedRound] and the handoff map, so a second
 * opinion about tensor identity cannot grow here.
 *
 * A dead session is an error rather than an empty read. Returning `null` for every name would let a
 * round produce the codec's "the package and the checkpoint disagree about which factors exist"
 * message, which points at the package — the wrong place to look when the truth is that no session was
 * ever created.
 */
internal class NativeCheckpointTensorStore(
    private val trainer: ORTTrainerNative,
) : CheckpointTensorStore {

    override fun read(name: String): ByteArray? =
        trainer.nativeExportCheckpointTensor(requireSession(), name)

    override fun write(name: String, data: ByteArray): Boolean =
        trainer.nativeImportCheckpointTensor(requireSession(), name, data)

    private fun requireSession(): Long {
        val session = trainer.trainingSessionHandle()
        if (session == 0L) {
            throw FederatedRecordException(
                "no live ORT training session; a federated round reads and writes the checkpoint of " +
                    "an open session, so prepare training before starting the round"
            )
        }
        return session
    }
}
