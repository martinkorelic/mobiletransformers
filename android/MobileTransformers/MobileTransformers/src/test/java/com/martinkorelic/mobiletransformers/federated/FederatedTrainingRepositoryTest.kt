package com.martinkorelic.mobiletransformers.federated

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * #36: the ORDER of a round is the thing this class owns, so the order is what is asserted.
 *
 * Export-before-train uploads the global adapter back unchanged (the round contributes nothing while
 * looking successful); train-without-import diverges from the cohort silently. Both are the project's
 * recurring failure shape — two halves each fine alone, the seam between them unverified — so the
 * sequence is recorded and asserted rather than left to the caller's call order.
 *
 * Robolectric because the round logs its outcome through `android.util.Log`, whose stubs THROW in a
 * plain JVM test (`isReturnDefaultValues = false`).
 */
@RunWith(RobolectricTestRunner::class)
class FederatedTrainingRepositoryTest {

    /** Records the sequence of operations; the payload content is irrelevant to what is under test. */
    private class RecordingExchange(
        private val importedCount: Int = 3,
        private val update: ByteArray = byteArrayOf(7, 7, 7, 7),
    ) : AdapterExchange {
        val calls = mutableListOf<String>()
        var lastRound: Int = -1
        var lastMetrics: Map<String, Double> = emptyMap()

        override fun exportUpdate(
            baseModelId: String,
            packageRevision: String,
            peftMethod: String,
            round: Int,
            metrics: Map<String, Double>,
        ): ByteArray {
            calls += "export"
            lastRound = round
            lastMetrics = metrics
            return update
        }

        override fun importAggregate(blob: ByteArray): Int {
            calls += "import"
            return importedCount
        }
    }

    private class RecordingTraining(private val calls: MutableList<String>) : LocalRoundTraining {
        var rounds = mutableListOf<Int>()
        override suspend fun trainOneRound(round: Int) {
            calls += "train"
            rounds += round
        }
    }

    private fun repository(
        exchange: AdapterExchange,
        training: LocalRoundTraining,
    ) = FederatedTrainingRepository(
        round = exchange,
        localTraining = training,
        baseModelId = "org/base",
        packageRevision = "rev-1",
    )

    @Test
    fun aRoundImportsThenTrainsThenExports() = runBlocking {
        val exchange = RecordingExchange()
        val training = RecordingTraining(exchange.calls)

        val result = repository(exchange, training).runRound(byteArrayOf(1, 2, 3), roundNumber = 4)

        assertEquals(listOf("import", "train", "export"), exchange.calls)
        assertEquals(3, result.importedTensors)
        assertEquals(4, exchange.lastRound)
        assertEquals(listOf(4), training.rounds)
        assertTrue(result.trainedLocally)
    }

    @Test
    fun theFirstRoundRunsWithNothingToImport() = runBlocking {
        // A device must be able to join a cohort that has not published an aggregate yet. Refusing
        // would make round 0 impossible; importing an empty blob would fail the codec's own checks.
        val exchange = RecordingExchange()
        val training = RecordingTraining(exchange.calls)

        val result = repository(exchange, training).runRound(null, roundNumber = 0)

        assertEquals(listOf("train", "export"), exchange.calls)
        assertEquals(0, result.importedTensors)
    }

    @Test
    fun theUploadPayloadSizeIsReported() {
        // The #36 DoD asks for the on-device communication size to be MEASURED, so the round has to
        // carry it out rather than leaving the caller to size the array.
        val result = FederatedRoundResult(0, 0, ByteArray(1_868_857), trainedLocally = true)

        assertEquals(1_868_857, result.payloadBytes)
        assertTrue(result.describe().contains("1868857 B"))
    }

    @Test
    fun anImportFailureEndsTheRoundBeforeAnythingIsUploaded() = runBlocking {
        // A half-applied global adapter is neither the local model nor the global one. Training and
        // exporting on top of it would upload an update derived from a state nobody chose.
        val exchange = object : AdapterExchange {
            val calls = mutableListOf<String>()
            override fun exportUpdate(
                baseModelId: String,
                packageRevision: String,
                peftMethod: String,
                round: Int,
                metrics: Map<String, Double>,
            ): ByteArray {
                calls += "export"
                return ByteArray(0)
            }

            override fun importAggregate(blob: ByteArray): Int =
                throw FederatedRecordException("aggregated record carries 'x', which this package does not declare")
        }
        val training = RecordingTraining(exchange.calls)

        assertThrows(FederatedRecordException::class.java) {
            runBlocking { repository(exchange, training).runRound(byteArrayOf(9), roundNumber = 1) }
        }
        assertFalse("nothing may be exported after a failed import", exchange.calls.contains("export"))
        assertTrue("training must not run on a half-applied adapter", training.rounds.isEmpty())
    }

    @Test
    fun localMetricsTravelWithTheUpdate() = runBlocking {
        val exchange = RecordingExchange()
        val training = RecordingTraining(exchange.calls)

        repository(exchange, training)
            .runRound(null, roundNumber = 2, metrics = mapOf("loss" to 0.5, "numExamples" to 8.0))

        assertEquals(mapOf("loss" to 0.5, "numExamples" to 8.0), exchange.lastMetrics)
    }
}
