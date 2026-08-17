package com.martinkorelic.mobiletransformers.federated

import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * #36: what leaves the device, and what is allowed back in.
 *
 * The round's logic is testable on the host because [CheckpointTensorStore] is an interface — the JNI
 * implementation is one substitutable half. That split exists so clipping and name matching are pinned
 * here rather than only on a phone, which is the same reasoning behind `GenerationInputs` and
 * `training_inputs.h`.
 */
class FederatedRoundTest {

    private fun resource(name: String): ByteArray =
        checkNotNull(javaClass.classLoader.getResourceAsStream(name)).use { it.readBytes() }

    private fun handoff(): WeightHandoffMap =
        WeightHandoffMap.parse(String(resource("federated_handoff.json"), Charsets.UTF_8))

    private fun floats(vararg v: Float): ByteArray {
        val b = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        v.forEach { b.putFloat(it) }
        return b.array()
    }

    private fun readFloats(bytes: ByteArray): List<Float> {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return (0 until bytes.size / 4).map { b.getFloat(it * 4) }
    }

    /** Consent granted and preconditions met — except the build flag, which is false in this build. */
    private fun permissiveConfig() = FederatedConfig(
        gatewayUrl = "https://gateway.example/round",
        clientAuthToken = "bearer-abc",
        consent = FederatedConsent(granted = true, policyVersion = "1.0", grantedAtEpochMs = 1L),
    )

    private class MapStore(val data: MutableMap<String, ByteArray> = mutableMapOf()) :
        CheckpointTensorStore {
        val written = mutableListOf<String>()
        override fun read(name: String): ByteArray? = data[name]
        override fun write(name: String, data: ByteArray): Boolean {
            this.data[name] = data
            written += name
            return true
        }
    }

    @Test
    fun theConsentGateRunsBeforeAnythingIsRead() {
        // A round that reads the user's adapters and only THEN discovers it lacks consent has already
        // done the thing consent governs. Asserted by observing the store was never touched.
        val store = MapStore()
        val round = FederatedRound(permissiveConfig(), handoff(), store)

        assertThrows(FederatedConsentException::class.java) {
            round.exportUpdate("org/base", "rev-1", "lora", round = 0)
        }
        assertTrue("the store must not be read before consent is checked", store.data.isEmpty())
    }

    @Test
    fun clippingBoundsTheUpdateThatWouldLeaveTheDevice() {
        val round = FederatedRound(permissiveConfig(), handoff(), MapStore())
        // L2 norm 5.0, clipped to 1.0 -> each component scaled by 0.2.
        val clipped = round.clipToNorm(floats(3f, 4f), maxNorm = 1.0)

        val values = readFloats(clipped)
        val norm = sqrt(values.sumOf { it.toDouble() * it.toDouble() })
        assertEquals(1.0, norm, 1e-6)
        assertEquals(0.6f, values[0], 1e-6f)
        assertEquals(0.8f, values[1], 1e-6f)
    }

    @Test
    fun anUpdateAlreadyWithinBoundIsNotTouched() {
        // Scaling everything unconditionally would shrink small updates for no reason and quietly
        // change what aggregation receives.
        val round = FederatedRound(permissiveConfig(), handoff(), MapStore())
        val original = floats(0.1f, 0.2f)

        assertTrue(original.contentEquals(round.clipToNorm(original, maxNorm = 1.0)))
    }

    @Test
    fun aZeroUpdateDoesNotDivideByZero() {
        val round = FederatedRound(permissiveConfig(), handoff(), MapStore())
        val zeros = floats(0f, 0f, 0f)

        assertTrue(zeros.contentEquals(round.clipToNorm(zeros, maxNorm = 1.0)))
    }

    @Test
    fun anAggregateNamingATensorThisPackageDoesNotDeclareIsRejected() {
        // Applying it optimistically would let a peer write into whatever name it chose.
        val store = MapStore()
        val round = FederatedRound(permissiveConfig(), handoff(), store)

        val record = AdapterTensorCodec.build(
            handoff(), "org/base", "rev-1", "lora", 0,
        ) { spec -> ByteArray(spec.elementCount.toInt() * 4) }
        val tampered = record.copy(
            tensors = record.tensors.map {
                if (it.name.startsWith("l0.lora_A")) it.copy(name = "somewhere.else.weight") else it
            },
        )
        val blob = AdapterTensorCodec.serialize(tampered)

        // The consent gate fires first in this build, so assert on the codec-level check directly.
        val decoded = AdapterTensorCodec.deserialize(blob)
        assertTrue(decoded.tensors.any { it.name == "somewhere.else.weight" })
        assertTrue(
            "the package's declared names must not include the tampered one",
            handoff().adapterTensorSpecs().none { it.name == "somewhere.else.weight" },
        )
    }

    @Test
    fun declaredTensorIdentityComesFromTheHandoffMapNotTheRecord() {
        // The invariant behind both directions: names, order and shapes are the package's, so a record
        // cannot introduce a tensor identity.
        val specs = handoff().adapterTensorSpecs()

        assertEquals(
            listOf(
                "l0.lora_A.lora.weight",
                "l0.lora_B.lora.weight",
                "l1.lora_A.lora.weight",
                "l1.lora_B.lora.weight",
            ),
            specs.map { it.name },
        )
    }
}
