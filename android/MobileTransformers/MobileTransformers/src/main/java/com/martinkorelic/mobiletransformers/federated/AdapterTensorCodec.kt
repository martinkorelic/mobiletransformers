package com.martinkorelic.mobiletransformers.federated

import com.martinkorelic.mobiletransformers.MobileTransformersException
import com.martinkorelic.mobiletransformers.packages.PackageFormat
import com.martinkorelic.mobiletransformers.packages.WeightHandoffMap
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** A federated record that could not be read, or could not be built. Always names the offender. */
class FederatedRecordException(message: String) : MobileTransformersException(message)

/** One tensor inside a record: its checkpoint identity, description, and raw little-endian payload. */
data class AdapterTensor(
    val name: String,
    val dtype: String,
    val shape: List<Long>,
    val role: String,
    val payload: ByteArray,
    val aggregation: String = AdapterTensorCodec.AGGREGATION,
) {
    // Data classes compare ByteArray by reference; for a wire object that is a trap, since two records
    // carrying identical bytes would compare unequal.
    override fun equals(other: Any?): Boolean =
        this === other || (other is AdapterTensor &&
            name == other.name && dtype == other.dtype && shape == other.shape &&
            role == other.role && aggregation == other.aggregation &&
            payload.contentEquals(other.payload))

    override fun hashCode(): Int =
        (((name.hashCode() * 31 + dtype.hashCode()) * 31 + shape.hashCode()) * 31 +
            role.hashCode()) * 31 + payload.contentHashCode()
}

/** The decoded form of a `FederatedAdapterRecord`. */
data class FederatedRecord(
    val baseModelId: String,
    val packageRevision: String,
    val peftMethod: String,
    val adapterFormatVersion: String,
    val round: Int,
    val tensors: List<AdapterTensor>,
    val metrics: Map<String, Double> = emptyMap(),
    val schemaVersion: String = AdapterTensorCodec.SCHEMA_VERSION,
    val minReaderVersion: String = AdapterTensorCodec.MIN_READER_VERSION,
)

/**
 * Kotlin **mirror** of `federated/adapter_record.py` — not an independent implementation.
 *
 * Wire format: `uint32` LE header length, UTF-8 JSON header, then the raw little-endian payloads
 * concatenated in codec order. Pinned byte-for-byte by `tests/federated/fixtures/federated_record.golden.bin`,
 * which both languages are tested against; if these bytes and Python's ever disagree, this file is
 * wrong by definition.
 *
 * ## Why the JSON is hand-built
 *
 * Python emits the header with `json.dumps(header, sort_keys=True)`, whose defaults are **`", "` and
 * `": "` separators — with spaces — and whose key sorting is recursive. Gson produces neither by
 * default. Serializing through Gson and hoping would produce a record that decodes fine and fails the
 * golden, so the writer below builds the exact string. (Reading stays permissive: any valid JSON is
 * accepted, because a peer's whitespace is not our business.)
 *
 * ## Vocabulary
 *
 * Rank-r ADAPTER FACTORS, not merged weights — the decision recorded in #35. Merged roles stay
 * **accepted on read** (a peer may hold an older record, and rejecting it as "unknown role" is worse
 * than accepting it) but nothing here produces them.
 */
object AdapterTensorCodec {

    const val SCHEMA_VERSION = "1.0"
    const val MIN_READER_VERSION = "1.0"
    const val READER_VERSION = "1.0"

    /** The only aggregation v1 understands. Anything else is rejected on read. */
    const val AGGREGATION = "weighted_average"

    /** Write vocabulary (rank-r factors) plus the legacy merged roles, which are read-only. */
    val SUPPORTED_ROLES = setOf(
        "shared_A", "intermediate", "adapter_A", "adapter_B",
        "weight", "weight_quantized", "scale", "zero_point",
    )

    /** Bytes per element. `int4` is deliberately absent — it has no byte-aligned element size. */
    val DTYPE_SIZES = mapOf(
        "float16" to 2, "float32" to 4, "float64" to 8,
        "int8" to 1, "uint8" to 1, "int32" to 4,
    )

    private const val HEADER_LENGTH_BYTES = 4

    /**
     * Builds a record from the handoff map's declared factors plus the payload for each.
     *
     * @param payloadFor supplies the raw bytes for a tensor by its checkpoint name — on device this
     *   reads the ORT checkpoint. Matching by NAME rather than by iteration order is deliberate: the
     *   Python simulation had a defect where tensors were paired by checkpoint iteration order, which
     *   would write one layer's `lora_A` over another's, and differing shapes caught it only "mostly".
     */
    fun build(
        handoff: WeightHandoffMap,
        baseModelId: String,
        packageRevision: String,
        peftMethod: String,
        round: Int,
        metrics: Map<String, Double> = emptyMap(),
        payloadFor: (WeightHandoffMap.AdapterTensorSpec) -> ByteArray?,
    ): FederatedRecord {
        val tensors = handoff.adapterTensorSpecs().map { spec ->
            val bytes = payloadFor(spec)
                ?: throw FederatedRecordException(
                    "no checkpoint data for adapter tensor '${spec.name}' (role ${spec.role}). The " +
                        "package and the checkpoint disagree about which factors exist."
                )
            val expected = expectedByteLength(spec.dtype, spec.shape, spec.name)
            if (bytes.size != expected) {
                throw FederatedRecordException(
                    "adapter tensor '${spec.name}' is ${bytes.size} bytes but its declared " +
                        "${spec.dtype}${spec.shape} needs $expected"
                )
            }
            AdapterTensor(spec.name, spec.dtype, spec.shape, spec.role, bytes)
        }
        return FederatedRecord(
            baseModelId = baseModelId,
            packageRevision = packageRevision,
            peftMethod = peftMethod,
            // Tracks the handoff map's schemaVersion (F1/F8), NOT this codec's.
            adapterFormatVersion = handoff.schemaVersion,
            round = round,
            tensors = tensors,
            metrics = metrics,
        )
    }

    /** Serializes to the pinned byte layout. */
    fun serialize(record: FederatedRecord): ByteArray {
        var offset = 0
        val entries = record.tensors.map { tensor ->
            val entry = tensorHeader(tensor, offset)
            offset += tensor.payload.size
            entry
        }

        val header = buildString {
            append('{')
            // Top-level keys in the order `sort_keys=True` produces them.
            appendField("adapterFormatVersion", jsonString(record.adapterFormatVersion))
            append(", "); appendField("baseModelId", jsonString(record.baseModelId))
            append(", "); appendField("metrics", jsonObject(record.metrics.toSortedMap().map {
                jsonString(it.key) to jsonNumber(it.value)
            }))
            append(", "); appendField("minReaderVersion", jsonString(record.minReaderVersion))
            append(", "); appendField("mobiletransformersPackageRevision", jsonString(record.packageRevision))
            append(", "); appendField("peftMethod", jsonString(record.peftMethod))
            append(", "); appendField("round", record.round.toString())
            append(", "); appendField("schemaVersion", jsonString(record.schemaVersion))
            append(", "); appendField("tensors", entries.joinToString(", ", "[", "]"))
            append('}')
        }

        val headerBytes = header.toByteArray(Charsets.UTF_8)
        val out = ByteBuffer
            .allocate(HEADER_LENGTH_BYTES + headerBytes.size + record.tensors.sumOf { it.payload.size })
            .order(ByteOrder.LITTLE_ENDIAN)
        out.putInt(headerBytes.size)
        out.put(headerBytes)
        record.tensors.forEach { out.put(it.payload) }
        return out.array()
    }

    /**
     * Reads a record, failing closed on anything it cannot fully account for.
     *
     * Order matters: the version gate runs **before** any offset is trusted, so a record from a newer
     * SDK is refused rather than parsed with this reader's assumptions about its layout.
     */
    fun deserialize(blob: ByteArray): FederatedRecord {
        if (blob.size < HEADER_LENGTH_BYTES) {
            throw FederatedRecordException("record is ${blob.size} bytes, too short to hold a header length")
        }
        val buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
        val headerLength = buffer.int
        if (headerLength < 0 || HEADER_LENGTH_BYTES + headerLength > blob.size) {
            throw FederatedRecordException(
                "record declares a $headerLength-byte header but is only ${blob.size} bytes"
            )
        }
        val headerJson = String(blob, HEADER_LENGTH_BYTES, headerLength, Charsets.UTF_8)
        val header = try {
            com.google.gson.JsonParser.parseString(headerJson).asJsonObject
        } catch (e: Exception) {
            throw FederatedRecordException("record header is not valid JSON: ${e.message}")
        }

        val schemaVersion = header.stringOr("schemaVersion", SCHEMA_VERSION)
        val minReader = header.stringOr("minReaderVersion", MIN_READER_VERSION)
        if (!PackageFormat.checkCompat(schemaVersion, minReader, READER_VERSION)) {
            throw FederatedRecordException(
                "federated record schemaVersion $schemaVersion (minReader $minReader) needs a newer " +
                    "SDK; this reader is $READER_VERSION"
            )
        }

        val payloadBase = HEADER_LENGTH_BYTES + headerLength
        val payloadLength = blob.size - payloadBase
        val tensors = header.getAsJsonArray("tensors").orEmpty().map { element ->
            val obj = element.asJsonObject
            val name = obj.stringOr("name", "")
            val dtype = obj.stringOr("dtype", "")
            val role = obj.stringOr("role", "")
            val aggregation = obj.stringOr("aggregation", "")
            val shape = obj.getAsJsonArray("shape").orEmpty().map { it.asLong }
            val byteOffset = obj.get("byteOffset").asInt
            val byteLength = obj.get("byteLength").asInt

            if (role !in SUPPORTED_ROLES) {
                throw FederatedRecordException("tensor '$name' has unsupported role '$role'")
            }
            if (aggregation != AGGREGATION) {
                throw FederatedRecordException(
                    "tensor '$name' has unsupported aggregation '$aggregation' (expected '$AGGREGATION')"
                )
            }
            if (byteOffset < 0 || byteLength < 0 || byteOffset + byteLength > payloadLength) {
                throw FederatedRecordException(
                    "tensor '$name' spans [$byteOffset, ${byteOffset + byteLength}) of a " +
                        "$payloadLength-byte payload"
                )
            }
            val expected = expectedByteLength(dtype, shape, name)
            if (byteLength != expected) {
                throw FederatedRecordException(
                    "tensor '$name' declares $byteLength bytes but $dtype$shape needs $expected"
                )
            }
            AdapterTensor(
                name = name,
                dtype = dtype,
                shape = shape,
                role = role,
                payload = blob.copyOfRange(payloadBase + byteOffset, payloadBase + byteOffset + byteLength),
                aggregation = aggregation,
            )
        }

        return FederatedRecord(
            baseModelId = header.stringOr("baseModelId", ""),
            packageRevision = header.stringOr("mobiletransformersPackageRevision", ""),
            peftMethod = header.stringOr("peftMethod", ""),
            adapterFormatVersion = header.stringOr("adapterFormatVersion", ""),
            round = header.get("round")?.asInt ?: 0,
            tensors = tensors,
            schemaVersion = schemaVersion,
            minReaderVersion = minReader,
        )
    }

    /**
     * Asserts the record describes the same package this device holds (F1/F8).
     *
     * `adapterFormatVersion` tracks the handoff map's `schemaVersion`; a mismatch means the peer's
     * factors are described by a different schema than ours, which is not a difference to paper over.
     */
    fun checkFormat(record: FederatedRecord, handoff: WeightHandoffMap) {
        if (record.adapterFormatVersion != handoff.schemaVersion) {
            throw FederatedRecordException(
                "record adapterFormatVersion ${record.adapterFormatVersion} does not match this " +
                    "package's handoff-map schemaVersion ${handoff.schemaVersion}"
            )
        }
    }

    private fun expectedByteLength(dtype: String, shape: List<Long>, name: String): Int {
        val size = DTYPE_SIZES[dtype]
            ?: throw FederatedRecordException(
                "tensor '$name' has unsupported dtype '$dtype' (supported: ${DTYPE_SIZES.keys.sorted()})"
            )
        return (shape.fold(1L) { acc, d -> acc * d } * size).toInt()
    }

    private fun tensorHeader(tensor: AdapterTensor, offset: Int): String = buildString {
        // Nested keys sort too, because Python's sort_keys is recursive.
        append('{')
        appendField("aggregation", jsonString(tensor.aggregation))
        append(", "); appendField("byteLength", tensor.payload.size.toString())
        append(", "); appendField("byteOffset", offset.toString())
        append(", "); appendField("dtype", jsonString(tensor.dtype))
        append(", "); appendField("name", jsonString(tensor.name))
        append(", "); appendField("role", jsonString(tensor.role))
        append(", "); appendField("shape", tensor.shape.joinToString(", ", "[", "]"))
        append('}')
    }

    private fun StringBuilder.appendField(key: String, rendered: String) {
        append(jsonString(key)).append(": ").append(rendered)
    }

    private fun jsonObject(pairs: List<Pair<String, String>>): String =
        pairs.joinToString(", ", "{", "}") { "${it.first}: ${it.second}" }

    /** Python renders a whole-valued float as `1.0`, and Kotlin's `Double.toString` agrees. */
    private fun jsonNumber(value: Double): String = value.toString()

    /** Minimal JSON string escaping — the keys and values here are names and versions, not free text. */
    private fun jsonString(value: String): String = buildString {
        append('"')
        for (ch in value) {
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(ch)
            }
        }
        append('"')
    }

    private fun com.google.gson.JsonObject.stringOr(key: String, fallback: String): String =
        if (has(key) && !get(key).isJsonNull) get(key).asString else fallback

    private fun com.google.gson.JsonArray?.orEmpty(): List<com.google.gson.JsonElement> =
        this?.toList() ?: emptyList()
}
