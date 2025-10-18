package com.martinkorelic.ortmobile.entity

import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id
import io.objectbox.annotation.Index
import io.objectbox.annotation.IndexType
import io.objectbox.annotation.VectorDistanceType
import org.w3c.dom.Document

// Common interface for all vector entities
interface VectorEntityInterface {
    var id: Long
    var name: String
    var document : String
    var content: String
    var embedding: FloatArray
    var metadata: String
    var timestamp: Long
}

@Entity
data class VectorEntity64(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 64, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity64)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}

@Entity
data class VectorEntity128(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 128, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity128)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}

@Entity
data class VectorEntity256(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 256, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity256)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}

@Entity
data class VectorEntity384(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 384, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity384)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}

@Entity
data class VectorEntity512(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 512, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity512)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}
@Entity
data class VectorEntity768(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 768, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity768)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}

@Entity
data class VectorEntity1024(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 1024, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity1024)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}
@Entity
data class VectorEntity1536(
    @Id override var id: Long = 0,
    override var name: String = "",
    override var document: String = "",
    @Index(type = IndexType.VALUE) override var content: String = "",
    @HnswIndex(dimensions = 1536, distanceType = VectorDistanceType.COSINE)
    override var embedding: FloatArray = floatArrayOf(),
    override var metadata: String = "",
    override var timestamp: Long = System.currentTimeMillis()
) : VectorEntityInterface {
    override fun equals(other: Any?) = (other as? VectorEntity1536)?.let {
        id == it.id && embedding.contentEquals(it.embedding)
    } ?: false

    override fun hashCode() = 31 * id.hashCode() + embedding.contentHashCode()
}

// TODO: Could add other popular dimensions...