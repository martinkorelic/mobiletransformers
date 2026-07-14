# Retrieval-Augmented Generation (RAG)

MobileTransformers runs retrieval on-device: an embedding model produces query/document vectors, an
on-device vector store (ObjectBox HNSW) does nearest-neighbour search, and the retrieved context is fed
to generation.

> **Scope of this page.** The **vector-store boundary** (`03_code_plans/03`, #25) is implemented and
> documented below. **Ingestion/chunking** (`03_code_plans/04`, #26) and **grounded generation +
> `RagConfig`** (`03_code_plans/05`, #27) are not yet implemented — this page is extended when those
> contracts lock.

## The `VectorStore` boundary

Retrieval goes through a small `VectorStore` interface (`com.martinkorelic.mobiletransformers.rag`), so
the logic is testable on the JVM with no Android/ObjectBox:

```kotlin
data class RagDocument(val id: String, val title: String, val text: String,
                       val metadata: Map<String, String> = emptyMap())
data class RagMatch(val document: RagDocument, val score: Double)  // score = similarity (1 - distance)

interface VectorStore {
    fun insert(document: RagDocument, embedding: FloatArray): Long
    fun search(queryEmbedding: FloatArray, topK: Int, minScore: Double = 0.0): List<RagMatch>
    fun textSearch(query: String, topK: Int): List<RagMatch>
    fun count(): Long
    fun close()
}
```

- **`ObjectBoxVectorStore`** is the default on-device backing store (wraps `ORTVectorDatabase`).
- **`InMemoryVectorStore`** (test source set) is a pure-Kotlin cosine store for JVM unit tests.
- Backends are pluggable by key via `VectorStoreRegistry` (F4); `objectbox` is the default key.

## Semantics (preserved and tested)

- **Distance:** COSINE (`@HnswIndex(distanceType = COSINE)`).
- **Similarity:** ObjectBox returns a cosine *distance*; similarity is `1 - distance`. `RagMatch.score`
  always carries the **similarity** (higher = closer), so callers never re-convert.
- **`minScore`:** filters on similarity after conversion.
- **Embeddings stripped:** results carry the document + similarity only; the embedding vector never
  crosses the boundary.
- **Text vs. semantic search:** `textSearch` is a substring/VALUE-index lookup and is **not**
  similarity-ranked — every hit carries a fixed score (`TEXT_SEARCH_SCORE`). `search` is the
  cosine-ranked semantic path.

## Embedding dimensions (fail-closed registry)

The supported embedding dimensions are declared once in `DimensionRegistry`:
`{64, 128, 256, 384, 512, 768, 1024, 1536}`. A dimension outside the registry is rejected with a clear
error — the store never silently picks a box. Adding a dimension is one `DimensionRegistry.register(dim)`
call plus a declared `@HnswIndex VectorEntity<dim>` entity (an ObjectBox platform constraint).

The embedding model and its dimension come from the pulled package's embedding/RAG variant; the
dimension must be one the registry supports or installation/retrieval fails closed.

## Not yet (tracked)

- **Ingestion / chunking** (`ingestData()`, `.txt`/`.md`/`.jsonl`, document-loader registry) — #26.
- **Grounded generation** (public `RagConfig`, retrieve → assemble prompt → generate) — #27.
- The `searchType` field becomes the `SearchType` enum (`semantic`/`text`) when the facade lands
  (#17/#19).
