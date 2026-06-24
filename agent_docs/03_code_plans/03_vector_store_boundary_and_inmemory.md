# Vector Store Boundary & In-Memory Test Implementation

**Priority #24 | Prerequisites: #16 (`00_code_plans/05_android_facade_foundation.md`) | Blocks: #25 (`03_code_plans/04`), #26 (`03_code_plans/05`)**

> **Consumes `00_code_plans/09`.** Search mode is the `SearchType` enum (09: `semantic`/`text`), not a bare `String`. The supported-dimension set is **config-driven data** (a single declared dimension registry), not a scatter of `if dim == 256` literals — this resolves the `VectorEntity.kt:164` `// TODO: Could add other popular dimensions` by making the set extensible from one place rather than hand-edited per call site. ObjectBox still needs a declared `@HnswIndex` entity per dimension (a platform constraint), so "dynamic" here means *one declared registry + generated/extensible entities*, with a clear fail-closed error for any dimension not in the registry — never a silent box pick.

## Purpose

Put a small, testable `VectorStore` interface around the current ObjectBox implementation so chunking, ingestion, and retrieval are unit-testable on the JVM without Android/ObjectBox, while preserving the exact ObjectBox semantics (COSINE distance, the `1 - score` similarity conversion, `minScore` filtering, the config-declared dimension set, and the separate text-search path). ObjectBox stays the default; an `InMemoryVectorStore` enables tests.

## Touched / new files

Kotlin:
- NEW `android/.../rag/VectorStore.kt` — the interface + `RagDocument` / `RagMatch` types.
- NEW `android/.../rag/ObjectBoxVectorStore.kt` — wraps `ORTVectorDatabase`.
- NEW (test source set) `.../rag/InMemoryVectorStore.kt` — pure-Kotlin cosine store for JVM tests.
- `android/.../ORTVectorDatabase.kt` — unchanged behavior; the wrapper calls `queryDocuments` (`:217-246`), `searchVectors`/`nearestNeighbors` (`:255-256`), the `1 - result.score` conversion (`:262`), `minScore` filter (`:238`), and `queryByContent` (`:293`).
- `android/.../entity/VectorEntity.kt` — the per-dimension entity classes (`:23-162`) plus the `// TODO: Could add other popular dimensions` (`:164`): the `embeddingDimension`→entity map becomes a single declared **dimension registry** (one source of truth for the supported set), so adding a dimension is one registry entry + its `@HnswIndex` entity, not edits scattered across call sites. The wrapper rejects unsupported sizes with a clear error.
- `android/.../ORTVectorDatabase.kt` — resolve `// TODO: Clear embeddings from returning` (`:226`): strip the embedding `FloatArray` from results before returning (memory), since `RagMatch` only needs the document + similarity.
- `android/.../ORTRetriever.kt` — retrieval (`:76-134`) routes through `VectorStore`.

## Data contracts / interfaces

```kotlin
data class RagDocument(
    val id: String, val title: String, val text: String,
    val metadata: Map<String, String> = emptyMap(),
)
data class RagMatch(val document: RagDocument, val score: Double)   // score = similarity (already 1 - distance)

interface VectorStore {
    fun insert(document: RagDocument, embedding: FloatArray): Long
    fun search(queryEmbedding: FloatArray, topK: Int, minScore: Double = 0.0): List<RagMatch>
    fun textSearch(query: String, topK: Int): List<RagMatch>
    fun count(): Long
    fun close()
}
```

### Preserved ObjectBox semantics (must be documented + tested)

- **Distance**: COSINE via `@HnswIndex(... distanceType = VectorDistanceType.COSINE)` (`VectorEntity.kt:23-162`).
- **Similarity**: ObjectBox `findWithScores()` returns a distance; similarity = `1 - result.score` (`ORTVectorDatabase.kt:262`). `RagMatch.score` carries the **similarity** (post-conversion), so callers never re-convert.
- **`minScore`**: applied as `it.second >= minScore` after conversion (`ORTVectorDatabase.kt:238`).
- **Dimensions**: the supported set (today `{64,128,256,384,512,768,1024,1536}`) lives in **one declared dimension registry**, not hardcoded per call site; an `embeddingDimension` not in the registry must throw a clear error, not silently pick a box. Adding a dimension = one registry entry + its `@HnswIndex` entity.
- **No embeddings in results**: returned `RagMatch`es carry document + similarity only; embedding vectors are stripped (`ORTVectorDatabase.kt:226`).
- **Text search**: `textSearch` maps to the VALUE-index `queryByContent` (`ORTVectorDatabase.kt:293`), assigning a fixed similarity (today `1.0`, `ORTRetriever.kt:118`) — document that text matches are not similarity-ranked.

## Implementation steps

1. Define `VectorStore` + `RagDocument` + `RagMatch`.
2. `ObjectBoxVectorStore`: delegate to `ORTVectorDatabase`; centralize the dimension→entity mapping with an explicit unsupported-dimension error; convert results into `RagMatch` (similarity already applied).
3. `InMemoryVectorStore`: store `(RagDocument, FloatArray)`; `search` computes cosine similarity, applies `topK` + `minScore`; `textSearch` does a substring/contains match returning fixed-score `RagMatch`es — mirroring ObjectBox text semantics.
4. Route `ORTRetriever` retrieval and the future ingestion (#25) through `VectorStore`.
5. Document the `1 - score` and text-vs-vector semantics in `docs/RAG.md` (`05_code_plans/04`).

## Interactions

- **#25 (ingestion)**: writes via `VectorStore.insert`.
- **#26 (grounded generation)**: reads via `VectorStore.search` / `textSearch`.
- **`02_code_plans/04` (Hub pull)**: embedding model + its dimension come from the pulled package; the dimension must be one of the eight or the store rejects it.

## Tests & smokes

- **`InMemoryVectorStore` unit tests (JVM)**: insert/search/count; cosine ordering correct; `topK` and `minScore` honored.
- **Score-semantics test**: a known distance maps to the expected `1 - distance` similarity; `minScore` filters as documented.
- **Unsupported-dimension test**: `embeddingDimension = 300` (not in the registry) throws a clear error; a dimension added to the registry is accepted.
- **No-embeddings test**: returned matches contain no embedding vector (memory-stripped per `:226`).
- **Text-search test**: `textSearch` returns substring matches with the documented fixed score.
- **ObjectBox parity smoke (Android, supported dims)**: insert + semantic query returns the same ordering as the in-memory store for the same vectors.
