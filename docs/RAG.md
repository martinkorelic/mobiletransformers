# Retrieval-Augmented Generation (RAG)

MobileTransformers runs retrieval on-device: an embedding model produces query/document vectors, an
on-device vector store (ObjectBox HNSW) does nearest-neighbour search, and the retrieved context is fed
to generation.

> **Scope of this page.** The **vector-store boundary**, **ingestion/chunking** and
> **grounded generation + `RagConfig`** are all implemented and documented below. The remaining
> gap is device acceptance: the instrumented `RagDeviceTest` and the ObjectBox parity smoke both
> require a package pushed to a device.

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

`searchType` is validated against the `SearchType` enum (`semantic` | `text`) when the config is
parsed, and `ORTRetriever` dispatches on the enum — an unrecognized value fails closed at the parse
boundary rather than reaching the retriever.

## Ingestion and chunking

`model.ingest(path, RagConfig(...))` chunks a document, embeds each chunk and inserts it into the
vector store. The loader is resolved from the file extension through `DOCUMENT_LOADER_REGISTRY`:

| Extension | Loader | Notes |
| --- | --- | --- |
| `.txt` | plain text | whole file, then chunked |
| `.md` | markdown | treated as text; no structural parsing |
| `.jsonl` | JSON Lines | one document per line |

**PDF and Word are rejected fail-closed** — there is no on-device extractor, and silently importing an
empty document would poison retrieval. Convert to `.txt`/`.md` first.

Chunking is pure character windowing (`chunkSize` / `chunkOverlap` on `RagConfig`), so it is JVM-testable
and independent of the tokenizer. `IngestionProgress` reports per-chunk progress.

## Retrieval on its own

`model.retrieve(query, RagConfig())` returns ranked matches with scores and generates nothing. It is a
first-class operation, not merely the first half of `generateWithRag`, for two reasons.

**It is the only part of the retrieval story a pure encoder can show.** `retrieve` requires
`capabilities.supportsEmbedding` and nothing else — no generative head, no KV cache. An
`all-MiniLM-L6-v2` package installed on its own can search, and the sample app's drawer reflects that
by offering Retrieval while hiding Chat.

**It is the only place retrieval can be judged.** Inside a grounded answer, bad retrieval and a model
ignoring good retrieval produce the same symptom — a wrong answer — and are indistinguishable. Looking
at the matches directly separates them, which is why this is a screen in the sample app and not a
debug flag.

```kotlin
val hits = model.retrieve("how do I merge an adapter?", RagConfig(topK = 4, minScore = 0.2))
hits.matches.forEach { println("%.3f  %s  %s".format(it.score, it.title, it.text)) }

println("${hits.matches.size} passages from ${hits.documentCount} documents")
```

A match is a **chunk**, not a document: ingestion splits each file into `chunkSize` pieces and each is
stored, ranked and returned separately, so several matches routinely come from one file. `title` is
that file's name, `chunkId` is `<documentId>#<n>`, and `RetrievalResult.documentCount` /
`documentTitles` do the grouping — which is how the sample app can say "found 4 passages in 2
documents" rather than conflating the two counts.

## Grounded generation

`model.generateWithRag(query, rag, generation, promptStrategy, callback)` runs retrieve → assemble →
generate and returns a `GroundedResult` carrying the answer, the matches, **and the assembled prompt**
so the exact context sent to the model is inspectable. `PromptAssembler` is overridable via
`PromptStrategy`.

Pass a `GenerateCallback` to stream the answer. It observes the generation leg, so its first event is
also the signal that retrieval finished. A grounded turn is the slowest operation in the SDK — an
embedding pass, a vector search, then a decode over a prompt several hundred tokens longer than a plain
one — and without a callback it produces nothing at all until it is completely done, which is not
distinguishable from a hang.

Pass a `RetrieveCallback` to receive the matches **when they are found**, rather than in the returned
`GroundedResult` after the answer is complete. The two arrive tens of seconds apart, so a UI that wants
to show what it retrieved before the answer built on it — as the sample app's Chat screen does, as its
own turn above the reply — needs them at that moment.

`RagConfig` carries `topK`, `minScore` (a similarity floor applied during search), `searchType` and
`indexingMode`. A changed config applies on every call: query-shaping fields are pushed onto the live
retriever, and a change of embedding model rebuilds it.

`indexingMode` is `precompute` in v1; `dynamic` fails closed rather than silently behaving like
`precompute`.

## Not yet (tracked)

- **Device acceptance**: the instrumented `RagDeviceTest` and the ObjectBox parity smoke
  both `assumeTrue` on a package pushed to a device.
