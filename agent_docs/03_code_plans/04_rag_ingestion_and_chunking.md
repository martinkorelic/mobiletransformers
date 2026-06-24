# RAG Ingestion & Chunking

**Priority #25 | Prerequisites: #24 (`03_code_plans/03`), #22 (`03_code_plans/01`) | Blocks: #26 (`03_code_plans/05`)**

## Purpose

Implement the currently-empty `ORTRetriever.ingestData()` (`ORTRetriever.kt:172-175`, a double-TODO) so a user can do "bring your own documents" symmetrically with "bring your own dataset": local text/Markdown/JSONL → chunk → tokenize → embed → store → progress. Keep v1 scope strictly to plain text, Markdown, and JSONL — **no PDF/Word/HTML parsing** (explicit scope guard from the Tier-2 risks).

## Touched / new files

Kotlin:
- `android/.../ORTRetriever.kt` — implement `ingestData()` (`:172-175`); reuse the embedding session created in `createEmbeddingModel()` (`:26-63`) and `performEmbeddingStep(...)` (`:192-200`); store via `VectorStore.insert` (#24).
- NEW `android/.../rag/DocumentChunker.kt` — pure-Kotlin chunking by `chunkSize`/`chunkOverlap` (`ORTRagConfig.kt:25-26`); JVM-testable.
- NEW `android/.../rag/DocumentSource.kt` — readers for `.txt`, `.md`, `.jsonl` from the filesystem.
- `android/.../ORTRagConfig.kt` — reuse `chunkSize` (`:25`), `chunkOverlap` (`:26`), `maxTextLength` (`:24`); add `indexingMode` later (`05` plan), not here.
- `android/.../repository/RagRepository.kt` (or `LLMRepository` RAG path) — expose an `ingest(documents, progress)` orchestration with cooperative cancellation.

## Data contracts / interfaces

### Document input shape (JSONL line / record)

```json
{
  "id": "doc-001",
  "title": "Personal notes",
  "text": "Full document text",
  "metadata": { "source": "local" }
}
```

`.txt`/`.md` files map to one record each (filename → `id`/`title`, file body → `text`). JSONL = one record per line.

### Ingestion pipeline

```
for each DocumentSource record:
    text = readAndNormalize(record)              # honor maxTextLength
    chunks = DocumentChunker.split(text, chunkSize, chunkOverlap)
    for each chunk:
        ids = embeddingTokenizer.encode(chunk)
        emb = performEmbeddingStep(session, ids, mask, tokenTypeIds, 1, ids.size, embeddingDimension)
        vectorStore.insert(RagDocument(id="${record.id}#${i}", title=record.title, text=chunk, metadata), emb)
    progress.onIngested(record.id, chunkCount)
```

### Progress callback

```kotlin
interface IngestionProgress {
    fun onDocumentStart(id: String, totalDocs: Int)
    fun onChunkEmbedded(docId: String, chunkIndex: Int, totalChunks: Int)
    fun onDocumentComplete(id: String)
    fun onError(id: String?, error: Throwable)
}
```

## Implementation steps

1. `DocumentChunker.split(text, size, overlap)` — deterministic windowing; assert `overlap < size`; last chunk handled; whitespace-aware boundaries optional.
2. `DocumentSource` readers for `.txt`/`.md`/`.jsonl`; reject other extensions with a clear "v1 supports text/Markdown/JSONL only" error.
3. Implement `ingestData()` to walk a provided directory or record list, run the pipeline above, and emit `IngestionProgress`.
4. Use `VectorStore.insert` (#24) so ingestion is testable with `InMemoryVectorStore`.
5. Embedding dimension validation up front (must be one of the eight; #24).
6. Cooperative cancellation between documents/chunks (reuse the coroutine job pattern).

## Interactions

- **#24 (VectorStore)**: the storage sink; enables JVM tests without ObjectBox.
- **#26 (grounded generation)**: consumes the ingested chunks at query time.
- **`02_code_plans/04` (Hub pull)**: the embedding model/tokenizer are pulled into `<cacheDir>/<repo>/embedding/`.

## Tests & smokes

- **Chunker unit tests (JVM)**: exact boundaries for `size`/`overlap`; `overlap >= size` rejected; single-chunk and empty-text cases.
- **Source reader tests**: `.txt`/`.md`/`.jsonl` parse to the record shape; unsupported extension rejected.
- **Ingestion test with `InMemoryVectorStore`**: small fixture → expected chunk count inserted; `count()` matches.
- **Progress test**: callback sequence per document is correct.
- **Android smoke**: ingest one local text file → `VectorStore.count() > 0`.
