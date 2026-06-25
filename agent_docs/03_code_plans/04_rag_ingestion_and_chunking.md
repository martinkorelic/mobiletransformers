# RAG Ingestion & Chunking

**Priority #26 | Prerequisites: #25 (`03_code_plans/03`), #23 (`03_code_plans/01`) | Blocks: #27 (`03_code_plans/05`)**

## Purpose

Implement the currently-empty `ORTRetriever.ingestData()` (`ORTRetriever.kt:172-175`, a double-TODO) so a user can do "bring your own documents" symmetrically with "bring your own dataset": local text/Markdown/JSONL → chunk → tokenize → embed → store → progress. Keep v1 scope strictly to plain text, Markdown, and JSONL — **no PDF/Word/HTML parsing** (explicit scope guard from the Tier-2 risks).

## Touched / new files

Kotlin:
- `android/.../ORTRetriever.kt` — implement `ingestData()` (`:172-175`); reuse the embedding session created in `createEmbeddingModel()` (`:26-63`) and `performEmbeddingStep(...)` (`:192-200`); store via `VectorStore.insert` (#25).
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

**Future-proofing (F3 — document-loader registry).** Readers live behind a `DOCUMENT_LOADER_REGISTRY` keyed by file extension (`txt`/`md`/`jsonl` now) rather than an `if ext == "txt"` chain, so a new format is one registry entry that yields the same `RagDocument` record shape — no edit to the chunk/embed/store pipeline. `pdf`/`html` loaders are explicitly **out of v1 scope** (they slot in as later registry entries); any extension not in the registry is rejected with the "v1 supports text/Markdown/JSONL only" error.

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
4. Use `VectorStore.insert` (#25) so ingestion is testable with `InMemoryVectorStore`.
5. Embedding dimension validation up front (must be one of the eight; #25).
6. Cooperative cancellation between documents/chunks (reuse the coroutine job pattern).

## Interactions

- **#25 (VectorStore)**: the storage sink; enables JVM tests without ObjectBox.
- **#27 (grounded generation)**: consumes the ingested chunks at query time.
- **`02_code_plans/04` (Hub pull)**: the embedding model/tokenizer are pulled into `<cacheDir>/<repo>/embedding/`.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Chunker unit tests (JVM)** `DocumentChunkerTest.kt`: exact boundaries for `size`/`overlap`; `overlap >= size` rejected; single-chunk and empty-text cases.
- **Source reader tests (JVM)**: `.txt`/`.md`/`.jsonl` parse to the record shape; an extension not in `DOCUMENT_LOADER_REGISTRY` is rejected with the "v1 supports text/Markdown/JSONL only" error.
- **Progress test (JVM)**: the `IngestionProgress` callback sequence per document is correct (`onDocumentStart` → N × `onChunkEmbedded` → `onDocumentComplete`).
- Module **compiles**: `./gradlew :MobileTransformers:compileDebugKotlin`.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Ingestion test with `InMemoryVectorStore`**: small `.txt`/`.jsonl` fixture → expected chunk count inserted; `count()` matches the hand-computed number of windows for the given `chunkSize`/`chunkOverlap`.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Android ingestion smoke**: ingest one local text file on device → `VectorStore.count() > 0` against the real ObjectBox store.

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- The double-TODO `ingestData()` (`ORTRetriever.kt:172-175`) is implemented: local `.txt`/`.md`/`.jsonl` → chunk → tokenize → embed → `VectorStore.insert` → `IngestionProgress`.
- `DocumentChunker` is deterministic and pure-Kotlin (asserts `overlap < size`, handles last/single/empty chunks) and JVM-testable.
- Readers sit behind `DOCUMENT_LOADER_REGISTRY` (F3); only txt/md/jsonl are accepted in v1, with PDF/Word/HTML explicitly out of scope and any other extension rejected with a clear error.
- Embedding dimension is validated up front (one of the eight, #25) and ingestion supports cooperative cancellation between documents/chunks.
