# RAG Config Surface & Grounded Generation

**Priority #26 | Prerequisites: #25 (`03_code_plans/04`), #23 (`03_code_plans/02`), #20 (`02_code_plans/04_hub_pull_and_cache_flow.md`) | Blocks: `05_code_plans/04` (RAG docs)**

> **Consumes `00_code_plans/09`.** `searchType` is the `SearchType` enum (09), not a `String`; the Python-side `RagConfig` is 09's Pydantic model (the `rag_config.json` contract Kotlin validates against). This plan also resolves `LLMRepository.kt:306` (`// TODO: Override RAG config if needed`) by making `prepareRetriever` apply the public `RagConfig` override through the typed mapper.

## Purpose

Expose a public `RagConfig` mapped onto the internal `ORTRagConfig`, plus one high-level, **inspectable** grounded-generation flow (retrieve → assemble prompt → generate → return text + retrieval metadata). Do not hide the prompt template in v1 — the user must be able to inspect or override prompt assembly (Tier-2 risk: hidden assembly makes results undebuggable).

## Touched / new files

Kotlin:
- NEW `android/.../config/RagConfig.kt` (public) — mapped into `ORTRagConfig` (`ORTRagConfig.kt:15-45`) via the existing `ORTRagArguments`/`overwriteWith` pattern (`:3-13, 30-44`).
- NEW `android/.../rag/PromptAssembler.kt` — default, overridable context-builder; emits the final prompt + the list of `RagMatch` used.
- `android/.../ORTRetriever.kt` — `retrieve(query, ragConfig)` returns `List<RagMatch>` (semantic `:77-114` or text `:116-128`).
- `android/.../repository/LLMRepository.kt` — `prepareRetriever` (`:306`, `// TODO: Override RAG config if needed`) applies the public `RagConfig` override via the typed mapper instead of the no-op TODO.
- `android/.../<facade>.kt` (from #18) — add `generateWithRag(query, ragConfig, generationConfig)` returning `GroundedResult`.

## Data contracts / interfaces

### Public `RagConfig` ↔ internal `ORTRagConfig`

| Public field | Internal (`ORTRagConfig`) |
| --- | --- |
| `embeddingRepoId` | `repoName` (`:17`) |
| `embeddingModelFile` | `onnxName` (`:18`) |
| `embeddingDimension` | `embeddingDimension` (`:19`) — must be one of the eight (#24) |
| `searchType` (`SearchType` enum: `semantic`/`text`; `hybrid` later) | `searchType` (`:21`) |
| `topK` | `topK` (`:20`) |
| `minScore` (NEW) | (new internal field; default `0.0`, used by `VectorStore.search`) |
| `chunkSize` / `chunkOverlap` | `:25` / `:26` |
| `maxTextLength` | `:24` |
| `indexingMode` (`precompute`/`dynamic`, NEW) | (new internal field) |
| `deviceConfig` | `deviceOptions` (`:28`) |

`similarityMetric` is fixed to ObjectBox COSINE for v1 (the only supported metric, #24) — expose it read-only, do not pretend it is configurable.

### Grounded generation flow

```kotlin
data class GroundedResult(
    val text: String,
    val matches: List<RagMatch>,        // what was retrieved (transparency)
    val prompt: String,                 // the exact assembled prompt (inspectable)
)

fun generateWithRag(query: String, rag: RagConfig, gen: GenerationConfig): GroundedResult {
    val matches = retriever.retrieve(query, rag.toInternal())
    val prompt  = promptAssembler.assemble(query, matches)   // default OR caller-supplied
    val text    = runtime.generate(prompt, gen.toInternal())
    return GroundedResult(text, matches, prompt)
}
```

## Implementation steps

1. Add public `RagConfig` (`searchType: SearchType`) + mapper reusing `ORTRagArguments.overwriteWith` (`ORTRagConfig.kt:30-44`); add the two new internal fields (`minScore`, `indexingMode`); wire the override into `LLMRepository.prepareRetriever` (`:306`).
2. Implement `PromptAssembler` with a sensible default template; allow a caller-supplied lambda/strategy.
3. `retrieve(...)` delegates to the existing semantic/text paths via `VectorStore` (#24).
4. `generateWithRag(...)` on the facade returns `GroundedResult` (text + matches + prompt).
5. Document the default template, override hook, and `precompute` vs `dynamic` indexing in `docs/RAG.md`.

## Interactions

- **#23**: reuses the public `GenerationConfig`.
- **#24 / #25**: retrieval and ingestion sit beneath this flow.
- **#20 (Hub pull)**: `embeddingRepoId` pulls the embedding model to `<cacheDir>/<repo>/embedding/`.

## Tests & smokes

- **Config mapping test (JVM)**: every public field maps; `minScore`/`indexingMode` defaults correct.
- **PromptAssembler test**: default template includes retrieved context + query; caller override replaces it; `GroundedResult.prompt` equals what was sent to `generate`.
- **Grounded flow test (with `InMemoryVectorStore` + fake runtime)**: matches feed the prompt; `GroundedResult` carries matches + prompt.
- **Android end-to-end smoke**: ingest one doc → `generateWithRag` → non-empty text, non-empty `matches`, inspectable `prompt`.
