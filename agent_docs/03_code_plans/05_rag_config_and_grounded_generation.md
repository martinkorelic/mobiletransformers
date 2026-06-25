# RAG Config Surface & Grounded Generation

**Priority #27 | Prerequisites: #26 (`03_code_plans/04`), #24 (`03_code_plans/02`), #21 (`02_code_plans/04_hub_pull_and_cache_flow.md`) | Blocks: `05_code_plans/04` (RAG docs)**

> **Consumes `00_code_plans/09`.** `searchType` is the `SearchType` enum (09), not a `String`; the Python-side `RagConfig` is 09's Pydantic model (the `rag_config.json` contract Kotlin validates against). This plan also resolves `LLMRepository.kt:306` (`// TODO: Override RAG config if needed`) by making `prepareRetriever` apply the public `RagConfig` override through the typed mapper.

## Purpose

Expose a public `RagConfig` mapped onto the internal `ORTRagConfig`, plus one high-level, **inspectable** grounded-generation flow (retrieve → assemble prompt → generate → return text + retrieval metadata). Do not hide the prompt template in v1 — the user must be able to inspect or override prompt assembly (Tier-2 risk: hidden assembly makes results undebuggable).

## Touched / new files

Kotlin:
- NEW `android/.../config/RagConfig.kt` (public) — mapped into `ORTRagConfig` (`ORTRagConfig.kt:15-45`) via the existing `ORTRagArguments`/`overwriteWith` pattern (`:3-13, 30-44`).
- NEW `android/.../rag/PromptAssembler.kt` — default, overridable context-builder; emits the final prompt + the list of `RagMatch` used.
- `android/.../ORTRetriever.kt` — `retrieve(query, ragConfig)` returns `List<RagMatch>` (semantic `:77-114` or text `:116-128`).
- `android/.../repository/LLMRepository.kt` — `prepareRetriever` (`:306`, `// TODO: Override RAG config if needed`) applies the public `RagConfig` override via the typed mapper instead of the no-op TODO.
- `android/.../<facade>.kt` (from #19) — add `generateWithRag(query, ragConfig, generationConfig)` returning `GroundedResult`.

## Data contracts / interfaces

### Public `RagConfig` ↔ internal `ORTRagConfig`

| Public field | Internal (`ORTRagConfig`) |
| --- | --- |
| `embeddingRepoId` | `repoName` (`:17`) |
| `embeddingModelFile` | `onnxName` (`:18`) |
| `embeddingDimension` | `embeddingDimension` (`:19`) — must be one of the eight (#25) |
| `searchType` (`SearchType` enum: `semantic`/`text`; `hybrid` later) | `searchType` (`:21`) |
| `topK` | `topK` (`:20`) |
| `minScore` (NEW) | (new internal field; default `0.0`, used by `VectorStore.search`) |
| `chunkSize` / `chunkOverlap` | `:25` / `:26` |
| `maxTextLength` | `:24` |
| `indexingMode` (`precompute`/`dynamic`, NEW) | (new internal field) |
| `deviceConfig` | `deviceOptions` (`:28`) |

`similarityMetric` is fixed to ObjectBox COSINE for v1 (the only supported metric, #25) — expose it read-only, do not pretend it is configurable.

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
3. `retrieve(...)` delegates to the existing semantic/text paths via `VectorStore` (#25).
4. `generateWithRag(...)` on the facade returns `GroundedResult` (text + matches + prompt).
5. Document the default template, override hook, and `precompute` vs `dynamic` indexing in `docs/RAG.md`.

## Interactions

- **#24**: reuses the public `GenerationConfig`.
- **#25 / #26**: retrieval and ingestion sit beneath this flow.
- **#21 (Hub pull)**: `embeddingRepoId` pulls the embedding model to `<cacheDir>/<repo>/embedding/`.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- **Config mapping test (JVM)** `RagConfigMapperTest.kt`: every public field maps to its `ORTRagConfig` target; `minScore`/`indexingMode` defaults correct; `similarityMetric` is read-only COSINE.
- **PromptAssembler test (JVM)**: default template includes retrieved context + query; a caller override replaces it; `GroundedResult.prompt` equals what was sent to `generate`.
- Module **compiles**: `./gradlew :MobileTransformers:compileDebugKotlin`.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- **Grounded flow test (with `InMemoryVectorStore` + fake runtime)**: a small fixture of ingested chunks → `generateWithRag(...)` → assert the retrieved `matches` feed the prompt and `GroundedResult` carries the exact `matches` + assembled `prompt`.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- **Android end-to-end smoke**: ingest one doc → `generateWithRag` → non-empty text, non-empty `matches`, inspectable `prompt`.

**Workflow (end-to-end)** — the #27 checkpoint.
- Ingest a few local docs → retrieve top-k → grounded generation returns text + matches + the assembled prompt. The retrieve leg runs on the JVM with `InMemoryVectorStore` (fast, asserted); the generate leg runs against a real device-manual model. Pass = `GroundedResult` carries non-empty `text`, the `matches` that were retrieved, and the exact `prompt` that was sent to the runtime.

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- A public `RagConfig` maps onto `ORTRagConfig` via the `ORTRagArguments.overwriteWith` pattern, adding the new `minScore` and `indexingMode` internal fields; `searchType` is the `SearchType` enum and `similarityMetric` is exposed read-only (COSINE).
- `LLMRepository.prepareRetriever` (`:306`) applies the public override through the typed mapper instead of the no-op TODO.
- `generateWithRag(query, ragConfig, generationConfig)` returns a `GroundedResult(text, matches, prompt)`; the prompt template is the default-but-overridable `PromptAssembler`, never hidden.
