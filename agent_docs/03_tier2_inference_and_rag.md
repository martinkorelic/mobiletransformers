# Tier 2 - Inference And RAG As First-Class Subsystems

> Detailed code-implementation plans for these features live in `agent_docs/03_code_plans/` (and cross-referenced 00/01 plans); global order in `agent_docs/IMPLEMENTATION_ORDER.md`.
>
> **Config is typed + registry-driven.** All public config here uses the enums + Pydantic models from `00_code_plans/09` (e.g. `SamplingMethod`, `SearchType`, `GenerationConfig`, `RagConfig`); engine/method selection is registry-driven, not string-`if` dispatch.

## Purpose

Tier 2 turns existing inference and retrieval code into deliberate, documented subsystems. This is where MobileTransformers differentiates itself from training-only mobile fine-tuning projects: on the same device, a model can train, merge, generate, and optionally ground output with local retrieval.

## Current Repo Evidence

- `ORTGeneratorNative` implements the active Kotlin generation loop and delegates token-by-token inference to native C++.
- `inference.cpp` implements direct ONNX Runtime forward and KV-cache generation helpers.
- `sampling.cpp` and `SamplingOptions` expose greedy, top-k, and top-p style controls.
- `ORTGenAINative` and `onnx-genai.cpp` show a prior abandoned GenAI integration, so the new spike should reuse lessons from that failed path.
- `ORTGenerationConfig` includes `type = "native"`, `loadMergedWeights`, `sampling`, `maxSequenceLength`, metrics, and device options.
- `LLMRepository.prepareGeneration` currently routes only the `"native"` generation type; GenAI is commented out.
- `ORTRetriever` already loads an embedding tokenizer/model and supports semantic or text search.
- `ORTVectorDatabase` uses ObjectBox entities per supported embedding dimension and calls `nearestNeighbors`.
- `ORTRagConfig` exposes `repoName`, `onnxName`, `embeddingDimension`, `topK`, `searchType`, chunking fields, and device options.
- `ORTRetriever.ingestData()` is still a TODO, so "bring your own documents" is not yet complete.

## External Research Summary

- ONNX Runtime GenAI provides model, tokenizer, generator, parameters, and generation APIs that can replace custom generation loops when the model package format fits. Source: https://onnxruntime.ai/docs/genai/
- GenAI's value is not just convenience: it can reduce maintenance for KV cache, search/sampling, streaming decode, and structured/tool generation if it supports the merged-weight handoff. Source: https://onnxruntime.ai/docs/genai/
- ObjectBox documents on-device vector search and nearest-neighbor queries, matching the repo's current ObjectBox vector database implementation. Source: https://docs.objectbox.io/on-device-vector-search
- Hugging Face Hub download can support embedding model acquisition and ready-package distribution. Source: https://huggingface.co/docs/huggingface_hub/guides/download
- MobileFineTuner does not appear to provide the same inference/RAG stack, making this tier central to positioning. Source: https://arxiv.org/abs/2512.08211

## Recommended Decision

**The inference engine boundary is already owned upstream — Tier 2 consumes it, it does not redefine it.** Tier 0 / Tier 1 lock the `ModelRuntime` interface with two implementations (`ORTGeneratorNative` = default/guaranteed floor, `ORTGeneratorGenAI` = opt-in, gated on Gate 0.1) in `01_code_plans/03_inference_engine_abstraction_native_and_genai.md`. Both engines run over the **same package / same `inference/` folder**. Tier 2's job is therefore **hardening + RAG**, not inventing a parallel engine abstraction.

> **Do not introduce a new `InferenceEngine` interface.** Earlier drafts of this doc defined one; it is superseded by `ModelRuntime`. The engine selection, fallback, and streaming-parity contract live in `01_code_plans/03`. Tier 2 plans reference it.

The public API exposes one `generate()` path; the engine is auto-selected from the manifest. A `config.engine` flag exists only for advanced users/diagnostics.

For RAG, keep ObjectBox as the default vector store but introduce an interface around it so chunking, ingestion, and retrieval behavior are testable without Android ObjectBox.

## Inference Plan

### Engine Boundary (inherited, not redefined)

The engine interface is `ModelRuntime` from `01_code_plans/03_inference_engine_abstraction_native_and_genai.md`:

- `ORTGeneratorNative` implements `ModelRuntime` (Native, default floor).
- `ORTGeneratorGenAI` implements `ModelRuntime` (GenAI, opt-in; gated on Gate 0.1).
- `LLMRepository` / the facade select via `ModelRuntimeFactory.create(config, manifest)` with transparent Native fallback — never a direct `ORTGeneratorNative(...)` construction.
- Both engines load and validate `weight_handoff_map.json` (`00_code_plans/07`) before accepting merged weights and **fail closed** if a merged tensor cannot be matched to an external initializer (or GenAI graph input).

Tier 2 does not re-specify this; it (1) wires the native path into the boundary, (2) hardens it, and (3) keeps `GenerationCallback` / `InferenceProgress` parity (`LLMRepository.kt:40-47`, `ORTProgress.kt:18-27`) across engines. Detailed in `03_code_plans/01`.

### Native Path Hardening

The Native engine is hardened regardless of the GenAI decision (it is the guaranteed floor):

- Document supported input/output names and graph requirements from the real loop: `input_ids`, `attention_mask`, optional `position_ids`, past/present KV-cache shapes, and `logits` (`inference.cpp:12-59` forward; `inference.cpp:102-183` KV-cache generate; KV shapes from `session_cache.h:462-505`).
- Fix or document the current conversation-state TODO where a token from the previous assistant message can prepend.
- Add explicit session lifecycle tests for switching between training and generation.
- **Handoff is external initializers, not `inference/merged/`.** The stable contract (canonical per `IMPLEMENTATION_ORDER.md`, `01_code_plans/01`, `00_code_plans/07`) is: trained/merged weights are **flat per-tensor external initializers in `<cacheDir>/<model>/inference/`** loaded via `OrtSessionOptions::AddExternalInitializers` (`session_cache.h:662-709`), keyed by `weight_handoff_map.json`. The `loadMergedWeights` precondition becomes "handoff map present + all `external_file`s exist with valid checksums," **replacing** the legacy `inference/merged/` directory probe (`ORTGeneratorNative.kt:41-47`). Do not preserve the `inference/merged/` subdir as a contract.
- Add fail-closed error messages for unsupported models, missing/handoff-mismatched weights (wrong name/dtype/shape/quant role), and shape mismatch — before session creation.

### GenAI Path Adoption

If GenAI is adopted (per `01_code_plans/02` spike + `01_code_plans/03`):

- **Delete the dead `ORTGenAINative.kt` and `onnx-genai.cpp`**; implement `ORTGeneratorGenAI.kt` over a new `genai_runtime.cpp` (narrow wrapper around `OgaModel`, `OgaGeneratorParams`, `OgaGenerator`, `OgaTokenizer`, tokenizer stream). Do **not** revive the abandoned classes.
- The manifest declares engines via `supported_engines` / `default_engine` (owned by `00_code_plans/06`), not a bespoke `inferenceEngine` field.
- Both engines consume the **same `inference/` folder** (external initializers). Use `genai_config.json` `session_options.config_entries` (`session.model_external_initializers_file_folder_path`) for the external-data folder; rely on `weight_handoff_map.json` for the trained-tensor contract. Avoid `OgaNamedTensors` / `SetInputs` for decoder-only LLM handoff (upstream treats it as graph-input batching and rejects LLM/pipeline model types).
- Prove the engine can run before and after a train/merge cycle (the runtime expression of Gate 0.1 equivalence).
- Native remains the guaranteed fallback until GenAI passes all release smokes.

### Sampling And Streaming

Standardize config names:

- `sampling.method`: `greedy`, `top_k`, `top_p`
- `sampling.temperature`
- `sampling.topK`
- `sampling.topP`
- `sampling.seed`
- `maxNewTokens`, mapped from current `maxSequenceLength` in the public facade.

Callbacks should stay compatible:

- `onModelLoadStart`
- `onModelLoadEnd`
- `onStartGeneration`
- `onPartialResult`
- `onCompletion`
- `onError`

## RAG Plan

### Vector Store Boundary

Introduce:

```kotlin
interface VectorStore {
    fun insert(document: RagDocument, embedding: FloatArray): Long
    fun search(queryEmbedding: FloatArray, topK: Int, minScore: Double = 0.0): List<RagMatch>
    fun textSearch(query: String, topK: Int): List<RagMatch>
    fun count(): Long
    fun close()
}
```

Default implementation:

- `ObjectBoxVectorStore`, backed by the current `ORTVectorDatabase`.

Real ObjectBox semantics this wrapper must preserve and document (see `03_code_plans/03`):

- **COSINE distance** via `@HnswIndex(... distanceType = VectorDistanceType.COSINE)` per dimension (`entity/VectorEntity.kt:23-162`).
- **Score conversion:** `findWithScores()` returns a distance; the code maps it to similarity as `1 - result.score` (`ORTVectorDatabase.kt:262`), then a `minScore` filter is applied (`ORTVectorDatabase.kt:238`). This `1 - score` semantics must be documented and tested, not silently passed through.
- **Eight fixed dimensions only** — `64, 128, 256, 384, 512, 768, 1024, 1536`, one entity class per dimension (`VectorEntity.kt`). Arbitrary embedding sizes are unsupported; the wrapper must reject an unsupported `embeddingDimension` with a clear error.
- **Text search** is a separate ObjectBox VALUE-index path (`queryByContent`, `ORTVectorDatabase.kt:293`), not a vector query.

Testing implementation:

- `InMemoryVectorStore`, used for JVM/unit tests and ingestion tests (cosine similarity in pure Kotlin so behavior is testable without Android ObjectBox).

### RAG Config Surface

Public `RagConfig` should include:

- `embeddingRepoId`
- `embeddingModelFile`
- `embeddingDimension`
- `searchType`: `semantic`, `text`, `hybrid` later
- `topK`
- `minScore`
- `chunkSize`
- `chunkOverlap`
- `maxTextLength`
- `similarityMetric`, defaulting to ObjectBox-compatible nearest-neighbor behavior
- `indexingMode`: `precompute` or `dynamic`
- `deviceConfig`

Current `ORTRagConfig` can remain the internal representation while public config is mapped into it.

### Ingestion

Complete `ORTRetriever.ingestData()` (currently a TODO at `ORTRetriever.kt:172-175`) in stages (detailed in `03_code_plans/04`):

1. Accept local text, Markdown, JSONL, and plain document records.
2. Split text by `chunkSize` and `chunkOverlap` (`ORTRagConfig.kt:25-26`).
3. Tokenize chunks with the embedding tokenizer.
4. Generate embeddings through the embedding ONNX session (`performEmbeddingStep`, `ORTRetriever.kt:192`).
5. Store chunks with metadata in ObjectBox (dimension-routed entity).
6. Emit progress callbacks.

Document ingestion input shape:

```json
{
  "id": "doc-001",
  "title": "Personal notes",
  "text": "Full document text",
  "metadata": {
    "source": "local"
  }
}
```

### Grounded Generation Flow

Expose one high-level flow:

1. `model.retrieve(query, ragConfig)`
2. Build prompt context from returned documents.
3. Call `model.generate(promptWithContext, generationConfig)`
4. Return generated text plus retrieval metadata.

Do not force a hidden prompt template in v1. The user should be able to inspect or override the RAG prompt assembly.

## HF API Alignment Audit

Create a mapping table in docs:

| HuggingFace concept | MobileTransformers public name | Current internal name |
| --- | --- | --- |
| `from_pretrained` | `fromPretrained` | `LLMRepository(initialModel)` plus cache lookup |
| `GenerationConfig.max_new_tokens` | `GenerationConfig.maxNewTokens` | `ORTGenerationConfig.maxSequenceLength` |
| `temperature`, `top_k`, `top_p` | `SamplingConfig` | `SamplingOptions` |
| `Trainer.train` | `model.train` | `TrainingRepository.performTraining` |
| PEFT config | `PeftConfig` | `train_method`, `lora_rank`, `mars.optimization_level` |
| dataset | `DatasetConfig` or local dataset object | `DatasetOptions` and task preprocessors |

## Implementation Sequence

Prerequisites before Tier 2 implementation starts:

- Tier 0 has chosen the supported train -> merge -> infer handoff path.
- Tier 1 has a manifest/cache validator and public facade shape, even if direct Hub download is not finished.
- Public `GenerationConfig` and `RagConfig` names are stable enough to document.

Sequence:

1. Adopt `ModelRuntime` (from `01_code_plans/03`) and wrap current native inference behind it without changing behavior; retire the legacy `inference/merged/` probe in favor of the handoff-map precondition.
2. Add tests around the runtime wrapper, repository-backed selection/fallback, and facade routing.
3. Reflect the Tier 0 GenAI/manual decision in package manifest `supported_engines` / `default_engine`.
4. If GenAI wins, complete `ORTGeneratorGenAI` (per `01_code_plans/03`); otherwise document Native as intentional and harden it.
5. Keep the non-default engine behind diagnostics/experimental flags until train/merge/generate smokes pass.
6. Define `VectorStore` and wrap `ORTVectorDatabase`.
7. Add `InMemoryVectorStore` for JVM/unit tests before touching ObjectBox behavior.
8. Implement document ingestion and chunking for text, Markdown, and JSONL only.
9. Add RAG prompt assembly and grounded generation helper through the public facade.
10. Publish `RAG.md` and HF alignment mapping after the public config names are confirmed.

## Risks

- GenAI adoption may break the merged-weight handoff.
- Native inference cleanup may accidentally change token streaming behavior.
- ObjectBox entity classes per embedding dimension make arbitrary embedding sizes awkward.
- Retrieval scoring currently converts ObjectBox score with `1 - result.score`; the semantics should be documented and tested.
- RAG ingestion can become a scope sink if PDF/Word/HTML parsing is attempted in v1. Keep v1 to plain text, Markdown, and JSONL.
- Hidden prompt assembly can make results hard to debug.

## Tests And Smokes

- Native inference wrapper unit test using a fake engine or small fixture.
- Android smoke: load current native inference and generate one token.
- Conversation reset test.
- Merged weights missing-path test for `loadMergedWeights`.
- Handoff-map mismatch test: wrong name, dtype, shape, or quantization metadata must fail before session creation.
- Vector store insert/search/count tests with `InMemoryVectorStore`.
- ObjectBox semantic query test for supported dimensions.
- RAG ingestion test with a small local text fixture.
- End-to-end Android smoke: ingest one document -> query -> generate with retrieved context.
- If GenAI wins: GenAI one-token smoke before and after merge.

## Acceptance Criteria

- Inference has a documented engine boundary.
- The chosen GenAI/manual decision from Tier 0 is reflected in implementation docs.
- Native inference remains available as fallback until GenAI is proven or rejected.
- RAG has a public config, vector store boundary, ingestion path, and on-device example.
- ObjectBox remains the default vector database.
- HF API mapping is documented.

## Source Links

- ONNX Runtime GenAI: https://onnxruntime.ai/docs/genai/
- ONNX Runtime GenAI config reference: https://onnxruntime.ai/docs/genai/reference/config.html
- ObjectBox vector search: https://docs.objectbox.io/on-device-vector-search
- Hugging Face Hub download guide: https://huggingface.co/docs/huggingface_hub/guides/download
- Hugging Face Hub file download API: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- MobileFineTuner: https://arxiv.org/abs/2512.08211
