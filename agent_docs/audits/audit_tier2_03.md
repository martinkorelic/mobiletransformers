# Tier 2 — 03_code_plans audit

> ## ⚠️ SNAPSHOT — 2026-08-07, at HEAD `54e0a8e`. NOT a live defect list. (Banner added 2026-08-14.)
>
> This audit is a **point-in-time photograph**, and it carries **no closure annotations of its own** —
> nothing in this file was ever struck out as findings were fixed. Six later cycles of work landed on
> top of it (the 2026-08-07 remediation pass, then 08-08 / 08-09 / 08-10 device acceptance, then the
> 08-14 cleaning phase). **Reading it as a to-do list generates phantom work**, which is the specific
> failure this banner exists to prevent.
>
> Spot-verification on 2026-08-14 found the audits materially **over-report** what is open. Every one
> of these, recorded here as a defect, is fixed in the tree with the fix documented at the site:
>
> | Audit finding | Where it is fixed |
> | --- | --- |
> | #21 "installer deletes live cache before rename" | `ModelPackageInstaller.kt:47-75` — renames aside, publishes, rolls back on failure ("#21 crash safety") |
> | #17 "`ORT*` leak in public `TrainingResult`" | `runtime/Results.kt:25-26` records the retype |
> | #27 "config override applies only on the FIRST retrieve" | `RagRepository.kt:36` — "A changed config now always applies" |
> | #24 "GenAI carries its own private method map with silent-greedy fallback" | `ORTGeneratorGenAI.kt:78` uses the shared `SamplingMethod.fromWire(...).nativeOrdinal` |
> | #26 "`maxTextLength` silently dropped" | threaded through `ConfigMappers.kt:135` / `ORTRagConfig.kt:45` |
> | #34 "`ORTScheduler.kt` TODO still open" | fixed; `ORTScheduler.kt:161-162` records it |
> | #25 "`SearchType` String→enum swap never landed" | done 2026-08-07 |
> | #6 "grep-guard DoD fails, `build_adapter_mapping` missing" | done 2026-08-07 (see #6's self-check) |
> | #22 "Mode-1 never writes `adapter_model.safetensors`" | fixed; see #22's self-check |
> | #15 "`--validate` missing entirely" | it exists |
>
> **The authoritative list of what is actually open is `agent_docs/HANDOFF.md`**, whose numbers are
> re-measured each cycle. Use this file for its *reasoning* — why a finding mattered, what the failure
> shape was — not for its verdicts.


Scope: `agent_docs/03_code_plans/01..05` = global plans #23–#27. Verified against the tree at
`android/MobileTransformersApp/MobileTransformers/src/` (branch `restructure`, working tree clean).
All paths below are relative to
`/home/martin/Documents/Projects/Development/LLM_finetuning/mobiletransformers/`.
Abbreviation used for the Kotlin package root:
`AND = android/MobileTransformersApp/MobileTransformers/src/main/java/com/martinkorelic/mobiletransformers`,
`TST = android/MobileTransformersApp/MobileTransformers/src/test/java/com/martinkorelic/mobiletransformers`,
`CPP = android/MobileTransformersApp/MobileTransformers/src/main/cpp`.

## Summary table

| # | Plan | Claimed | Verified | % | Verdict |
| --- | --- | --- | --- | --- | --- |
| 23 | `01_inference_handoff_alignment_and_native_hardening` | code-complete, box open (device) | Kotlin+C++ map-driven fail-closed load real; conv-state fix real; **graph-I/O contract undocumented, no missing-required-input gate, C++ dtype/shape failure silently downgrades, one dead `NotImplementedError` stub survives** | **75%** | Mostly real, 3 normative DoD items unmet |
| 24 | `02_sampling_and_streaming_public_config` | code-complete, box open (parity device leg) | Native `methodMap` genuinely retired; `maxNewTokens` locked + tested. **GenAI engine still carries its own private method map with silent-greedy fallback; the two engines' loop bounds and completion payloads differ → parity contract is NOT held** | **65%** | Claim of parity is unsupported by code |
| 25 | `03_vector_store_boundary_and_inmemory` | **done** (box `[x]`) | Boundary, registries, InMemory store, 23 JVM tests all real and good | **90%** | Justifiably done; 2 stale TODOs + device smoke |
| 26 | `04_rag_ingestion_and_chunking` | code-complete, self-check boxes all `[ ]` | Chunker/loader-registry/pipeline/progress/`ingestData`/`RagRepository.ingest`/facade all real + tested. **`maxTextLength` silently dropped; no directory walk; PDF/Word not documented anywhere user-visible** | **80%** | Bookkeeping *correctly* open (doc item genuinely fails) |
| 27 | `05_rag_config_and_grounded_generation` | code-complete, self-check boxes all `[ ]` | Public `RagConfig`, `minScore`/`indexingMode`, `PromptAssembler`, `GroundedResult`, `generateWithRag` all real. **Config override applies only on the FIRST retrieve of a session (minScore/topK silently dropped after); Python `RagConfig` never extended; `GroundedFlowTest` never calls `generateWithRag`; internal `searchType` still `String`** | **75%** | Real flow, real override bug |

**Tier estimate: ~77% complete.** Everything the plans named exists as a symbol; the deficits are
(i) two fail-closed gates that log instead of raising, (ii) an engine-parity claim the GenAI code
contradicts, (iii) three silently-dropped config fields, (iv) doc pages left describing the pre-#26/#27
world.

On the "#26/#27 boxes unchecked while HANDOFF says code-complete" question: **this is not purely stale
bookkeeping.** #26's third self-check ("Is PDF/Word explicitly out of v1 scope and *documented*?") is a
genuine **no** — `docs/RAG.md:57-64` still says ingestion and grounded generation "are not yet
implemented". #27's third ("Does the ingest → retrieve → grounded-generate workflow pass?") is a genuine
**no** (device-only leg). The other four questions are all satisfiable from code today.

---

## Per-plan findings

### #23 — Inference handoff alignment & native path hardening

**Required:** `ORTGeneratorNative` implements `ModelRuntime` and is factory-constructed; map-driven,
fail-closed merged-weight load (map + every `externalDataLocation[role]` file + checksum + dtype/shape),
raising **before** `createInferenceSession` and naming the tensor; zero `inference/merged/` references;
dead GenAI stubs deleted; conversation-prepend bug fixed + `resetConversation()`; documented native graph
I/O contract; fail-closed errors for unsupported models (missing required input); JVM handoff-precondition
tests, grep regressions, and a **JVM fixture** test asserting names come from `inferenceInitializerNames`.

**Verified present:**
- `AND/ORTGeneratorNative.kt:12` implements `ModelRuntime`; `:33-39` `load()` calls `resetConversation()`
  then `createInferenceModel()`; `:282-285` `resetConversation()`; `:22-28` `capabilities` uses the
  non-throwing presence query.
- `AND/internal/runtime/HandoffPrecondition.kt:35-73` — full gate: schema `checkCompat`, per-role `.bin`
  existence (`:52-56`), sha256 from map or `<bin>.sha256` sidecar (`:59-64`), mismatch throws naming the
  tensor (`:65-69`). Absent map → `false` (documented "nothing merged" case), consumed at
  `ORTGeneratorNative.kt:78-81`.
- `AND/packages/WeightHandoffMap.kt:18-51` Gson read model with `inferenceInitializerNames`/`sha256`.
- `CPP/handoff_io.h:60-118` `load_handoff_entries` + `check_compat` — the single on-device reader, used by
  both `weight_merger.cpp` and `session_cache.h`.
- `CPP/session_cache.h:64-141` `WeightSessionCache::init` — names taken from
  `entry.inferenceInitializerNames.at(role)` (`:81-82`), no `<dirname>.<filestem>` reconstruction;
  `validate_dtype_shape` (`:154-173`); `clearWeights(); return false` on any failure (no partial set).
- Factory-only construction: the only `ORTGeneratorNative(` / `ORTGeneratorGenAI(` call sites are
  `AND/runtime/ModelRuntime.kt:129` and `:134` (`ModelRuntimeFactory.create`). `LLMRepository.kt:275`
  routes through it.
- Conversation fix: `AND/ORTConversationState.kt:45-60` — rendered-offset advance (`indexOf(content)`
  inside the re-rendered history) instead of `content.length`.
- Zero live `inference/merged/` path construction anywhere (only comments/docs mention the retired name);
  `ORTGenAINative.kt` and `onnx-genai.cpp` are absent from the tree.
- Tests: `TST/internal/runtime/HandoffPreconditionTest.kt` (11 cases incl. missing bin, checksum mismatch,
  no-checksum-source, incompatible major, invalid JSON, presence-query non-throwing);
  `TST/runtime/NativeLoadRegressionTest.kt:28-53` (three grep guards); `TST/ORTConversationStateTest.kt`.
- Device legs exist as instrumented tests: `androidTest/.../ConversationResetTest.kt`,
  `TrainMergeGenerateTest.kt`, `FacadeLoadGenerateTest.kt`.

**Gaps:**
- **(a) genuinely missing — the C++ leg is not fail-closed to the caller.** `CPP/session_cache.h:740-744`:
  when `weight_session->init(...)` returns false (missing file, dtype mismatch, shape mismatch — the exact
  cases the plan says must raise), the code only `LOGE`s and **continues to `createInferenceSession` with
  base weights**. dtype/shape validation is C++-only (the Kotlin gate cannot do it), so a shape-mismatched
  merged tensor is a **silent downgrade to base weights**, directly contradicting "raise before
  `createInferenceSession`, with the offending tensor name" (plan `Data contracts`, `Implementation steps`
  6, and the DoD).
- **(a) genuinely missing — no unsupported-model gate.** `CPP/session_cache.h:499-524` (ctor) creates the
  `Ort::Session` first and only then calls `generateInputOutputNames()` (`:840-882`), which merely records
  names and sets `has_position_ids`. There is no check that `input_ids` / `attention_mask` / `logits`
  exist, and no error before session creation. Implementation step 6 ("unsupported model (missing required
  input) … all before session creation") is unimplemented.
- **(a) genuinely missing — the graph I/O contract is documented nowhere.** Plan deliverable (c). No
  header block in `CPP/inference.cpp` (file starts straight at `namespace inference` / hardcoded
  `{"input_ids","attention_mask","position_ids"}` at `:19-22`), and `grep -rn "input_ids|position_ids|
  past_key_values|logits" docs/*.md` returns **zero hits**. `docs/ARCHITECTURE.md` does not exist.
  (Page ownership is #31, but the contract text this plan was to produce does not exist in any form.)
- **(a) residual dead stub.** `AND/ORTGenAITokenizer.kt:33-42` is still a live class whose four methods
  `throw NotImplementedError("ORTGenAITokenizer is retired (see #11)")`, and it is still referenced by a
  never-assigned field `LLMRepository.kt:147` (+ import `:6`). The DoD says the dead `NotImplementedError`
  GenAI stubs are "deleted, not left half-implemented"; `NativeLoadRegressionTest` only greps for the two
  *deleted files*, so this class slipped past the regression guard.
- **(b) implemented-but-untested — the prepend fix.** `ORTConversationStateTest` constructs
  `ORTConversationState(null, …)` (null template handler), so `renderHistory()` returns `""` and the
  rendered-offset branch at `ORTConversationState.kt:52-59` is **never exercised**; the test only covers
  `resetForNewConversation()` bookkeeping and the system-prompt path. The self-check claim
  "`ORTConversationStateTest` covers reset" is true but the *bug fix itself* has no host coverage.
- **(b) missing integration test.** The plan's "Map-driven load assertion (JVM, fixture)" — a fixture map
  + flat files asserting names come from `inferenceInitializerNames[role]` — does not exist.
  `NativeLoadRegressionTest.handoffLoadIsMapDriven` (`:40-43`) only asserts the string
  `"load_handoff_entries"` appears in `session_cache.h`. That is a grep, not the required assertion.
- **(c) device-only, correctly outstanding:** map-driven load-and-generate over a real #9 package,
  two-prompt no-leak, train↔generate lifecycle (`ConversationResetTest`, `TrainMergeGenerateTest` are
  written and skip without a pushed package).

**Drift / doubtful claims:**
- `ORTGeneratorNative.kt:130-131` still carries the literal
  `// NOTE: Sometimes one token from the previous assistant message keeps prepending` /
  `// TODO: Will need fix`. The fix landed elsewhere (`ORTConversationState`), but the marker the plan
  cites as the bug site was never cleared — a reader greps `TODO` and concludes it is open.
- IMPLEMENTATION_ORDER #23 self-check box 1 is `[x]` for "**fail-closed** external-initializer load …
  *both* fail closed naming the tensor". The Kotlin side does; the C++ side logs and proceeds (above).
  The claim is half-supported.
- `ORTGeneratorNative.updateSamplingOptions` is invoked from the `generationConfig` setter (`:53`) which
  can run before `createInferenceModel`, i.e. with `inferenceModel == 0`. Pre-existing, not introduced
  here, but it means `setSamplingConfig(0, …)` is reachable.

**% complete: 75%** — the map/checksum/naming spine is genuinely built and tested; three normative
DoD/step items (C++ raise-before-session, missing-input gate, documented I/O contract) plus one residual
dead stub are unmet.

---

### #24 — Sampling & streaming public config alignment

**Required:** public `SamplingConfig`/`GenerationConfig` with HF names, `method: SamplingMethod`, defaults
byte-equal to internal; `methodMap` magic replaced by `SamplingMethod.fromWire(...).nativeOrdinal`,
fail-closed on unknown; `maxNewTokens → maxSequenceLength` locked with documented new-token semantics;
**both engines emit the identical `GenerationCallback`/`InferenceProgress` sequence — one shared
interface, no per-engine forking**; JVM mapping/default/enum tests + a `maxNewTokens`-semantics
integration test.

**Verified present:**
- `AND/config/PublicConfigs.kt:32-38` `SamplingConfig(method: SamplingMethod = GREEDY, temperature=1f,
  topK=10, topP=0.9f, seed=42)` — defaults match `AND/ORTGenerationConfig.kt:3-9` exactly.
  `:58-64` `GenerationConfig(maxNewTokens = 128, …)` matches `ORTGenerationConfig.maxSequenceLength = 128`.
- `AND/constants/SamplingMethod.kt:17-23` `nativeOrdinal` = 0/1/2, matching `CPP/sampling.h:16-20`
  (`GREEDY=0, TOP_K=1, TOP_P=2` — verified). `fromWire` errors on unknown (`:26-27`).
- `AND/ORTGeneratorNative.kt:305` `SamplingMethod.fromWire(args.method).nativeOrdinal` — the old
  `methodMap = mapOf(...)` is gone from this file.
- `AND/internal/config/ConfigMappers.kt:35-42` (`SamplingConfig.toOrt`), `:83-94`
  (`GenerationConfig.toOrt`, `maxSequenceLength = maxNewTokens`).
- `AND/runtime/ModelRuntime.kt:19-33` is a single shared `ModelRuntime`; `AND/repository/LLMRepository.kt:
  40-47` is a single `GenerationCallback`; `AND/ORTProgress.kt:18-27` a single `InferenceProgress` — no
  per-engine subclassing anywhere.
- Tests: `TST/facade/SamplingMappingTest.kt` (5: ordinals, round-trip, fail-closed on `"beam"`,
  wire mapping, `maxNewTokens→maxSequenceLength`); `TST/facade/ConfigMapperTest.kt:27`
  `assertEquals(ORTGenerationConfig(), GenerationConfig().toOrt())` (the byte-equal default test).
- Enum parity: `src/mobiletransformers/config/constants.py:18-21` and `schemas/enums.json`
  `"SamplingMethod": ["greedy","top_k","top_p"]` agree with the Kotlin mirror.

**Gaps:**
- **(a) genuinely missing — the `methodMap` magic was only half-retired.**
  `AND/ORTGeneratorGenAI.kt:118-123`:
  ```kotlin
  private fun samplingMethodInt(method: String): Int =
      when (method) { "top_k" -> 1; "top_p" -> 2; else -> 0 /* greedy */ }
  ```
  This is exactly the pattern the plan names ("the `methodMap = mapOf("greedy" to 0, …)` magic") and it
  **silently defaults to greedy on an unknown method** — the fail-closed requirement is violated on the
  GenAI engine. Used at `ORTGeneratorGenAI.kt:45`.
- **(a) parity contract broken — loop bound.** Native: `while (decoded <= generationArgs
  .maxSequenceLength)` (`ORTGeneratorNative.kt:174`, inclusive). GenAI: `while (decoded <
  generationArgs.maxSequenceLength && …)` (`ORTGeneratorGenAI.kt:73`, exclusive). For the same public
  `maxNewTokens` the two engines emit a different number of `onPartialResult` events (off by one). This
  is both a parity break and an unresolved instance of plan step 3 ("confirm `maxNewTokens →
  maxSequenceLength` … adjust the mapping if the internal value is total-length rather than new-tokens").
  The code itself admits it is unresolved: `ORTGeneratorNative.kt:172-173` — "The exact stop-count of this
  inclusive bound is pinned by the device generation smoke."
- **(a) parity contract broken — payload fields.** The plan lists all eight `InferenceProgress` fields as
  "populated by both engines". Divergences found:
  - `onCompletion.avgTokensPerSecond`: Native carries the running value (`ORTGeneratorNative.kt:271`),
    GenAI hardcodes `0.0` (`ORTGeneratorGenAI.kt:100`).
  - `onStartGeneration.isCompleted`: Native = `isEosToken(inputIds.last())` (`:166`), GenAI = `false`
    (`:69`).
  - `prefillTimeMs` / `generationTimeMs` / `avgTokensPerSecond`: Native only populates them when
    `generationArgs.trackMetrics` (`:186-211`); GenAI always populates (`:75-80`). Same config → different
    payloads.
- **(b) missing integration test.** The plan's "`maxNewTokens` semantics check" (drive the loop bound
  against a fixture and assert the stop count) does not exist in `TST/`; it was deferred to a device smoke
  by comment. Given the `<=`/`<` split above this is the test that would have caught the parity break.
- **(d) documented deferral, partially:** plan step 5 says update the HF mapping table in
  `03_tier2_inference_and_rag.md` **and** `docs/PUBLIC_API.md`. The tier doc has its table; `docs/
  PUBLIC_API.md` is Python/CLI-only (`grep maxNewTokens docs/PUBLIC_API.md` → no hits) and has no Kotlin
  facade section. HANDOFF/IMPLEMENTATION_ORDER note `docs/ANDROID_SDK.md` "awaits #24" — consistent.
- **(c) device-only:** the cross-engine ordered-event assertion. `androidTest/.../DualEngineParityTest.kt`
  exists but asserts only that the greedy **first token text** matches — it does **not** assert the
  callback *sequence* or the `InferenceProgress` payloads, so even the device leg as written would not
  detect the three divergences above.

**Drift / doubtful claims:**
- IMPLEMENTATION_ORDER #24: "`ORTGeneratorNative.updateSamplingOptions` dropped the `methodMap` magic …
  (fail-closed on unknown, **no silent greedy**)" — true for Native, false for the codebase: GenAI's
  `samplingMethodInt` is a silent-greedy map.
- IMPLEMENTATION_ORDER #24 self-check: "the public callback surface **and Native sequence** are in place;
  the Native-vs-GenAI ordered-event assertion needs a real #9 package" — this framing implies only the
  *proof* is missing. In fact the *implementation* diverges (loop bound, payload fields) in ways a host
  test could have caught without a device.

**% complete: 65%** — the Native-side rename/enum/lock work is complete and tested; the plan's headline
deliverable (cross-engine parity, no per-engine magic) is contradicted by `ORTGeneratorGenAI`.

---

### #25 — Vector store boundary & in-memory implementation

**Required:** `VectorStore` + `RagDocument`/`RagMatch`; `ObjectBoxVectorStore`; test-only
`InMemoryVectorStore`; single declared dimension registry with fail-closed unsupported dims; `VectorStore`
factory/registry (F4, `objectbox` default); `ORTRetriever` routed through the boundary; ObjectBox
semantics preserved (COSINE, `1-score`, `minScore`, embeddings stripped, text path separate) and tested.

**Verified present:**
- `AND/rag/VectorStore.kt:17-46` — `RagDocument`, `RagMatch`, `TEXT_SEARCH_SCORE = 1.0`, the exact
  five-method interface from the plan (with `minScore: Double = 0.0` default).
- `AND/rag/ObjectBoxVectorStore.kt:12-38` — delegates to `queryDocuments`/`queryByContent`/`getVectorCount`;
  `init` block (`:14-16`) calls `DimensionRegistry.requireSupported`.
- `AND/rag/VectorStoreRegistry.kt:15-38` `DimensionRegistry` (`{64,128,256,384,512,768,1024,1536}`,
  `register`, `requireSupported` with a clear message); `:52-73` `VectorStoreRegistry` with
  `DEFAULT_KEY = "objectbox"` and a throw on unknown key.
- `AND/ORTVectorDatabase.kt:215-217` — `SUPPORTED_DIMENSIONS` now *delegates* to `DimensionRegistry`
  (single declared source, as claimed).
- Semantics preserved: `ORTVectorDatabase.kt:288-295` `Pair(entity, 1 - result.score)` and
  `entity.embedding = floatArrayOf()`; `:238` `.filter { it.second >= minScore }`.
- `ORTRetriever.kt:72` `vectorStore()` and `:113`/`:129` route search + text search through the boundary.
- Tests: `TST/rag/InMemoryVectorStoreTest.kt` (16 assertions across insert/count, wrong-dimension reject,
  cosine ordering with hand-computed 0.8/0.6, topK, `minScore` on similarity with identical→1.0 and
  orthogonal→0.0, no-embedding-in-result, text search fixed score, dim-300 fail-closed, `register(301)`
  accepted, registry unknown-key throw). `TST/rag/InMemoryVectorStore.kt` is the pure-Kotlin store.
- `docs/RAG.md:12-56` documents the boundary, the `1-distance` semantics, `minScore`, the text path, and
  the dimension registry — the #25-scoped page really exists.

**Gaps:**
- **(b) minor, stale markers.** `AND/ORTVectorDatabase.kt:229` still reads `// TODO: Clear embeddings from
  returning` even though the clearing is implemented 60 lines below (`:293`). `AND/entity/
  VectorEntity.kt:164` still reads `// TODO: Could add other popular dimensions...` even though the plan
  says the registry replaces it. Both are the plan's named TODO sites — behaviour done, marker not.
- **(b) text path does not strip embeddings.** `queryByContent` (`:293-315`) returns whole entities with
  their `embedding` arrays intact (no `embedding = floatArrayOf()`), unlike `searchVectors`. The plan's
  "No embeddings in results" bullet is satisfied at the `RagMatch` level only because `RagDocument` has no
  embedding field — the memory motivation is not met for the text path.
- **(b) registered-but-unbacked dimension is not fail-closed.** `DimensionRegistry.register(dim)` accepts
  any positive int, after which `ORTVectorDatabase.queryDocuments`'s `else -> emptyList()` (`:246`) and
  `getAllVectors`'s `else -> emptyList()` return **silently empty** rather than throwing — the exact
  "never a silent box pick" behaviour the plan forbids, reachable via the extension path the plan
  advertises.
- **(b) lossy metadata round-trip.** `ObjectBoxVectorStore.kt:50-51` encodes `Map<String,String>` as
  `k=v;k=v`, `:42-48` decodes it back as `mapOf("metadata" to <whole string>)`. Insert→search does not
  round-trip `RagDocument.metadata`.
- **(c) device-only:** ObjectBox parity smoke (supported dims) — correctly deferred and never claimed done.
- **(d) documented deferral that never closed:** "`SearchType` String→enum swap in `ORTRagConfig` rides
  with the facade plans (#17/#19)" — see #27 below; it did **not** land.

**Drift / doubtful claims:** none material. This is the one plan whose `[x]` is well supported.
Minor: `ORTRetriever.vectorStore()` allocates a fresh `ObjectBoxVectorStore` wrapper on every call
(`:72`), which re-runs `requireSupported` per query — harmless, but it bypasses `VectorStoreRegistry`
(the F4 entry point is never used in production code, only in tests).

**% complete: 90%.**

---

### #26 — RAG ingestion & chunking

**Required:** implement the double-TODO `ingestData()` (txt/md/jsonl → chunk → tokenize → embed →
`VectorStore.insert` → `IngestionProgress`); `DocumentChunker` (pure, character-based, `overlap < size`,
last/single/empty handled); `DocumentSource` behind `DOCUMENT_LOADER_REGISTRY` (F3) rejecting anything
else with "v1 supports text/Markdown/JSONL only"; `RagRepository.ingest(documents, progress)` with
cooperative cancellation; embedding-dimension validation up front; pipeline honours `maxTextLength`;
PDF/Word explicitly out of scope **and documented**.

**Verified present:**
- `AND/ORTRetriever.kt:188-223` — `ingestData(documents, progress)` is implemented; `:189`
  `DimensionRegistry.requireSupported(...)` up front; `:201-221` binds the real tokenizer +
  `performEmbeddingStep` as the injected `embed` lambda. **The double-TODO is gone** (`grep TODO
  ORTRetriever.kt` → no hits).
- `AND/rag/DocumentChunker.kt:10-28` — `require(chunkSize > 0)`, `require(chunkOverlap in 0 until
  chunkSize)`, empty→`emptyList`, short→single chunk, stride `size - overlap`, last window clamped.
- `AND/rag/DocumentSource.kt:39-44` `DOCUMENT_LOADER_REGISTRY` (`txt`/`md`/`jsonl`); `:47-55`
  `loadDocuments` throws `"v1 supports text/Markdown/JSONL only, got '.$ext'"`; extension lowercased.
- `AND/rag/IngestionPipeline.kt:14-47` — pure loop, `coroutineContext.ensureActive()` between documents
  (`:24`) and between chunks (`:29`), `CancellationException` rethrown (`:40-41`), per-document errors to
  `onError` (`:43`), chunk ids `"${record.id}#$i"` (`:33`) exactly per the plan's pipeline sketch.
- `AND/rag/IngestionProgress.kt:7-15` — the four-method interface verbatim from the plan.
- `AND/repository/RagRepository.kt:17-29` `ingest(path, ragConfig, progress)` (owner is RagRepository, no
  parallel entry point on `LLMRepository` — checked); facade `AND/MobileTransformerModel.kt:64-68` and
  `AND/runtime/ModelSession.kt:37` expose `ingest`.
- Tests: `TST/rag/DocumentChunkerTest.kt` (6: exact windows 0-40/30-70/60-100, single, empty,
  `overlap>=size` rejected, size 0 rejected, exact tiling); `TST/rag/DocumentSourceTest.kt` (txt/md/jsonl
  record shape, metadata parse, **`.pdf` and `.docx` both rejected with the exact message**, case-insens.);
  `TST/rag/IngestionPipelineTest.kt` (chunk count 3 for 100 chars @40/10, exact progress event sequence
  `start → chunk 0/2 → chunk 1/2 → done`, embed-failure path, multi-doc).
- Device leg written: `androidTest/.../RagDeviceTest.kt:24-50`.

**Gaps:**
- **(a) silently dropped deliverable — `maxTextLength` is never applied.** The plan's normative pipeline
  says `text = readAndNormalize(record)   # honor maxTextLength`. `grep -rn maxTextLength AND/` returns
  only: the config field (`ORTRagConfig.kt:11,28,45`), the mapper (`ConfigMappers.kt:109`), and the JSON
  parser (`FileUtil.kt:264`). No reader in `ORTRetriever`, `IngestionPipeline`, or `DocumentSource`. The
  field is a **defined-but-never-consumed surface** — the RAG analogue of the `TrainingEvent.Metric`
  pattern the audit asked about (see also `indexingMode`, below).
- **(a) partial — no directory ingestion.** Implementation step 3: "walk a provided **directory** or record
  list". `loadDocuments(path)` hard-requires `file.isFile` (`DocumentSource.kt:49`) and
  `RagRepository.ingest` takes a single `path`. Ingesting a folder of notes — the "bring your own
  documents" story in the Purpose — requires the caller to loop.
- **(a) documentation gap — PDF/Word scope is not documented.** Self-check #26 item 3. The rejection
  exists in code + error string + a unit test, but `docs/RAG.md:57-64` still lists ingestion under
  "**Not yet (tracked)** … #26" and no page states the v1 format scope. This is the concrete reason the
  #26 self-check should stay unchecked.
- **(b) untested seam.** No test exercises `loadDocuments → ingestData → store` together, nor
  `RagRepository.ingest`. The plan's integration test ("small `.txt`/`.jsonl` fixture → expected chunk
  count inserted; `count()` matches") is satisfied only in two halves (loader test + pipeline test);
  the file→pipeline join is unproven on host.
- **(c) device-only:** the on-device ingest smoke (`RagDeviceTest`), correctly deferred.

**Drift / doubtful claims:**
- IMPLEMENTATION_ORDER #26 note claims "`ORTRetriever.ingestData` binds the real embedder,
  `RagRepository.ingest` + facade `ingest`" — all three verified true.
- HANDOFF W1 claims tests "`DocumentChunkerTest`/`DocumentSourceTest`/…" — verified present with the
  asserted content.

**% complete: 80%.**

---

### #27 — RAG config surface & grounded generation

**Required:** public `RagConfig` mapped to `ORTRagConfig` via the `ORTRagArguments.overwriteWith` pattern,
adding internal `minScore` + `indexingMode`; `searchType` as the `SearchType` enum; `similarityMetric`
read-only COSINE; `indexingMode=dynamic` a fail-closed F7 stub; `LLMRepository.prepareRetriever` applies
the override (the `:306` TODO); `PromptAssembler` with the exact default template + caller override;
`ORTRetriever.retrieve(query, ragConfig): List<RagMatch>`; facade `generateWithRag → GroundedResult(text,
matches, prompt)`; JVM mapper/prompt/**grounded-flow** tests; the ingest→retrieve→generate workflow.

**Verified present:**
- `AND/config/PublicConfigs.kt:71-86` public `RagConfig` — `searchType: SearchType`, `minScore: Double =
  0.0`, `indexingMode: IndexingMode = PRECOMPUTE`, and `val similarityMetric: String get() = "COSINE"`
  (read-only, no setter).
- `AND/ORTRagConfig.kt:24-25` new internal `minScore` + `indexingMode` fields; `:9-10` mirrored in
  `ORTRagArguments`; `:43-44` threaded through `overwriteWith`.
- `AND/internal/config/ConfigMappers.kt:96-114` `RagConfig.toOrt()` with the F7 gate at `:98-100`
  (`NotImplementedFeatureException("indexingMode=dynamic …")`) — same pattern as unsupported `HandoffMode`.
- `AND/repository/LLMRepository.kt:293-306` `prepareRetriever` — the `// TODO: Override RAG config if
  needed` is gone; `val finalRagConfig = ragArgs ?: ragConfig; ragConfig = finalRagConfig` and
  `makeOrtRag(finalRagConfig)`; `makeOrtRag` (`:278-289`) now passes `ortArgs` into the `ORTRetriever`
  (comment `:284` "was previously ignoring ortArgs").
- `minScore` threading into search: `AND/ORTRetriever.kt:113`
  `vectorStore()?.search(embeddings, ragArgs.topK, ragArgs.minScore)` → `ObjectBoxVectorStore.search` →
  `ORTVectorDatabase.queryDocuments(..., minScore)` → `.filter { it.second >= minScore }` (`:238`).
  The chain is complete.
- `AND/rag/PromptAssembler.kt:14-32` — the default template matches the plan byte-for-byte ("Use the
  following context to answer the question." / blank / "Context:" / `- ` bullets in match order / blank /
  "Question: {q}" / "Answer:"), plus a `PromptStrategy` fun-interface override hook.
- `AND/runtime/Results.kt:54-58` `GroundedResult(text, matches, prompt)`;
  `AND/internal/runtime/RepositoryBackedModelSession.kt:215-227` implements retrieve → assemble →
  generate and returns all three; facade `AND/MobileTransformerModel.kt:71-76`.
- Constants + parity: `AND/constants/IndexingMode.kt` mirrors
  `src/mobiletransformers/config/constants.py:51-55`; `schemas/enums.json` carries
  `"IndexingMode": ["precompute","dynamic"]`.
- Tests: `TST/facade/RagConfigMapperTest.kt` (field-by-field mapping, defaults, read-only COSINE,
  `DYNAMIC` fail-closed); `TST/rag/PromptAssemblerTest.kt` (template content, bullet order, caller
  override, empty-matches).
- Device checkpoint written: `androidTest/.../RagDeviceTest.kt:44-48` asserts non-empty matches +
  inspectable prompt.

**Gaps:**
- **(a) real bug — the public `RagConfig` override is applied only ONCE per session.**
  `RepositoryBackedModelSession.retrieve` (`:187-206`) calls `rag.initialize(config.toOrt(), adapter)`,
  but `RagRepository.initialize` (`:31-44`) only reaches `prepareRetriever(ragConfig)` **`if
  (llmRepository.ortRetriever == null)`**. It then calls `rag.query(query, ragCallback = adapter)` with
  `ragConfig = null` (`:205`), so `LLMRepository.runRetriever` (`:411-413`) resolves
  `ragConfig.overwriteWith(null)` = the *stored* config. Net effect: the second and subsequent
  `retrieve(...)`/`generateWithRag(...)` calls in a session **silently ignore** a changed `minScore`,
  `topK`, or `searchType`. The DoD ("`prepareRetriever` applies the public override through the typed
  mapper") holds only for the first call. Same defect path applies to `ingest` (`RagRepository.ingest:22`
  → same `initialize`).
- **(a) missing symbol.** The plan names `ORTRetriever.retrieve(query, ragConfig): List<RagMatch>`. No
  such method exists; retrieval is still `query(queryText, ragArgs, ragCallback)` (`:74`) delivering
  results through the `RagCallback`, with the session adapter reshaping to `RetrievalResult`. Functionally
  equivalent, but the normative signature was not implemented and the callback round-trip is what makes
  the override bug above possible.
- **(a) cross-language contract drift — the Python `RagConfig` was never extended.**
  `src/mobiletransformers/config/models.py:109-112` is still
  `{searchType: SearchType = semantic, topK: int = 5, embeddingDim: int = 384}`, and
  `schemas/RagConfig.schema.json` + `docs/CONFIGURATION.md:61-67` document exactly those three.
  The device reads eleven fields from `rag_config.json` (`AND/FileUtil.kt:256-268`), including the two
  fields this plan introduced (`minScore`, `indexingMode`) plus `repoName`/`onnxName`/`chunkSize`/
  `chunkOverlap`/`maxTextLength`/`deviceOptions`. Worse, the wire names disagree for the dimension:
  Python alias `embeddingDim` (default 384) vs Kotlin `embeddingDimension` (default 256), and `topK`
  defaults differ (5 vs 10). The plan states the Python `RagConfig` **is** the `rag_config.json` contract
  Kotlin validates against — it demonstrably is not.
- **(a) the deferred `SearchType` String→enum swap never landed.** `ORTRagConfig.searchType` is still
  `String = "semantic"` (`ORTRagConfig.kt:23`) and `ORTRetriever.query` still dispatches on raw strings
  (`when (ragArgs.searchType) { "semantic" -> …; "text" -> … }`, `:85-145`). The enum exists only on the
  public side. #25's deferral note ("rides with the facade plans #17/#19") is therefore still open, and
  the #25 canonical note "Search mode is the `SearchType` enum … not a bare `String`" is unmet internally.
  Dead code as a side effect: `ORTRetriever.kt:80` `if (ragArgs.searchType == null)` compares a
  non-nullable `String` to `null` — always false.
- **(a) `indexingMode` is a defined-but-never-read internal field.** It is validated at the mapper
  boundary (F7 gate) and stored on `ORTRagConfig`, but nothing in `ORTRetriever`/`IngestionPipeline`/
  `RagRepository` ever reads `ragConfig.indexingMode`. Combined with `maxTextLength` (#26), that is two
  RAG config fields that exist on the wire and in the type but influence no behaviour — the
  `TrainingEvent.Metric`-style dead surface the audit asked about does have RAG analogues.
- **(b) the grounded-flow test does not test the grounded flow.** `TST/rag/GroundedFlowTest.kt:20-41`
  searches an `InMemoryVectorStore`, calls `PromptAssembler.assemble`, then **hand-constructs**
  `GroundedResult(text = fakeGenerate(prompt), matches, prompt)` and asserts on its own construction. It
  never calls `ModelSession.generateWithRag`, so the composition in
  `RepositoryBackedModelSession.kt:215-227` (the actual deliverable) has zero host coverage. The plan
  asked for "`generateWithRag(...)` → assert the retrieved matches feed the prompt".
- **(a) docs.** `docs/RAG.md:57-64` "Not yet (tracked)" still lists grounded generation and the
  `SearchType` swap as pending; implementation step 5 (document the default template, override hook, and
  `precompute` vs `dynamic`) is unstarted. (Page owned by #31, but no content exists to move.)
- **(c) device-only:** the #27 end-to-end checkpoint (`RagDeviceTest`), correctly outstanding.

**Drift / doubtful claims:**
- IMPLEMENTATION_ORDER #27 note: "fixed `makeOrtRag`/`prepareRetriever` override + `minScore` threading" —
  true as written, but incomplete: the override never reaches a *second* call because
  `RagRepository.initialize` short-circuits. Verified by reading the three-hop call chain.
- IMPLEMENTATION_ORDER #27 note: "JVM tests (mapper/prompt/**grounded flow**)" — the grounded-flow test
  does not exercise `generateWithRag`; the claim overstates coverage.
- IMPLEMENTATION_ORDER #27 note: "new `IndexingMode` enum (Python+Kotlin+parity)" — verified true
  (`constants.py`, Kotlin mirror, `schemas/enums.json`). But the *model* (`models.py::RagConfig`) never
  gained the field, so the enum is parity-clean while the config contract is not.

**% complete: 75%.**

---

## Tier-doc requirements not picked up by any code plan

From `agent_docs/03_tier2_inference_and_rag.md`:

1. **"Document supported input/output names and graph requirements from the real loop"** (`Native Path
   Hardening`, first bullet) — assigned to #23, produced nowhere: no comment block in
   `CPP/inference.cpp`, no `docs/` hit for `input_ids`/`position_ids`/`past_key_values`/`logits`,
   no `docs/ARCHITECTURE.md`. The table exists only inside the plan file itself.
2. **"Add explicit session lifecycle tests for switching between training and generation"** — only a
   device test (`androidTest/.../TrainMergeGenerateTest.kt`). No host-side lifecycle/leak test exists,
   and none is possible without JNI; effectively an unbudgeted device-only item.
3. **"Reflect the Tier 0 GenAI/manual decision in package manifest `supportedEngines` / `defaultEngine`"**
   (Implementation Sequence step 3) — not wired. `LLMRepository.kt:273-275` documents the hole:
   "supportedEngines would come from the manifest variant (#13); **until wired here**, offer both".
   `ModelRuntimeFactory.create` defaults `supportedEngines = setOf("native","genai")` unconditionally
   (`ModelRuntime.kt:122`). Nominally #11's, but no Tier-2 plan re-claims it and it is the mechanism the
   tier doc names for expressing the Gate 0.1 decision on-device.
4. **"HF API alignment audit — create a mapping table in docs"** (its own section) — the table lives only
   in the tier doc. `docs/PUBLIC_API.md` covers Python + CLI only; there is no Kotlin/HF alignment page.
   #24 step 5 named `docs/PUBLIC_API.md` as a target; #31 lists `docs/ANDROID_SDK.md` as pending.
5. **`similarityMetric` in the public `RagConfig` list** — implemented as a read-only `"COSINE"` string
   (`PublicConfigs.kt:85`) rather than the `SimilarityMetric` enum the "enums own every closed string set"
   canonical decision would imply. Deliberate per the plan text ("expose it read-only, do not pretend it
   is configurable"), so this is a conscious, documented narrowing rather than a gap — noted for
   completeness.

## Remaining work, ordered

### Host-doable now (no device, no Gradle run needed to write)
1. **#24 — replace `ORTGeneratorGenAI.samplingMethodInt` (`ORTGeneratorGenAI.kt:118-123`) with
   `SamplingMethod.fromWire(method).nativeOrdinal`.** Direct DoD violation; one-line fix + a test.
2. **#24 — reconcile the generation loop bound.** `ORTGeneratorNative.kt:174` (`<=`) vs
   `ORTGeneratorGenAI.kt:73` (`<`). Pick new-token semantics, then add the plan's missing
   `maxNewTokens`-semantics test (a fake `ModelRuntime` makes this host-testable).
3. **#24 — align the `InferenceProgress` payloads** (`onCompletion.avgTokensPerSecond`,
   `onStartGeneration.isCompleted`, the `trackMetrics` gating) and add a host callback-sequence test over
   two fake `ModelRuntime`s so parity stops depending on a device.
4. **#27 — fix the one-shot override.** `RagRepository.initialize` (`:31-44`) must re-apply a changed
   `RagConfig` (or `retrieve`/`ingest` must pass `ORTRagArguments` down `runRetriever`). Add a host test
   asserting a second `retrieve` with a different `minScore`/`topK` takes effect.
5. **#27 — make `GroundedFlowTest` call `ModelSession.generateWithRag`** against a fake runtime + the
   in-memory store, instead of hand-constructing `GroundedResult`.
6. **#27 — extend `models.py::RagConfig`** with `minScore`/`indexingMode` (+ the fields the device
   actually reads), reconcile `embeddingDim` vs `embeddingDimension` and the `topK`/dimension defaults,
   regenerate `schemas/RagConfig.schema.json`, update `docs/CONFIGURATION.md:61-67`.
7. **#26 — honour `maxTextLength`** in `ORTRetriever.ingestData`/`IngestionPipeline`, or delete the field
   from the public surface. Same call for `indexingMode` (#27): consume it or mark it explicitly reserved.
8. **#26 — accept a directory** in `loadDocuments`/`RagRepository.ingest` (implementation step 3), plus a
   host test covering `loadDocuments → ingestData → count()`.
9. **#23 — make the C++ load fail closed to the caller.** `session_cache.h:740-744` must propagate
   `weight_session->init(...) == false` (throw naming the tensor) instead of `LOGE` + base-weight
   fallback; add the required "unsupported model / missing required input" check before
   `createInferenceSession` (`session_cache.h:499-524`).
10. **#23 — write the JVM map-driven-naming fixture test** the plan requires (currently a grep in
    `NativeLoadRegressionTest.kt:40-43`), and give `ORTConversationStateTest` a real (non-null) template
    handler so the rendered-offset fix is actually exercised.
11. **#23 — delete `ORTGenAITokenizer.kt`** (and its unused field/import at `LLMRepository.kt:6,147`), and
    extend `NativeLoadRegressionTest` to grep for surviving `NotImplementedError` GenAI stubs, not just
    the two deleted files. Clear the stale `TODO: Will need fix` at `ORTGeneratorNative.kt:130-131`.
12. **#25 — clear the two stale TODOs** (`ORTVectorDatabase.kt:229`, `VectorEntity.kt:164`), strip
    embeddings in `queryByContent`, and make the `else -> emptyList()` dimension branches throw.
13. **#23/#24 — write the graph-I/O contract** (input/output names, optional `position_ids`, KV shapes)
    as a comment block in `inference.cpp` and as the seed of `docs/ARCHITECTURE.md`.
14. **#26/#27 — update `docs/RAG.md`** (remove "Not yet (tracked)", state the txt/md/jsonl v1 scope with
    PDF/Word/HTML explicitly out, document the default prompt template + override hook and
    `precompute`/`dynamic`). This alone unblocks #26 self-check item 3.
15. **Internal `SearchType` enum swap** in `ORTRagConfig`/`ORTRetriever.query` — the #25 deferral to
    #17/#19 that never landed; also removes the always-false `searchType == null` branch.

### Device-required (instrumented tests already written; need a real #9 package pushed)
- `ConversationResetTest` — two-prompt no-leak (#23).
- `TrainMergeGenerateTest` — train→merge→generate reflecting merged weights, session-lifecycle/leak (#23).
- `FacadeLoadGenerateTest` — map-driven load + generate over a real package (#23).
- `DualEngineParityTest` — currently only first-token equality; needs extending to the ordered callback
  sequence + payload fields before it can close #24's parity box.
- `RagDeviceTest` — ingest → `generateWithRag` (#26 device smoke and the #27 checkpoint).
- ObjectBox parity smoke for supported dimensions (#25) — no test class exists yet.

### Manual / user-run
- Produce and push the real #9 export package (with `weight_handoff_map.json` + flat per-tensor `.bin` +
  `genai_config.json`) that every device test above `assumeTrue`s on — this is the single blocker for
  the #23/#24/#27 boxes.
- Run `./gradlew :MobileTransformers:testDebugUnitTest` + `compileDebugKotlin` after the host fixes above
  (not run in this audit — static verification only, per instructions).
