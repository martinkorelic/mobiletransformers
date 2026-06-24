# Implementation Order — Code Plans Index

This index orders every feature code-plan under `agent_docs/00_code_plans/` … `agent_docs/05_code_plans/` from **most foundational first**. Implement in this order; each plan file repeats its own Prerequisites / Blocks header.

The high-level "what & why" lives in the six tier docs (`00_repository_restructure_plan.md`, `01_tier0_foundation_decisions.md`, `02_tier1_hf_integrated_core.md`, `03_tier2_inference_and_rag.md`, `04_tier3_reach_extensions.md`, `05_cross_cutting_release_modernization.md`). These code plans are the "how" — concrete enough for another agent to implement.

## Canonical decisions every plan inherits

- **Dual inference engine over one package.** The same exported package + same Android-cache folder is consumable by **both** the native ORT engine and the ONNX Runtime GenAI engine. GenAI is a *selectable* engine (Native is the default/guaranteed path), not a separate package. See `01_code_plans/03_inference_engine_abstraction_native_and_genai.md`.
- **Unified weight handoff via external initializers.** Trainable/merged weights are ONNX **external initializers, one-file-per-tensor**, in `<cacheDir>/<model>/inference/`. The frozen quantized base is a separate immutable external blob. On-device merge overwrites the per-tensor files (atomic rename + checksum). No graph rewrite, no weights-as-inputs, no genai fork. See `01_code_plans/01_unified_merger_and_external_data_export.md`.
- **`weight_handoff_map.json` is the single source of truth** that replaces hard-coded name rewrites (`weight_merger.cpp:904`). Schema lives in `00_code_plans/07_weight_handoff_map_and_tensor_codec.md`.
- **Dependency profile isolation** (uv groups/extras): `onnxruntime-training` (source-built, provides `onnxruntime`), public `onnxruntime`, `onnxruntime-genai`, and `optimum-onnx[onnxruntime]` collide on the `onnxruntime` import and must never share one env. See `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`.
- **Apache-2.0** is the target framework license (model weights keep upstream licenses).
- **External-initializer handoff is the only merge path.** The legacy `inference/merged/` subdirectory is retired; merged weights are flat per-tensor external initializers in `inference/`. Tier 2 consumes the existing `ModelRuntime` engine boundary and must not define a competing `InferenceEngine`. See `03_code_plans/01`.
- **`FederatedAdapterRecord` derives from `TrainableTensorCodec`.** Federated exchange (Tier 3) is a thin wrapper over `weight_handoff_map.json` + the codec from `00_code_plans/07`; it never invents its own tensor ordering. See `04_code_plans/03`.
- **Registries replace hardcoded dispatch.** PEFT methods, model architectures, and merger variants are declared in data-driven registries; no `if x == "lora"/elif "mars"` or `architectures[0] == "..."` chains in business logic. Adding a method/architecture/merger is a registry entry + an enum member, not a new `elif`. Owned by `00_code_plans/09_typed_models_enums_and_registries.md`.
- **Pydantic v2 is the typed config contract.** Every tunable config object is a Pydantic model with camelCase aliases; cross-boundary JSON is `model_dump(by_alias=True)` + a generated `schemas/*.schema.json` the Kotlin parser and C++ loader validate against (fail-closed). Secrets stay in `Settings` (`00_code_plans/02`). `pydantic>=2` is a core dependency (`00_code_plans/03`). Owned by `00_code_plans/09`.
- **Enums own every closed string set**, mirrored Python (`config/constants.py`) ↔ Kotlin (`.../constants/*.kt`): sampling method, scheduler type, execution provider, core/memory config id, search type, quantization type, PEFT method, task type, handoff mode, merger variant. Owned by `00_code_plans/09`.
- **Full merger unification.** A single parameterized `build_merger_model(MergerSpec)` replaces `create_lora_merger_model{,_2}` / `create_mars_merger_model{,_2}` (the "merger 1/2" duplication); C++ `get_merger_type`/`run_merger_model` resolve the spec from the handoff map + merger registry instead of branching on `"lora_q"/"mars_q"`. Owned by `00_code_plans/09`, wired by `01_code_plans/01`.

## Global order

| # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- |
| 1 | `00_code_plans/01_python_package_and_uv_scaffolding.md` | Unblocks all Python work | — |
| 2 | `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md` | Lock the toolchain before building on it (incl. `pydantic>=2` core) | 1 |
| 3 | `01_code_plans/06_source_built_ort_training_pipeline.md` | Prove the train toolchain is alive (`generate_artifacts`) | 2 |
| 4 | `00_code_plans/02_config_layering_settings_constants.md` | Three config layers + secrets; needed by export/CLI | 1 |
| 5 | `00_code_plans/09_typed_models_enums_and_registries.md` | **Typed config + enums + PEFT/arch/merger registries** — kills hardcoded dispatch before builders are written | 2, 4 |
| 6 | `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md` | Inference-export front door (arch via registry) | 2, 4, 5 |
| 7 | `00_code_plans/07_weight_handoff_map_and_tensor_codec.md` | Data contract every later piece reads (codec consumes registries) | 4, 5 |
| 8 | `01_code_plans/01_unified_merger_and_external_data_export.md` | **Dual-engine core** + full merger unification | 5, 6, 7 |
| 9 | `01_code_plans/02_genai_external_data_swap_spike.md` | Feeds Gate 0.1 | 8 |
| 10 | `01_code_plans/03_inference_engine_abstraction_native_and_genai.md` | Engine selection | 9 |
| 11 | `01_code_plans/04_memory_mapping_experiments.md` | Optimizes 8–10 (non-blocking) | 8 |
| 12 | `00_code_plans/06_manifest_first_package_and_cache_bridge.md` | Package contract | 7, 8 |
| 13 | `02_code_plans/03_hub_model_package_format.md` | Hub repo shape | 12 |
| 14 | `02_code_plans/05_one_command_export_cli.md` | Wraps 6–8 + 12 | 6, 8, 12 |
| 15 | `00_code_plans/04_android_gradle_rename_migration.md` | Isolated, verified rename | — |
| 16 | `00_code_plans/05_android_facade_foundation.md` | Public SDK facade | 12, 15 |
| 17 | `00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md` | Job/progress/checkpoint API | 16 |
| 18 | `02_code_plans/01_hf_style_kotlin_facade.md` | `fromPretrained`/train/merge/generate | 10, 16 |
| 19 | `02_code_plans/02_optimum_support_matrix.md` | Reporting layer | 6 |
| 20 | `02_code_plans/04_hub_pull_and_cache_flow.md` | Python pull first, Android downloader next | 12, 13 |
| 21 | `02_code_plans/06_adapter_pushback.md` | Last Tier-1 piece | 8, 13 |

### Tier 2 — Inference & RAG (Phase 7)

| # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- |
| 22 | `03_code_plans/01_inference_handoff_alignment_and_native_hardening.md` | Wire Native into `ModelRuntime`, retire `inference/merged/` | 10, 8, 7 |
| 23 | `03_code_plans/02_sampling_and_streaming_public_config.md` | HF-aligned generation config (enum-typed) + callback parity | 22, 18 |
| 24 | `03_code_plans/03_vector_store_boundary_and_inmemory.md` | Testable `VectorStore`; `SearchType` enum; dynamic dimension registry | 16 |
| 25 | `03_code_plans/04_rag_ingestion_and_chunking.md` | Implement `ingestData()` | 24, 22 |
| 26 | `03_code_plans/05_rag_config_and_grounded_generation.md` | Public `RagConfig` + grounded flow | 25, 23, 20 |

### Cross-cutting — Release modernization (Phase 8; runs alongside, gated)

| # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- |
| 27 | `05_code_plans/01_makefile_and_cli_entrypoints.md` | One-command path | 1, 14, 2 |
| 28 | `05_code_plans/02_ci_staged_pipeline.md` | Standing "it works" proof | 27, 1, 3 |
| 29 | `05_code_plans/03_aar_maven_publication.md` | Portable Android consumption | 15, 27 |
| 30 | `05_code_plans/04_docs_set_and_compatibility_matrix.md` | Public docs + registry-driven matrix as contracts stabilize | 22-26, 18, 12 |
| 31 | `05_code_plans/05_versioning_license_release.md` | v1.0 release gate | 28, 29, 30 |

### Tier 3 — Reach extensions (Phase 9; each spike-gated, never blocks v1.0)

| # | Plan | Why here | Prerequisites |
| --- | --- | --- | --- |
| 32 | `04_code_plans/01_encoder_model_support.md` | Cheapest reach; arch-registry entry + codec/manifest reuse | 12, 6, 8 |
| 33 | `04_code_plans/02_training_scheduler_workmanager.md` | Charging-cycle training; closes `ORTScheduler` state-persistence gap | 17, 16 |
| 34 | `04_code_plans/03_federated_codec_and_python_simulation.md` | Python Flower sim (Option A first) | 7, 8, 12 |
| 35 | `04_code_plans/04_federated_android_gateway.md` | Android client + gateway (Option B) | 34, 33, 17 |
| 36 | `04_code_plans/05_functiongemma_architecture_gate_and_intents.md` | Gemma-3 inference-graph as arch-registry entry + intents | 10, 6, 8 |
