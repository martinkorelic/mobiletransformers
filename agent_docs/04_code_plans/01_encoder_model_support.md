# Encoder-Model Support

**Priority #32 | Prerequisites: #12 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`), #6 (`01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`), #8 (`01_code_plans/01_unified_merger_and_external_data_export.md`) | Blocks: #34 (`04_code_plans/03`, optional head federation)**

> Tier-3, best value/effort. Spike-gated; must not block v1.0. Avoids the autoregressive KV-cache path entirely.
>
> **Consumes `00_code_plans/09`.** Adding Bert/encoder support is an **architecture-registry entry** (`ArchitectureSpec` with `onnx_config_class=BertOnnxConfig`, encoder `target_modules`, `attention_module_name`), not a new `elif` in `trainer/builder.py:261-272` or `inference/builder.py:3234`. The encoder `target_modules` come from 09's registry, which also **de-duplicates** the two identical tables `TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING` (`peft_models/mars/utils.py:1`) ≡ `TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING` (`peft_models/ablation/utils.py:1`). The MARS architecture assumptions (`peft_models/mars/model.py:53,60,75,138`: attention module named `self_attn`, `kwargs['hidden_states']` location) become per-architecture **registry data**, which is exactly what makes encoder support clean.

## Purpose

Extend the framework from "decoder LLM fine-tuner" to "general on-device transformer PEFT framework" by supporting encoder models for classification, embedding, and similarity/intent tasks. The repo already has the footholds (`task_type` branch, `BertOnnxConfig`, pooling builder, native `generateEmbedding`); this plan proves and packages an encoder train→infer path and verifies MARS transfer.

## Touched / new files

Python:
- `trainer/builder.py` — reuse the `task_type` branch (`:254-257`, `AutoModel` vs `AutoModelForCausalLM`); the `BertOnnxConfig` selection (`:271-272`) moves into 09's architecture registry (no `elif`). Encoder `peft_target` defaults come from the registry's `target_modules`, not a hand-added branch at `:282-346`. The `feature-extraction` + `add_pooling` path (`:375, :410`) already exists.
- `trainer/embedding_builder.py` — reuse `add_pooling_to_onnx_model` (`:6-64`) and `add_mean_pooling` (`:403-511`) for embedding heads.
- `trainer/utils.py` — the unified mapping builder (09, replacing `create_mars_adapter_mapping` `:533-668` / `create_lora_mapping` `:670-703`) must yield a valid `peft_mapping` for encoder linear layers; encoder target/attention rules are added as registry data (09), not as new branches here.
- NEW `trainer/classification_head.py` — optional supervised classification head export.
- `config.yml` — `task_type: feature-extraction` already documented (`:26-28`) with candidate encoders (`:18-21`).

Manifest:
- `mobiletransformers_manifest.json` (#12) — add `EncoderTaskConfig` / classification-head contract fields (task, num_labels, pooling, label map).

C++/Kotlin:
- `inference.cpp` — `generateEmbedding` (`:185-245`) is the encoder inference path; no KV-cache. Add a classification post-process (argmax/softmax) if a head is present.
- `android/.../ORTRetriever.kt` — embedding path already uses `performEmbeddingStep` (`:192-200`); classification reuses the same session with a head.

## Data contracts / interfaces

### `EncoderTaskConfig` (manifest, additive)

```json
{
  "taskType": "feature-extraction | text-classification | similarity",
  "pooling": "mean | cls | max | mean_sqrt_len",
  "numLabels": 0,
  "labelMap": {"0": "negative", "1": "positive"},
  "embeddingDimension": 384
}
```

`embeddingDimension` must be one of the eight supported by the vector store if used for RAG (`03_code_plans/03`).

### MARS-transfer verification (spike gate output)

`create_mars_adapter_mapping` must produce, for an encoder, a `peft_mapping` whose base-layer names join cleanly with the inference-graph initializer names (the `TrainableTensorCodec` invariant, `00_code_plans/07`). If the decoder-specific `self_attn`/`q_proj` naming doesn't match encoder modules, add encoder rules — do not silently emit a mismatched map.

## Implementation steps

1. **Spike (gate):** export one small encoder (`sentence-transformers/all-MiniLM-L6-v2` for embedding, or a small BERT/MiniLM classifier). Prove: ONNX export, training-artifact generation (`generate_artifacts`, #3), one desktop train step, one Android/device train step, and a simple metric.
2. Confirm MARS/LoRA mapping validity (above); add encoder target defaults.
3. Add `EncoderTaskConfig` + classification-head contract to the manifest (#12) and the handoff map (#8) for trainable encoder linears.
4. Wire classification post-process into the native embedding path (no KV-cache).
5. Add one worked example task (intent classification or embedding similarity).

## Interactions

- **#8 / #12**: encoder trainable tensors flow through the same codec/manifest; reuse, don't fork.
- **#6 (optimum-onnx export)**: `BertOnnxConfig` export goes through the same TasksManager front door.
- **`03_code_plans/03` (vector store)**: encoder embeddings feed RAG when dimension ∈ the eight.
- **#34 (federated)**: optional head/encoder tensors can be federated after this passes.

## Tests & smokes

- Encoder export smoke (MiniLM/BERT fixture) — ONNX + training artifacts generate.
- One desktop train step runs; loss is finite.
- MARS-mapping validity test: `peft_mapping` joins with inference initializers (no `HandoffMapError`).
- Embedding/classification inference: output shape + metric correct.
- Android one-step training smoke (or a documented blocker).
- No KV-cache code is reachable on the encoder path.
