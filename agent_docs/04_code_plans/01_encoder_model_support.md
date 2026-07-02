# Encoder-Model Support

**Priority #33 | Prerequisites: #13 (`00_code_plans/06_manifest_first_package_and_cache_bridge.md`), #7 (`01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`), #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`) | Blocks: #35 (`04_code_plans/03`, optional head federation)**

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
- `mobiletransformers_manifest.json` (#13) — add `EncoderTaskConfig` / classification-head contract fields (task, num_labels, pooling, label map).

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

The `taskType` closed set is **data-driven, not `if/elif`**: add a `TASK_REGISTRY` (F3) alongside the PEFT/architecture/merger registries, with one row per task (`feature-extraction`, `text-classification`, `similarity`, and decoder `text-generation`) carrying its `onnx_config_class` selection, default head, and pooling. Adding an encoder task = a `TASK_REGISTRY` row + a task enum member joined to the relevant `ArchitectureSpec` (`onnx_config_class=BertOnnxConfig`, encoder `target_modules`) — **no business-logic edit** to `trainer/builder.py:254-272` and **no KV-cache** (the encoder path runs `generateEmbedding`, never the autoregressive loop).

### MARS-transfer verification (spike gate output)

`create_mars_adapter_mapping` must produce, for an encoder, a `peft_mapping` whose base-layer names join cleanly with the inference-graph initializer names (the `TrainableTensorCodec` invariant, `00_code_plans/07`). If the decoder-specific `self_attn`/`q_proj` naming doesn't match encoder modules, add encoder rules — do not silently emit a mismatched map.

## Implementation steps

1. **Spike (gate):** export one small encoder (`sentence-transformers/all-MiniLM-L6-v2` for embedding, or a small BERT/MiniLM classifier). Prove: ONNX export, training-artifact generation (`generate_artifacts`, #3), one desktop train step, one Android/device train step, and a simple metric.
2. Confirm MARS/LoRA mapping validity (above); add encoder target defaults.
3. Add `EncoderTaskConfig` + classification-head contract to the manifest (#13) and the handoff map (#9) for trainable encoder linears.
4. Wire classification post-process into the native embedding path (no KV-cache).
5. Add one worked example task (intent classification or embedding similarity).

## Interactions

- **#9 / #13**: encoder trainable tensors flow through the same codec/manifest; reuse, don't fork.
- **#7 (optimum-onnx export)**: `BertOnnxConfig` export goes through the same TasksManager front door.
- **`03_code_plans/03` (vector store)**: encoder embeddings feed RAG when dimension ∈ the eight.
- **#35 (federated)**: optional head/encoder tensors can be federated after this passes.

## Worked example

A Bert/encoder entry is **registry data**, not new branches. Sketch of the `TASK_REGISTRY` + `ArchitectureSpec` rows (09) and the encoder export/inference calls they drive:

```python
# mobiletransformers/config/registry/ (09's registries) — closed sets, no if/elif
TASK_REGISTRY["feature-extraction"] = TaskSpec(
    onnx_config_class=BertOnnxConfig,          # optimum-onnx front door (#7)
    auto_model_class=AutoModel,                 # not AutoModelForCausalLM
    default_head=None,                          # pooling only
    pooling="mean",
)
TASK_REGISTRY["text-classification"] = TaskSpec(
    onnx_config_class=BertOnnxConfig,
    auto_model_class=AutoModelForSequenceClassification,
    default_head="classification",              # -> classification_head.py
    pooling="cls",
)

ARCHITECTURE_REGISTRY["bert"] = ArchitectureSpec(
    onnx_config_class=BertOnnxConfig,
    target_modules=["query", "key", "value", "dense"],   # encoder linears, not q_proj/self_attn
    attention_module_name="attention.self",
    inference_model_class=None,                  # encoder uses generateEmbedding, no KV-cache class
)
```

Export wires the pooling head and runs the non-autoregressive path:

```python
# trainer/embedding_builder.py — feature-extraction branch already exists (:375, :410)
add_pooling_to_onnx_model(onnx_model, pooling="mean")    # reuse :6-64 / add_mean_pooling :403-511
```

```kotlin
// Android: encoder inference is the embedding path — no token loop
val embedding: FloatArray = model.generateEmbedding(text)   // ORTRetriever.performEmbeddingStep :192-200
// if a classification head is present, argmax/softmax post-process in inference.cpp:185-245
```

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `pytest tests/encoder/test_mapping.py` — MARS/LoRA mapping validity: the unified builder (09) yields a `peft_mapping` whose encoder base-layer names join cleanly with the inference-graph initializer names (no `HandoffMapError`); assert encoder `target_modules` come from the registry, not a `self_attn`/`q_proj` decoder default.
- `pytest tests/encoder/test_no_kv_cache.py` — regression/grep: no KV-cache symbol is reachable on the encoder (`feature-extraction`/`text-classification`) path.
- `pytest tests/encoder/test_task_registry.py` — `TASK_REGISTRY` carries `feature-extraction`/`text-classification`/`similarity` rows; selecting one returns `AutoModel`/`BertOnnxConfig` (no `elif` in `trainer/builder.py`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `pytest tests/encoder/test_embedding_infer.py` — load a tiny pre-exported encoder fixture, run embedding/classification inference, assert output shape (== `embeddingDimension`) and a deterministic metric on a 2-example fixture.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Encoder export smoke (`sentence-transformers/all-MiniLM-L6-v2` or a small BERT/MiniLM classifier): ONNX + training artifacts (`generate_artifacts`, #3) generate.
- One desktop train step runs; loss is finite (requires the source-built ORT-training wheel).
- Android one-step training smoke on a device (or a documented blocker).

**Definition of done** — encoder support ships as a `TASK_REGISTRY` row + an `ArchitectureSpec` entry (no new `elif` in `trainer/builder.py:254-272` or `inference/builder.py:3234`); the spike exports a small encoder, generates training artifacts, runs one desktop train step (and one device step or a documented blocker), and the unified mapping joins cleanly with inference initializers; embedding/classification inference returns the correct shape + metric through `generateEmbedding`; no KV-cache code is reachable on the encoder path.
