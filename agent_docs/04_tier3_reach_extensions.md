# Tier 3 - Reach Extensions

> Detailed code-implementation plans for these features live in `agent_docs/04_code_plans/` (and cross-referenced 00 plans); global order in `agent_docs/IMPLEMENTATION_ORDER.md`.
>
> **Reach via registries, not branches.** New architectures (Bert/encoder, Gemma-3) and PEFT/merger variants are added as entries in the registries owned by `00_code_plans/09` — never new `if architectures[0] == "..."` / `if peft_method == "..."` chains.

## Purpose

Tier 3 expands the release claim after Tiers 0-2 are solid. These extensions should not block the core v1.0 framework. Each item needs a spike gate and a clear stop condition.

## Current Repo Evidence

- `config.yml` already includes `task_type: feature-extraction` and candidate embedding models such as `sentence-transformers/all-MiniLM-L6-v2`, `Qwen/Qwen3-Embedding-0.6B`, and `intfloat/multilingual-e5-small`.
- `trainer/builder.py` includes `BertOnnxConfig` and uses `AutoModel` when `task_type` is not text generation.
- `trainer/embedding_builder.py` can add pooling to ONNX embedding models.
- `inference.cpp` has `generateEmbedding`, and `ORTRetriever` uses an embedding session.
- `ORTTrainerNative` already saves and loads `training_state.json`, scheduler state, epoch, and global step, which is useful for interrupted or charge-cycle training.
- `ORTTrainerNative` saves checkpoints at configured `saveSteps`, records per-step memory/loss metrics, can merge weights at the end of training, and exposes native hooks for `saveModel`, `optimizerStep`, and `mergeExportWeights`.
- `TrainingRepository.performTraining` and `LLMRepository.runTraining` already separate repository orchestration from the native trainer, which gives a natural place to add scheduled or federated orchestration wrappers without burying network code inside C++.
- The current repo has no federated protocol, no stable trainable-tensor import/export API, and no canonical adapter tensor ordering for exchanging updates across clients.
- There is no WorkManager scheduler layer yet.
- `trainer/builder.py:263` includes Gemma architecture handling, including `Gemma3ForCausalLM` (training export), but the inference graph builder only handles Gemma/Gemma2 (`inference/builder.py:3234-3236`) — Gemma-3 inference-graph export is unproven and is the true FunctionGemma blocker.

## External Research Summary

- Android WorkManager supports persistent scheduled work with constraints such as charging, idle, and battery conditions. Source: https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started/define-work
- Android long-running background work often requires foreground-service treatment and user-visible notifications. Source: https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running
- Android Doze defers jobs and WorkManager tasks while the device is unplugged and idle; connecting a charger exits Doze and lets pending jobs run. Source: https://developer.android.com/training/monitoring-device-state/doze-standby
- Android 14 requires foreground services, including long-running WorkManager foreground work, to declare appropriate service types and permissions. Source: https://developer.android.com/develop/background-work/services/fgs/service-types
- Hugging Face hosts model/dataset ecosystems suitable for distributing ready packages and datasets. Source: https://huggingface.co/docs/hub/models-uploading
- ONNX Runtime GenAI can matter for agent/tool work if Tier 0 adopts it and structured/tool generation is supported in the selected release. Source: https://onnxruntime.ai/docs/genai/
- MobileFineTuner increases pressure to show a broader complete system; encoder tasks and real deployment traces can broaden the claim beyond decoder-only training. Source: https://arxiv.org/abs/2512.08211
- Flower is designed for federated learning across heterogeneous edge settings and supports migration from simulation to real devices, but its current Android quickstart is historical: the docs warn that the experimental Android SDK is not compatible with the latest Flower and is being reworked. Sources: https://arxiv.org/abs/2007.14390, https://flower.ai/docs/framework/tutorial-quickstart-android.html
- The Flower community has an ONNX Runtime support discussion, but it is a call for interest/PRs rather than evidence of an official ONNX Runtime integration. Source: https://discuss.flower.ai/t/onnx-runtime-support/786
- Flower's `NumPyClient` interface is framework-agnostic enough for ORT-backed clients because it exchanges lists of NumPy arrays through `get_parameters`, `fit`, and `evaluate`. Source: https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html
- Flower's newer strategy/message stack uses `ArrayRecord` for model parameters, gradients, embeddings, or other named arrays, and built-in strategies such as `FedAvg` expect `ArrayRecord` plus `MetricRecord` replies. Sources: https://flower.ai/docs/framework/ref-api/flwr.app.ArrayRecord.html, https://flower.ai/docs/framework/how-to-use-strategies.html
- ONNX Runtime's own on-device training docs explicitly list federated learning as a target scenario and split training into offline artifact generation plus on-device training, matching this repo's architecture. Source: https://onnxruntime.ai/docs/get-started/training-on-device.html
- ONNX Runtime artifact generation produces training/eval/optimizer models and checkpoints, with `requires_grad` and `frozen_params` defining the trainable parameter set that a federated exchange would need to serialize. Source: https://onnxruntime.ai/docs/api/python/on_device_training/training_artifacts.html
- Flower offers differential privacy hooks and strategy wrappers, but the docs mark differential privacy as preview and warn users to discuss production sensitive-data requirements. Source: https://flower.ai/docs/framework/how-to-use-differential-privacy.html
- Flower has published secure-aggregation work for hiding individual client updates during aggregation, but secure aggregation must be validated against the chosen Android/gateway transport rather than assumed. Source: https://arxiv.org/abs/2205.06117

## Recommended Decision

Prioritize Tier 3 in this order:

1. Encoder-model support, because the repo already has embedding/export footholds and it avoids autoregressive KV-cache complexity.
2. Sleep/charging-cycle training, because existing checkpoint/resume logic makes it a systems extension rather than a new ML algorithm.
3. Federated adapter fine-tuning with Flower, because the repo already has on-device training and Flower can aggregate arrays if MobileTransformers supplies a stable adapter tensor codec.
4. FunctionGemma/mobile-actions personalization, only if the architecture and differentiation gates pass.

Do not allow federated learning or FunctionGemma to delay the v1.0 release. Treat both as showcase extensions or future-work branches unless their spikes are clean.

## 3.1 Encoder-Model Support

### Scope

Support encoder models for:

- Text classification.
- Embedding generation.
- Similarity or retrieval fine-tuning.
- Intent classification for on-device mobile tasks.

### Why This Is Valuable

- It expands MobileTransformers from "decoder LLM fine-tuning" to "on-device transformer PEFT framework."
- Encoders are smaller and practical for mobile.
- Encoder tasks have clearer metrics and shorter inference paths.
- MARS's shared-projection idea should transfer to encoder linear layers, but this must be verified.

### Spike Gate

Use one small encoder model:

- `sentence-transformers/all-MiniLM-L6-v2` for embedding flow, or
- a small BERT/MiniLM classifier for supervised classification.

The spike must prove:

- ONNX export works.
- Training graph/artifacts can be generated.
- One train step runs on desktop.
- One train step runs on Android or a device-equivalent smoke.
- The output task can be evaluated with a simple metric.

### Implementation Notes

- Reuse the existing `task_type` branch (`trainer/builder.py:254-257`, `AutoModel` vs `AutoModelForCausalLM`) and `BertOnnxConfig` (`trainer/builder.py:271-272`); reuse `embedding_builder.add_pooling_to_onnx_model` (`trainer/embedding_builder.py:6-64`) for pooling, and the native `generateEmbedding` path (`inference.cpp:185-245`) for inference.
- **Verify MARS transfer:** `create_mars_adapter_mapping` (`trainer/utils.py:533-668`) was written for decoder QKV/MLP shared projections — confirm it produces a valid `peft_mapping` for encoder linear layers (or add encoder target defaults). This is a spike gate, not an assumption.
- Add encoder-specific `PeftConfig` target-module defaults.
- Add a classification head contract in the model package manifest.
- Avoid KV-cache code entirely (encoders are single forward pass).
- Add `EncoderTaskConfig` or task-specific fields to the manifest. Detailed in `04_code_plans/01`.

### Tests

- Export MiniLM/BERT fixture.
- Generate training artifacts.
- Run one train step.
- Run embedding/classification inference.
- Validate output shape and metric calculation.

## 3.2 Sleep And Charging-Cycle Training

### Scope

Add Android scheduling that trains during favorable device states:

- charging,
- idle when available,
- battery not low,
- optionally unmetered network only for model download, not training.

This should be framed as charging-cycle or opportunistic scheduled training, not invisible sleep training. Android may defer WorkManager jobs during Doze when the device is unplugged and idle, and long-running work needs foreground-service-style user visibility. The product promise should be: "train safely when constraints allow, checkpoint often, and resume later," not "run continuously in the background while the phone sleeps."

### Why This Is Valuable

This is a systems capability, not a new ML algorithm. The contribution is robust local execution under mobile constraints:

- checkpointing,
- interrupted training recovery,
- thermal/energy measurement,
- user-visible progress,
- privacy-preserving local adaptation.

### Spike Gate

Prove that a multi-step training job can:

- start under WorkManager constraints,
- run as long-running work with appropriate notification behavior,
- checkpoint periodically,
- stop cleanly on constraint loss,
- resume from `training_state.json`,
- produce measured energy/thermal traces.

### Implementation Notes

- Add a scheduler wrapper outside `ORTTrainerNative`; do not bury WorkManager inside the core trainer.
- Require the foundation contracts from `00_code_plans/08_training_lifecycle_and_checkpoint_contracts.md`: `TrainingJob`, progress events, checkpoint metadata, cooperative cancellation, and stable cache/package ownership.
- Resume is built on existing primitives: `TrainingState` (`ORTTrainerNative.kt:13-17`), `saveTrainingState`/`loadTrainingState` (`ORTTrainerNative.kt:85-150`), and the `saveSteps` checkpoint write (`ORTTrainerNative.kt:314-317`) that persists scheduler state, `globalStep`, and `epoch` to `training_state.json`. Detailed in `04_code_plans/02`.
- Add `TrainingScheduleConfig`:
  - `requiresCharging`
  - `requiresDeviceIdle`
  - `requiresBatteryNotLow`
  - `maxRuntimeMinutes`
  - `checkpointEverySteps`
  - `notificationTitle`
  - `notificationChannelId`
- Implement training in bounded chunks such as `maxSteps` or `maxRuntimeMinutes`; each chunk must checkpoint and release native resources cleanly.
- Separate download/network work from training work. Use network constraints for model/package downloads, then run training from local files.
- Use a foreground notification for long-running work, with visible progress and a cancel action.
- Decide Android 14+ foreground service type and manifest permissions before shipping; do not leave this to the sample app by accident.
- Add a single model/session lock so a scheduled worker cannot race foreground training, merge, or generation.
- Check thermal state, battery state, and available storage before each chunk.
- Use existing `saveSteps`, `loadFromState`, and scheduler state as the resume mechanism. (Caveat: `CosineLRScheduler` persists/restores today, but `LinearLRScheduler.stateDict()/loadFromState()` are `TODO` at `ORTScheduler.kt:156-161` — `04_code_plans/02` closes that gap before multi-chunk scheduling ships.)
- Expose callbacks for paused/resumed/interrupted/completed.
- Log thermal and battery metrics to the same evaluation style as `docs/mobile_evaluation.md`.

### Tests

- JVM/unit tests for config mapping.
- Android instrumentation test for worker creation.
- Manual device test with short constraints and a tiny dataset.
- Doze/constraint behavior test: verify unplugged idle defers work, charging resumes eligible work, and cancellation checkpoints cleanly.
- Resume test: interrupt after checkpoint, restart, confirm global step advances.
- Measurement export test for energy/thermal logs.

## 3.3 Federated Fine-Tuning With Flower

### Scope

Add a federated adapter fine-tuning spike that lets multiple clients train local MobileTransformers adapters and aggregate updates with Flower.

This should not federate the full base LLM. The first viable target is adapter/trainable tensor exchange:

- LoRA tensors.
- MARS tensors if their tensor names, shapes, and merge semantics are stable.
- Optional encoder/classifier head tensors after 3.1 passes.

### Feasibility Assessment

Feasibility is medium for simulation and early server-side orchestration, but high-risk for direct Android production integration.

What looks favorable:

- Flower can aggregate framework-agnostic arrays through `NumPyClient` or the newer `ArrayRecord`/`Message` APIs.
- ORT Training already defines trainable parameters through `requires_grad` and `frozen_params`.
- This repo already has local on-device training, checkpointing, metrics, and merge/export hooks.
- Federating adapters keeps communication far smaller than federating full LLM weights.

What is not solved by existing frameworks:

- The Flower ONNX Runtime support thread is currently a community discussion/call for PRs, not a maintained integration.
- Flower's Android quickstart is marked historical and warns that the experimental Android SDK is incompatible with the latest Flower.
- MobileTransformers lacks a stable `get_parameters` / `set_parameters` API for trainable adapter tensors.
- Secure aggregation, differential privacy, auth, dropout handling, and user consent are separate product/security gates.

Recommended framing: "Flower-compatible federated adapter experiments" first, not "production federated Android LLM training."

### Integration Model

Use Flower for orchestration, strategies, aggregation, metrics, and simulation. Use MobileTransformers/ORT for local model training.

Target parameter contract:

```text
FederatedAdapterRecord
  schemaVersion
  baseModelId
  mobiletransformersPackageRevision
  peftMethod
  adapterFormatVersion
  round
  tensors:
    - name
      shape
      dtype
      role: adapter | trainable_weight | head
      aggregation: average | weighted_average | server_only
      bytes | fileRef
  metrics:
    numExamples
    numTokens
    trainLoss
    peakMemoryMb
    durationMs
```

Python/Flower mapping:

- `FederatedAdapterRecord.tensors` -> Flower `ArrayRecord`.
- `numExamples` or `numTokens` -> Flower `MetricRecord` key used for weighted aggregation.
- round config such as local epochs, max steps, learning rate, clipping norm, and checkpoint behavior -> Flower `ConfigRecord`.
- `trainLoss`, memory, duration, failures, and device class -> metrics returned to the Flower strategy.

Android mapping:

- Add a `FederatedTrainingRepository` wrapper outside `ORTTrainerNative`.
- Add `exportTrainableTensors()` and `importTrainableTensors()` at the Kotlin/JNI boundary.
- Reuse `TrainingRepository.performTraining(...)` for local work.
- Reuse WorkManager constraints from 3.2 for round scheduling.
- Upload only adapter/trainable tensor updates, never raw user data.

### Recommended Architecture Options

| Option | Description | Feasibility | Recommendation |
| --- | --- | --- | --- |
| A | Python-only Flower simulation using MobileTransformers package artifacts and ORT Training Python | High | First spike. Proves tensor contract, aggregation, and metrics without Android transport risk. |
| B | Python Flower ServerApp plus Android clients connected through a thin MobileTransformers FL gateway/API | Medium | Second spike. Keeps Flower on the server while Android uses a small app-native protocol. |
| C | Direct Flower Android client integration | Low to medium | Defer until Flower's Android SDK is current and compatible. |
| D | Full model-weight federated LLM training | Low | Avoid. Too much bandwidth, storage, and memory for the first extension. |

Option A should be required before any Android work. Option B is likely the most practical path for this repo because it leverages Flower strategies while avoiding a dependency on an unstable Android SDK. Option C should be revisited only if Flower ships a current Android SDK that supports custom tensor payloads and mobile lifecycle constraints.

### Spike Gate

The first federated spike must prove:

- A tiny MobileTransformers-ready model exposes a canonical list of trainable adapter tensors.
- A Python Flower simulation with at least two clients can:
  - receive initial adapter tensors,
  - run local ORT-backed training for one or more steps,
  - return updated tensors and metrics,
  - aggregate with `FedAvg` or a simple custom strategy,
  - save a new global adapter artifact.
- The aggregated adapter can be loaded by the normal inference path and changes output/logits compared with the initial adapter.
- Communication size is measured and bounded.

The second spike must prove:

- One Android client can import global adapter tensors.
- The client can train locally with WorkManager-friendly constraints.
- The client can export updated adapter tensors.
- A server-side Flower wrapper or gateway can aggregate the returned update.

### Implementation Notes

- Start with LoRA. It has the clearest tensor semantics and smaller communication cost.
- Do not aggregate merged base weights. Aggregate adapter/trainable tensors only.
- **Do not invent a new tensor ordering.** `FederatedAdapterRecord` is a thin federated-exchange wrapper over the canonical `TrainableTensorCodec` + `weight_handoff_map.json` already owned by `00_code_plans/07`. Deterministic order, names, dtype, shape, and quantization policy come from there; the adapter tensor names originate from `create_lora_mapping` / `create_mars_adapter_mapping` (`trainer/utils.py:670-703` / `:533-668`) emitted as `peft_mapping` into `training_config.json` (`trainer/builder.py:379-391`). The record adds only: `round`, `aggregation` role per tensor, and `metrics`. Detailed in `04_code_plans/03`.
- Add a `mobiletransformers federated simulate` CLI:

```bash
mobiletransformers federated simulate \
  --package build/packages/tiny-mobile \
  --strategy fedavg \
  --clients 4 \
  --rounds 3 \
  --local-max-steps 2 \
  --output build/federated/tiny-run
```

- Add a server CLI only after the simulation works:

```bash
mobiletransformers federated server \
  --package mobiletransformers/TinyLlama-1.1B-mobile \
  --strategy fedavg \
  --rounds 10 \
  --min-clients 5
```

- Keep Flower dependencies in a separate `federated` extra/dependency group so the core export and Android SDK installs remain lean.
- Consider Flower's differential privacy wrappers and mods only after the adapter tensor path is proven.
- Treat secure aggregation as a future security milestone unless a compatible Flower deployment path is proven with Android clients.

### Privacy And Security Gates

Federated learning does not automatically make updates private. Before any real-user deployment:

- Add explicit user consent for federation participation.
- Use TLS and device/client authentication.
- Decide whether updates are adapter weights, deltas, or gradients.
- Document the leakage risk of model updates.
- Evaluate client-side clipping and local differential privacy.
- Evaluate secure aggregation compatibility with the chosen Flower/Android gateway path.
- Provide opt-out and local-delete behavior.

### Tests

- Tensor codec round-trip: manifest names -> arrays -> imported tensors -> exported tensors.
- Python Flower simulation: two clients, one round, deterministic tiny dataset, `FedAvg` aggregation.
- Multi-round simulation: loss/metric trends are recorded and final adapter is saved.
- Aggregated adapter load smoke in desktop inference.
- Communication-size test for LoRA and MARS tensor payloads.
- Android tensor import/export instrumentation test.
- Android one-client federated round smoke through a local gateway/mock server.
- Failure/dropout test: one client misses a round and aggregation still completes.
- Privacy config validation: federation cannot start without explicit consent/auth configuration.

## 3.4 FunctionGemma And Mobile Actions

### Scope

Personalize an on-device function-calling model that emits validated function calls bound to Android intents or app actions.

### Why This Is High-Visibility

It demonstrates a complete local personalization loop for a practical mobile-agent scenario:

- user-specific actions,
- private local fine-tuning,
- structured local generation,
- real Android intent binding.

### Hard Gates

#### Architecture Gate

FunctionGemma is **Gemma-3**. The repo's footing is asymmetric, and the gate is specifically the **inference graph**:

- **Training export** already detects `Gemma3ForCausalLM` and uses `GemmaOnnxConfig` (`trainer/builder.py:263`).
- **Inference graph builder only handles `GemmaForCausalLM` and `Gemma2ForCausalLM`** (`inference/builder.py:3234-3236`) — there is **no Gemma-3 branch**. The pre-built inference graph (the repo's core OOM-avoidance trick) therefore does not yet exist for Gemma-3.

So the real blocker is adding/validating a Gemma-3 inference-graph export, not the training side. Spike that first.

Pass conditions:

- HF model loads.
- Training artifacts generate (likely already works via `:263`).
- **Inference graph exports for Gemma-3** (the actual gate; delivered as a `Gemma3Model` class registered via the architecture registry's `inference_model_class` — a registry entry, **not** a new `elif` branch; see `04_code_plans/05` + `00_code_plans/09`).
- Android one-step training works or fails with a documented blocker.
- Inference emits valid structured function-call text. Detailed in `04_code_plans/05`.

#### Differentiation Gate

Do not merely reproduce an off-device fine-tuning tutorial. MobileTransformers must show at least two of:

- on-device training,
- personalized per-user action sets,
- local validated tool-call generation,
- Android intent binding,
- privacy-preserving local data.

#### Tool Grammar Gate

If GenAI wins in Tier 0, test whether grammar-constrained or structured generation can validate function calls. If GenAI does not win, implement lightweight JSON/function-call validation around the manual loop and document limitations.

### Implementation Notes

- Create a small synthetic per-user mobile-actions dataset.
- Define an action schema:
  - action name,
  - parameters,
  - allowed Android intent,
  - validation rules,
  - privacy/security classification.
- Bind only harmless demo intents first.
- Never execute arbitrary model output directly. Validate against a local allowlist.
- Add a dry-run mode that returns intended action without executing it.

### Tests

- Export spike test for the chosen FunctionGemma model.
- Training one-step smoke.
- Structured output validation tests.
- Invalid action rejection tests.
- Android intent dry-run test.
- End-to-end demo only after architecture and differentiation gates pass.

## Implementation Sequence

1. Finish Tiers 0-2 and the global foundation phases from `05_cross_cutting_release_modernization.md`.
2. Spike encoder export/training first, because it mostly extends package metadata and avoids decoder KV-cache decisions.
3. Add encoder manifest/config fields and one example task.
4. Add `TrainingJob`, checkpoint metadata, progress events, cancellation, and session-lock contracts if they did not land in the foundation pass.
5. Add WorkManager scheduling spike with a tiny training job, charging/battery constraints, foreground notification, and bounded training chunks.
6. Add measurement logging for scheduled training.
7. Add a trainable-tensor manifest/codec spike for LoRA adapter tensors.
8. Run a Python-only Flower simulation with two or more ORT-backed MobileTransformers clients.
9. If the simulation passes, add Android tensor import/export instrumentation tests.
10. If Android tensor exchange passes, add a one-client gateway/server round before attempting multi-client Android runs.
11. Run FunctionGemma architecture spike only after the inference engine and structured-output validation strategy are settled.
12. If FunctionGemma passes, build validation and intent binding before demo UI.
13. If any gate fails, document as future work and keep v1.0 focused.

## Risks

- Encoder support may expose assumptions in current training wrappers about causal LM inputs and labels.
- WorkManager behavior varies by Android version and OEM battery policy.
- Long-running training can create user trust issues if notifications and controls are unclear.
- Flower's official Android path is not currently a stable dependency target, so direct Android Flower clients may be blocked by upstream SDK status.
- Flower has no official ONNX Runtime integration today; MobileTransformers must supply the ORT tensor adapter layer.
- Federated updates can leak information about local data even if raw examples never leave the device.
- Adapter tensor ordering or dtype mismatches can silently corrupt aggregation unless the manifest and validator are strict.
- Full LLM weight federation is likely too bandwidth-heavy for mobile; the first extension must stay adapter-only.
- Client dropout, slow clients, thermal throttling, and WorkManager constraints can make synchronous rounds slow or unstable.
- FunctionGemma may not export cleanly or may require unsupported ops.
- Tool/action demos can become security-sensitive quickly.
- Reach extensions can distract from the core package/API release.

## Tests And Smokes

- Encoder export smoke for one MiniLM/BERT-style model.
- Encoder one-step desktop training smoke.
- Encoder Android one-step training or documented blocker.
- WorkManager short scheduled-training smoke on a device.
- Doze/charging behavior smoke: verify deferred work does not corrupt state and resumes from checkpoint.
- Checkpoint interruption and resume smoke using `training_state.json`.
- Energy/thermal metrics export for scheduled training.
- Federated tensor codec round-trip for one LoRA package.
- Python Flower simulation with two ORT-backed clients and one `FedAvg` round.
- Multi-round Flower simulation saving aggregated adapter artifacts.
- Aggregated adapter desktop inference smoke.
- Android federated tensor import/export instrumentation smoke.
- One Android client federated gateway/mock-server round.
- Federated privacy/auth config validation smoke.
- FunctionGemma architecture export smoke before any demo work.
- Function-call validation and invalid-action rejection tests.

## Acceptance Criteria

- Encoder support has a completed spike with a pass/fail decision.
- Scheduled training has a measured proof-of-concept and clear Android constraints.
- Federated adapter training has a completed Python simulation spike with a pass/fail decision.
- Android federated participation remains gated behind tensor import/export, WorkManager scheduling, privacy/auth, and Flower gateway feasibility.
- FunctionGemma has documented architecture and differentiation gate results.
- Failed gates become explicit future work, not hidden blockers.
- No Tier 3 work blocks Tier 0-2 release readiness.

## Source Links

- Android WorkManager: https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started/define-work
- Android long-running workers: https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running
- Android Doze and App Standby: https://developer.android.com/training/monitoring-device-state/doze-standby
- Android foreground service types: https://developer.android.com/develop/background-work/services/fgs/service-types
- Hugging Face model uploads: https://huggingface.co/docs/hub/models-uploading
- ONNX Runtime GenAI: https://onnxruntime.ai/docs/genai/
- ONNX Runtime on-device training: https://onnxruntime.ai/docs/get-started/training-on-device.html
- ONNX Runtime training artifact generation: https://onnxruntime.ai/docs/api/python/on_device_training/training_artifacts.html
- Flower ONNX Runtime support discussion: https://discuss.flower.ai/t/onnx-runtime-support/786
- Flower Android quickstart status: https://flower.ai/docs/framework/tutorial-quickstart-android.html
- Flower NumPyClient API: https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html
- Flower ArrayRecord API: https://flower.ai/docs/framework/ref-api/flwr.app.ArrayRecord.html
- Flower strategy usage: https://flower.ai/docs/framework/how-to-use-strategies.html
- Flower FedAvg API: https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html
- Flower differential privacy guide: https://flower.ai/docs/framework/how-to-use-differential-privacy.html
- Secure Aggregation for Federated Learning in Flower: https://arxiv.org/abs/2205.06117
- Flower framework paper: https://arxiv.org/abs/2007.14390
- On-device Federated Learning with Flower: https://arxiv.org/abs/2104.03042
- MobileFineTuner: https://arxiv.org/abs/2512.08211
