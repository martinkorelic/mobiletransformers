# Federated Adapter Codec & Python Flower Simulation

**Priority #34 | Prerequisites: #7 (`00_code_plans/07_weight_handoff_map_and_tensor_codec.md`), #8 (`01_code_plans/01_unified_merger_and_external_data_export.md`), #12 (`00_code_plans/06`) | Blocks: #35 (`04_code_plans/04`, Android gateway)**

> Tier-3 showcase. **Option A (Python-only simulation) is required before any Android work.** Must not block v1.0. Framing: "Flower-compatible federated adapter experiments," not "production federated Android LLM training."

## Purpose

Prove a federated adapter-aggregation loop in a Python Flower simulation over ORT-backed MobileTransformers clients, exchanging **only adapter/trainable tensors** (LoRA first). The exchange record is a thin wrapper over the existing `TrainableTensorCodec` + `weight_handoff_map.json` (#7) — **no new tensor ordering is invented.**

## Touched / new files

Python:
- NEW `src/mobiletransformers/federated/adapter_record.py` — `FederatedAdapterRecord` built on `TrainableTensorCodec` (#7); serialize/deserialize adapter tensors.
- NEW `src/mobiletransformers/federated/flower_client.py` — `NumPyClient`/`ArrayRecord`-based ORT client (`get_parameters`/`fit`/`evaluate`).
- NEW `src/mobiletransformers/federated/flower_sim.py` — `FedAvg` simulation driver.
- NEW `src/mobiletransformers/cli/federated.py` — `mobiletransformers federated simulate` (wired into the CLI from `02_code_plans/05`).
- Reuse `trainer/utils.py` `create_lora_mapping` (`:670-703`) / `create_mars_adapter_mapping` (`:533-668`) and `peft_mapping` from `training_config.json` (`trainer/builder.py:379-391`) as the tensor-name source.
- Keep Flower deps in a separate `federated` extra/dependency group (lean core).

## Data contracts / interfaces

### `FederatedAdapterRecord` (wrapper over the canonical codec)

```text
FederatedAdapterRecord
  schemaVersion
  baseModelId
  mobiletransformersPackageRevision
  peftMethod                 # lora | mars
  adapterFormatVersion       # == weight_handoff_map schemaVersion
  round
  tensors:                   # ORDER + names + dtype + shape come from TrainableTensorCodec (#7)
    - name, shape, dtype
      role: adapter | trainable_weight | head
      aggregation: average | weighted_average | server_only
      bytes | fileRef
  metrics: { numExamples, numTokens, trainLoss, peakMemoryMb, durationMs }
```

### Flower mapping

- `tensors` → Flower `ArrayRecord` (order fixed by the codec).
- `numExamples` / `numTokens` → `MetricRecord` key for weighted aggregation.
- round config (local epochs, max steps, lr, clip norm, checkpoint) → `ConfigRecord`.
- `trainLoss`, memory, duration, failures, device class → returned metrics.

### Architecture options (Option A here)

| Option | Description | Feasibility |
| --- | --- | --- |
| A | Python-only Flower sim using package artifacts + ORT Training Python | High — **this plan** |
| B | Python ServerApp + Android clients via a thin MT FL gateway | Medium — #35 |
| C | Direct Flower Android client | Low — defer until Flower's Android SDK is current |
| D | Full base-weight LLM federation | Low — avoid (bandwidth/memory) |

## Implementation steps

1. `FederatedAdapterRecord` over `TrainableTensorCodec`: deterministic tensor order from the handoff map; round-trip serialize.
2. `flower_client.py`: load a tiny MobileTransformers-ready package, expose `get_parameters` (export adapter tensors), `fit` (local ORT train ≥1 step), `evaluate`.
3. `flower_sim.py`: ≥2 clients, `FedAvg`, save a new global adapter artifact per round.
4. CLI: `mobiletransformers federated simulate --package ... --strategy fedavg --clients 4 --rounds 3 --local-max-steps 2 --output ...`.
5. Measure and bound communication size per round (LoRA payload).
6. Confirm the aggregated adapter loads in the normal desktop inference path and changes logits vs. the initial adapter.

## Interactions

- **#7 / #8 (codec/merger)**: the single source of tensor names/order/dtype; the record is a wrapper.
- **#12 (manifest)**: `baseModelId` / package revision come from the manifest.
- **#35 (Android gateway)**: consumes this record + simulation as the server side.
- **#32 (encoder)**: optional head tensors become federatable after encoder support.

## Tests & smokes

- Tensor codec round-trip: manifest names → arrays → record → arrays (byte-identical, deterministic order).
- 2-client `FedAvg` round on a deterministic tiny dataset; aggregation completes.
- Multi-round sim: loss/metric trend recorded; global adapter artifact saved each round.
- Aggregated-adapter desktop inference smoke: logits differ from initial adapter.
- Communication-size test for LoRA (and MARS) payloads.
- Dropout test: one client misses a round, aggregation still completes.
