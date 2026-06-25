# Federated Adapter Codec & Python Flower Simulation

**Priority #35 | Prerequisites: #8 (`00_code_plans/07_weight_handoff_map_and_tensor_codec.md`), #9 (`01_code_plans/01_unified_merger_and_external_data_export.md`), #13 (`00_code_plans/06`) | Blocks: #36 (`04_code_plans/04`, Android gateway)**

> Tier-3 showcase. **Option A (Python-only simulation) is required before any Android work.** Must not block v1.0. Framing: "Flower-compatible federated adapter experiments," not "production federated Android LLM training."

## Purpose

Prove a federated adapter-aggregation loop in a Python Flower simulation over ORT-backed MobileTransformers clients, exchanging **only adapter/trainable tensors** (LoRA first). The exchange record is a thin wrapper over the existing `TrainableTensorCodec` + `weight_handoff_map.json` (#8) — **no new tensor ordering is invented.**

## Touched / new files

Python:
- NEW `src/mobiletransformers/federated/adapter_record.py` — `FederatedAdapterRecord` built on `TrainableTensorCodec` (#8); serialize/deserialize adapter tensors.
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
  tensors:                   # ORDER + names + dtype + shape come from TrainableTensorCodec (#8)
    - name, shape, dtype
      role: adapter | trainable_weight | head
      aggregation: average | weighted_average | server_only
      bytes | fileRef
  metrics: { numExamples, numTokens, trainLoss, peakMemoryMb, durationMs }
```

The record never invents its own tensor list (F8): `tensors` names/order/dtype are read straight from `TrainableTensorCodec` + `weight_handoff_map.json` (#8), and `adapterFormatVersion` **equals** `weight_handoff_map.schemaVersion` — a record whose `adapterFormatVersion` doesn't match the codec it was built from fails closed. The record's own `schemaVersion` follows the F1 contract (`"MAJOR.MINOR"` + `minReaderVersion`): readers preserve unknown fields (additive minor bumps are non-breaking) and reject only when `major` exceeds support or `minReaderVersion` is unmet, via the one shared `check_compat()` helper.

### Flower mapping

- `tensors` → Flower `ArrayRecord` (order fixed by the codec).
- `numExamples` / `numTokens` → `MetricRecord` key for weighted aggregation.
- round config (local epochs, max steps, lr, clip norm, checkpoint) → `ConfigRecord`.
- `trainLoss`, memory, duration, failures, device class → returned metrics.

### Architecture options (Option A here)

| Option | Description | Feasibility |
| --- | --- | --- |
| A | Python-only Flower sim using package artifacts + ORT Training Python | High — **this plan** |
| B | Python ServerApp + Android clients via a thin MT FL gateway | Medium — #36 |
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

- **#8 / #9 (codec/merger)**: the single source of tensor names/order/dtype; the record is a wrapper.
- **#13 (manifest)**: `baseModelId` / package revision come from the manifest.
- **#36 (Android gateway)**: consumes this record + simulation as the server side.
- **#33 (encoder)**: optional head tensors become federatable after encoder support.

## References

- Flower framework docs (root): https://flower.ai/docs/framework/
- `run_simulation` (Option A in-process sim): https://flower.ai/docs/framework/how-to-run-simulations.html
- `Message`/`RecordDict`/`ArrayRecord` reference: https://flower.ai/docs/framework/ref-api/flwr.app.Message.html
- `parameters_to_ndarrays` (framework-agnostic NumPy): https://flower.ai/docs/framework/ref-api/flwr.common.parameters_to_ndarrays.html
- `FedAvg` strategy reference: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html
- Modern Message API (upgrade guide): https://flower.ai/docs/framework/how-to-upgrade-to-message-api.html

## Worked example

A minimal Flower `ClientApp` that drives the existing ORT training loop and exchanges only adapter ndarrays:

```python
# flower_client.py — modern Message API
from flwr.client import ClientApp
from flwr.common import ArrayRecord, parameters_to_ndarrays

app = ClientApp()

@app.train()
def train(msg, ctx):
    arrays = parameters_to_ndarrays(msg.content["arrays"])     # incoming global adapter (codec order, #8)
    record = FederatedAdapterRecord.from_ndarrays(arrays)      # names/order/dtype from weight_handoff_map.json
    updated = run_ort_training(record, max_steps=2)            # existing ORT Training Python loop
    return msg.create_reply(content={
        "arrays": ArrayRecord(updated.to_ndarrays()),          # only adapter/trainable tensors back
        "metrics": {"numExamples": updated.num_examples, "trainLoss": updated.loss},
    })
```

```python
# flower_sim.py — Option A in-process simulation
run_simulation(server_app=server_app, client_app=app, num_supernodes=4,
               backend_config={"client_resources": {"num_cpus": 1}})   # FedAvg server_app
```

```bash
mobiletransformers federated simulate --package ./pkg --strategy fedavg \
    --clients 4 --rounds 3 --local-max-steps 2 --output ./global_adapter
```

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `pytest tests/federated/test_codec_roundtrip.py` — manifest names → arrays → `FederatedAdapterRecord` → arrays is byte-identical with deterministic order (the tensor list comes from `TrainableTensorCodec` + `weight_handoff_map.json`, not the record).
- `pytest tests/federated/test_format_version.py` — `adapterFormatVersion == weight_handoff_map.schemaVersion`; a mismatched version fails closed via `check_compat()` (F1/F8).
- `pytest tests/federated/test_comm_size.py` — serialized LoRA (and MARS) payload size is measured and bounded.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `pytest tests/federated/test_fedavg_aggregation.py` — feed canned per-client adapter ndarrays (no training) into `FedAvg`; assert the aggregated tensors equal the (weighted) mean and the global artifact is written.
- `pytest tests/federated/test_dropout.py` — one client's reply is missing; aggregation still completes over the remaining clients.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- 2-client `FedAvg` round on a deterministic tiny dataset that actually runs ORT `fit` (requires the source-built ORT Training Python wheel).
- Multi-round sim: loss/metric trend recorded; a new global adapter artifact is saved each round.
- Aggregated-adapter desktop inference smoke: logits differ from the initial adapter.

**Workflow (end-to-end)** — *(CHECKPOINT #35, user-run)* run an N-client Flower simulation over ORT-backed clients exchanging only adapter tensors: `mobiletransformers federated simulate --clients 4 --rounds 3 --local-max-steps 2`, where each `@app.train()` client runs the existing ORT training loop and returns updated adapter ndarrays in an `ArrayRecord`; assert that `FedAvg` aggregation runs each round and that the evaluation metric improves versus round 0.

**Definition of done** — `FederatedAdapterRecord` round-trips byte-identically with codec-derived tensor order and an `adapterFormatVersion` equal to the handoff-map `schemaVersion`; a `run_simulation` driver with ≥2 ORT-backed clients and `FedAvg` exchanges **only** adapter/trainable tensors, completes aggregation each round (including a dropped client), saves a new global adapter artifact per round, and shows the metric improving versus round 0; communication size per round is bounded; and the aggregated adapter loads in the normal desktop inference path and changes logits versus the initial adapter.
