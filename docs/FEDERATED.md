# Federated adapters

Federated fine-tuning exchanges **adapter tensors only** — never base weights — between clients and a
server that averages them. Status: the record codec and aggregation are implemented and tested; the
Flower simulation is a manual leg, and the Android gateway is not started.

## Why Flower is not a dependency

`flwr[simulation]` pulls Ray, pyarrow and a protobuf line that conflicts with this repo's: adding it
downgraded protobuf 7→6 and rich/typer repo-wide, and bumped mypy 1.11→1.19 (which broke the type gate).
So it stays **out of `uv.lock`**, exactly like the source-built ORT-training wheel:

```bash
pip install "flwr[simulation]"     # out-of-band, into the environment you run the sim from
```

Everything except the simulation driver works without it — deliberately, so the parts that matter are
CI-covered rather than gated behind an optional install.

## The record

One round's contribution is a `FederatedAdapterRecord`: a header plus tightly-packed tensor payloads.

```
uint32 LE header length | JSON header | payload (tensors in codec order)
```

The tensor order is **not invented here** — it comes from `TrainableTensorCodec` /
`weight_handoff_map.json` ([MODEL_FORMAT.md](MODEL_FORMAT.md)), so a record and the package it was
trained against agree by construction. The byte layout is pinned by a committed golden
(`tests/federated/fixtures/federated_record.golden.bin`); regenerate with
`python -m tests.federated.gen_serialization_golden`.

`deserialize` is fail-closed on untrusted input: the schema is version-gated (`check_compat`) *before*
any offset is used, and every tensor's `byteOffset`/`byteLength` is bounds-checked and cross-checked
against its declared dtype × shape.

## Aggregation

`federated_average` is a weighted mean by `numExamples`, in codec order. Dropped clients (`None`
entries) are skipped and the round still completes over the survivors; it fails closed when no client
survived, when survivors disagree on tensor count, or when total weight is zero.

`aggregate_round` is the whole server side of a round minus the messaging: aggregate → wrap in a record
→ write `global_adapter_round<N>.mtfed`. It is pure, so round semantics are unit-tested without Flower.

## Running a simulation

```bash
mobiletransformers federated simulate \
  --package <installed-package-dir> \
  --output  build/federated \
  --clients 4 --rounds 3 --local-max-steps 2
```

Requires `flwr` (above) **and** the ORT-training runtime for real client fitting. Each round writes one
`global_adapter_round<N>.mtfed` into `--output`; the command fails if a run produces none.

Only `fedavg` is supported in v1 — another strategy is rejected rather than silently falling back.

## Open items

- The real N-client simulation with ORT `fit` and an aggregated-adapter logits-differ smoke is a
  **manual** leg (no device needed, but it needs the out-of-band `flwr` + the training profile).
- The **role vocabulary** is **decided (2026-08-08): the codec's `{weight, weight_quantized, scale,
  zero_point}` is normative**, and the tier doc was amended to match the code rather than the reverse.
  The `{adapter, trainable_weight, head}` set was never implemented by anything. Consequence:
  `federated_record.golden.bin` is **unchanged**, and #36 mirrors one vocabulary instead of translating
  between two. #36 is therefore **no longer gated** on this.
- Still open, and a design constraint rather than a nit: v1 exchanges **merged-weight-shaped** tensors
  (`aggregation_role="merged_base_plus_adapter"`), so per-round traffic is the size of the adapted
  weights, not of the rank-r adapters — and that reads against the tier doc's "do not aggregate merged
  base weights". Whether v2 switches to adapter-delta exchange is an open decision.
- `aggregation` has exactly one v1 value, `weighted_average`; unknown values are now rejected on read.
