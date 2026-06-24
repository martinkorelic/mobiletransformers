# Federated Android Client & Gateway

**Priority #35 | Prerequisites: #34 (`04_code_plans/03`, Python simulation MUST pass first), #33 (`04_code_plans/02`, scheduler), #17 (`00_code_plans/08`) | Blocks: —**

> Tier-3 showcase, Option B. **Gated hard:** do not start until the Option-A simulation (#34) passes. Flower's Android SDK is marked historical/incompatible, so Android uses a thin app-native protocol to a Python ServerApp gateway, not a direct Flower Android client. Must not block v1.0.

## Purpose

Let one Android client import a global adapter, train locally under WorkManager-friendly constraints, export the updated adapter, and have a server-side Flower wrapper/gateway aggregate it — reusing the `FederatedAdapterRecord` codec (#34) and the scheduler (#33). Adapter/trainable tensors only; never raw user data.

## Touched / new files

Kotlin:
- NEW `android/.../federated/FederatedTrainingRepository.kt` — wrapper outside `ORTTrainerNative`; orchestrates import → train → export per round.
- NEW `android/.../federated/AdapterTensorCodec.kt` — Kotlin side of `FederatedAdapterRecord` (mirror of #34's Python codec; same names/order from `weight_handoff_map.json`).
- `android/.../ORTTrainerNative.kt` — add JNI `exportTrainableTensors()` / `importTrainableTensors()` at the Kotlin/JNI boundary (new; none exist today). Reuse merge/export hooks (`mergeExportSessionWeights`, `:587-589`).
- `android/.../repository/TrainingRepository.kt` — `performTraining` (`:9-22`) reused for the local round.
- `android/.../scheduler/*` (#33) — round scheduling reuses the WorkManager constraints.

Python (server):
- NEW `src/mobiletransformers/federated/gateway.py` — thin HTTP/gRPC gateway translating Android `FederatedAdapterRecord` payloads ↔ Flower `ServerApp`.
- NEW CLI `mobiletransformers federated server --package ... --strategy fedavg --rounds 10 --min-clients 5`.

## Data contracts / interfaces

- **Wire format** = `FederatedAdapterRecord` (#34), serialized identically on both sides; the Kotlin codec and Python codec must produce byte-identical tensor ordering (golden cross-language test).
- **JNI boundary**:
  ```kotlin
  external fun exportTrainableTensors(session: Long, handoffMapPath: String): /* serialized record */ ByteArray
  external fun importTrainableTensors(session: Long, handoffMapPath: String, record: ByteArray): Boolean
  ```
  Names/dtypes/order come from `weight_handoff_map.json` (#7) — fail closed on mismatch.

### Privacy & security gates (must pass before any real-user deployment)

- Explicit user consent for participation (federation cannot start without it).
- TLS + device/client authentication to the gateway.
- Decide update kind (adapter weights / deltas / gradients); document leakage risk.
- Evaluate client-side clipping / local DP (Flower DP is preview).
- Secure aggregation only if proven against the chosen gateway transport.
- Opt-out + local-delete behavior.

## Implementation steps

1. Implement `exportTrainableTensors`/`importTrainableTensors` JNI + Kotlin `AdapterTensorCodec`; cross-language golden test vs. #34.
2. `FederatedTrainingRepository`: import global adapter → `performTraining` (local, bounded) → export update; reuse the scheduler (#33) for round timing.
3. `gateway.py` + server CLI: translate Android payloads to Flower `ServerApp`; aggregate with `FedAvg`.
4. Wire the privacy/auth config; refuse to start a round without consent + auth configured.
5. One-client end-to-end round through a local gateway/mock server before any multi-client Android run.

## Interactions

- **#34 (codec/sim)**: provides the record format + server-side strategies; this is the Android/gateway half.
- **#33 (scheduler)**: round scheduling + constraints.
- **#7/#8**: tensor names/order/checksums; fail-closed import.
- **#17**: session lock / lifecycle for the local round.

## Tests & smokes

- Cross-language codec golden: Kotlin export == Python `FederatedAdapterRecord.from_dict` (count, names, dtype, order).
- Android tensor import/export instrumentation smoke.
- One-client federated round through a local gateway/mock server; aggregation succeeds.
- Privacy/auth config validation: round cannot start without consent + auth.
- Communication-size measurement for the LoRA payload on-device.
- Failure/dropout: client misses a round, server still aggregates.
