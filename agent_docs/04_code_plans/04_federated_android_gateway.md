# Federated Android Client & Gateway

**Priority #36 | Prerequisites: #35 (`04_code_plans/03`, Python simulation MUST pass first), #34 (`04_code_plans/02`, scheduler), #18 (`00_code_plans/08`) | Blocks: —**

> Tier-3 showcase, Option B. **Gated hard:** do not start until the Option-A simulation (#35) passes. Flower's Android SDK is marked historical/incompatible, so Android uses a thin app-native protocol to a Python ServerApp gateway, not a direct Flower Android client. Must not block v1.0.

## Purpose

Let one Android client import a global adapter, train locally under WorkManager-friendly constraints, export the updated adapter, and have a server-side Flower wrapper/gateway aggregate it — reusing the `FederatedAdapterRecord` codec (#35) and the scheduler (#34). Adapter/trainable tensors only; never raw user data.

## Touched / new files

Kotlin:
- NEW `android/.../federated/FederatedTrainingRepository.kt` — wrapper outside `ORTTrainerNative`; orchestrates import → train → export per round.
- NEW `android/.../federated/AdapterTensorCodec.kt` — Kotlin side of `FederatedAdapterRecord` (mirror of #35's Python codec; same names/order from `weight_handoff_map.json`).
- `android/.../ORTTrainerNative.kt` — add JNI `exportTrainableTensors()` / `importTrainableTensors()` at the Kotlin/JNI boundary (new; none exist today). Reuse merge/export hooks (`mergeExportSessionWeights`, `:587-589`).
- `android/.../repository/TrainingRepository.kt` — `performTraining` (`:9-22`) reused for the local round.
- `android/.../scheduler/*` (#34) — round scheduling reuses the WorkManager constraints.

Python (server):
- NEW `src/mobiletransformers/federated/gateway.py` — thin HTTP/gRPC gateway translating Android `FederatedAdapterRecord` payloads ↔ Flower `ServerApp`.
- NEW CLI `mobiletransformers federated server --package ... --strategy fedavg --rounds 10 --min-clients 5`.

## Data contracts / interfaces

- **Wire format** = `FederatedAdapterRecord` (#35), using #35's **pinned byte serialization** (length-prefixed JSON header + concatenated raw little-endian tensor payloads in codec order — see #35 §Byte serialization); the Kotlin codec and Python codec must produce byte-identical output (golden cross-language test).
- **JNI boundary**:
  ```kotlin
  external fun exportTrainableTensors(session: Long, handoffMapPath: String): /* serialized record */ ByteArray
  external fun importTrainableTensors(session: Long, handoffMapPath: String, record: ByteArray): Boolean
  ```
  Names/dtypes/order come from `weight_handoff_map.json` (#8) — fail closed on mismatch.

The Kotlin `AdapterTensorCodec` is a **mirror** of #35's Python `TrainableTensorCodec`, not an independent implementation (F8): both read tensor names/order/dtype from the same `weight_handoff_map.json`, so neither side ever invents an ordering, and the Kotlin `adapterFormatVersion` equals `weight_handoff_map.schemaVersion`. A checked-in golden vector enforces that the two codecs serialize byte-identically; drift fails the parity test (cf. F2-style mirror checks).

### Privacy & security gates (must pass before any real-user deployment)

- Explicit user consent for participation (federation cannot start without it).
- TLS + device/client authentication to the gateway.
- Decide update kind (adapter weights / deltas / gradients); document leakage risk.
- Evaluate client-side clipping / local DP (Flower DP is preview).
- Secure aggregation only if proven against the chosen gateway transport.
- Opt-out + local-delete behavior.

## Implementation steps

1. Implement `exportTrainableTensors`/`importTrainableTensors` JNI + Kotlin `AdapterTensorCodec`; cross-language golden test vs. #35.
2. `FederatedTrainingRepository`: import global adapter → `performTraining` (local, bounded) → export update; reuse the scheduler (#34) for round timing.
3. `gateway.py` + server CLI: translate Android payloads to Flower `ServerApp`; aggregate with `FedAvg`.
4. Wire the privacy/auth config; refuse to start a round without consent + auth configured.
5. One-client end-to-end round through a local gateway/mock server before any multi-client Android run.

## Interactions

- **#35 (codec/sim)**: provides the record format + server-side strategies; this is the Android/gateway half.
- **#34 (scheduler)**: round scheduling + constraints.
- **#8/#9**: tensor names/order/checksums; fail-closed import.
- **#18**: session lock / lifecycle for the local round.

## References

- Flower `ServerApp` reference (gateway server side): https://flower.ai/docs/framework/ref-api/flwr.serverapp.ServerApp.html
- Secure Aggregation (SecAgg+) / DP example (Option B privacy gate): https://flower.ai/docs/examples/flower-secure-aggregation.html
- Flower Intelligence Kotlin SDK status (immature → defer direct client, confirms Option B): https://flower.ai/blog/2025-05-14-flower-intelligence-kotlin-sdk/

## Worked example

The two JNI entry points and a single repository round (import → constrained local train → export):

```kotlin
// ORTTrainerNative.kt — new JNI boundary (none exists today); reuses mergeExportSessionWeights :587-589
external fun exportTrainableTensors(session: Long, handoffMapPath: String): ByteArray
external fun importTrainableTensors(session: Long, handoffMapPath: String, record: ByteArray): Boolean
```

```kotlin
// FederatedTrainingRepository.kt — one federated round under WorkManager constraints (#34)
suspend fun runRound(global: ByteArray): ByteArray {
    require(native.importTrainableTensors(session, handoffMapPath, global)) { "adapter import mismatch" }
    repository.performTraining(maxSteps = roundConfig.maxSteps)        // local, bounded; charging/idle (#34)
    return native.exportTrainableTensors(session, handoffMapPath)      // updated adapter only, codec order
}
```

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- Cross-language codec golden (`AdapterTensorCodecTest.kt`): the Kotlin `AdapterTensorCodec` serializes a fixture to bytes identical to a checked-in golden produced by Python `FederatedAdapterRecord.from_dict` (assert count, names, dtype, order — the F8 mirror parity).
- Privacy/auth config validation (`FederatedConfigTest.kt`): a round refuses to start unless consent + gateway auth (TLS/client auth) are configured.
- Plus the module **compiles** (`./gradlew :MobileTransformers:compileDebugKotlin`).

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `pytest tests/federated/test_gateway_dropout.py` — the `gateway.py` + `ServerApp` aggregates canned `FederatedAdapterRecord` payloads with one client missing; aggregation still completes and writes a global adapter.

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- Android tensor import/export instrumentation smoke (`importTrainableTensors`/`exportTrainableTensors` round-trip on a real session).
- One-client federated round through a local gateway/mock server: import global adapter → bounded local `performTraining` (under #34 constraints) → export → server aggregation succeeds.
- Communication-size measurement for the LoRA payload on-device.

**Definition of done** — the Kotlin `AdapterTensorCodec` and JNI `import/exportTrainableTensors` round-trip byte-identically with the Python codec against a checked-in golden (names/order/dtype from `weight_handoff_map.json`, fail-closed on mismatch); `FederatedTrainingRepository` runs a one-client round (import → bounded local train → export) through a local gateway/mock server and the `ServerApp` aggregates with `FedAvg` (including a dropped client); a round cannot start without consent + gateway TLS/auth; and the on-device LoRA payload size is measured. Only adapter/trainable tensors leave the device — never raw user data.
