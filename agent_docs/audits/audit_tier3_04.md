# Tier 3 — 04_code_plans audit

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


Audited read-only against the `restructure` branch (HEAD `54e0a8e`) on 2026-08-07.
Evidence is `path:line` in the repo root `/home/martin/Documents/Projects/Development/LLM_finetuning/mobiletransformers`.
Only automated check run: `.venv/bin/python -m pytest tests/federated -q` → **13 passed**. No installs, no Gradle, no `uv sync`.

## Summary table

| # | Plan | Claimed | Verified | % | Verdict |
| --- | --- | --- | --- | --- | --- |
| 33 | `01_encoder_model_support.md` | box `[ ]`, no claim | `BertModel` arch row + `TaskType.FEATURE_EXTRACTION` + `feature-extraction` pooling path exist; **no `TASK_REGISTRY`, no head, no manifest fields, no `tests/encoder/`, no spike** | **15%** | partial (prereq footholds only) |
| 34 | `02_training_scheduler_workmanager.md` | box `[ ]`; HANDOFF/order say "blocked on `LinearLRScheduler` TODO" | TODO **still open** (`ORTScheduler.kt:157,161`); **no `scheduler/` package at all**; no FGS manifest; no tests | **8%** | not-started-by-design (only #18 seams) |
| 35 | `03_federated_codec_and_python_simulation.md` | "**code-complete** 2026-07-15, box open pending manual leg" | Codec + FedAvg math + CLI + 13 passing tests + byte golden are **real**; the Flower half (`build_server_app`, `run_local_training_step`) is a **stub** that cannot satisfy the DoD even with `flwr` installed | **60%** | partial (over-claimed as code-complete) |
| 36 | `04_federated_android_gateway.md` | box `[ ]`, hard-gated on #35 | **Zero** Android/Python code (`grep TrainableTensors` → 0 hits; no `gateway.py`; no `federated server` CLI). Only #35's byte golden pre-freezes the wire format | **2%** | not-started-by-design |
| 37 | `05_functiongemma_architecture_gate_and_intents.md` | box `[ ]`, gate hard | `Gemma3ForCausalLM` registry row exists with `inference_model_class=None` (gate expressed as data); **no `Gemma3Model`**, no `tools/functiongemma/`, no Kotlin `agent/` | **5%** | not-started-by-design (gate open) |

**Tier 3 overall ≈ 18%**, and that is dominated by #35. Tiers' "spike-gated, never blocks v1.0" framing holds: #34/#36/#37 are legitimately NOT STARTED by design; #33 has incidental footholds from #7/#9's registry work; only #35 was actually worked.

---

## Per-plan findings

### #33 — Encoder-model support (`04_code_plans/01`)

**Required:** `TASK_REGISTRY`/`TaskSpec` rows for `feature-extraction` / `text-classification` / `similarity` (with `onnx_config_class`, `auto_model_class`, `default_head`, `pooling`); a `bert` `ArchitectureSpec`; NEW `trainer/classification_head.py`; `EncoderTaskConfig` manifest block (`taskType`/`pooling`/`numLabels`/`labelMap`/`embeddingDimension`); classification post-process in `inference.cpp:185-245`; verified MARS→encoder mapping; four test files under `tests/encoder/` (`test_mapping.py`, `test_no_kv_cache.py`, `test_task_registry.py`, `test_embedding_infer.py`); the export/train-step/device spike.

**Verified present:**
- `src/mobiletransformers/config/registry/architecture.py:90-96` — `BertModel` `ArchitectureSpec(BertOnnxConfig, ("query","value"), attention_module_name="attention", task=TaskType.FEATURE_EXTRACTION)`.
- `src/mobiletransformers/config/constants.py:72-75` — `TaskType` = `{TEXT_GENERATION, FEATURE_EXTRACTION}` only.
- `trainer/builder.py:269-276` — registry-driven dispatch; `if spec.task == TaskType.FEATURE_EXTRACTION:` selects the no-`use_past` `OnnxConfig` (the plan's "no new `elif`" for the *architecture* half is satisfied).
- `trainer/builder.py:261-264` — `AutoModelForCausalLM` vs `AutoModel` `task_type` branch (pre-existing).
- `trainer/builder.py:379,414` — `feature-extraction` + `add_pooling` path; `trainer/embedding_builder.py` present.
- `src/mobiletransformers/export/registry.py:32-37` — `TASK_PREFERENCE` includes `feature-extraction`, `sentence-similarity`.
- `tests/unit/test_registries.py:33,38-41` — `BertModel` is covered by the legacy-arch parametrization.

**Gaps:**
- (a) **No `TASK_REGISTRY` / `TaskSpec` anywhere** — `grep TASK_REGISTRY src/` → 0 hits. `TaskType` has no `TEXT_CLASSIFICATION`/`SIMILARITY` member, so the plan's closed task set does not exist; `auto_model_class`/`default_head`/`pooling` are not data anywhere.
- (a) **No `trainer/classification_head.py`** (`ls trainer/` → builder, embedding_builder, merge_validator, utils, validator).
- (a) **No `EncoderTaskConfig`** — `grep -rn "numLabels|labelMap|embeddingDimension|taskType" src/mobiletransformers/artifacts/ schemas/` → 0 hits. `schemas/` holds only Generation/Rag/Training config schemas.
- (a) **No classification post-process** on the native embedding path.
- (a) **No `tests/encoder/` directory**; none of the four required tests exist.
- (a) **MARS transfer never verified** — the duplicated tables the plan wanted de-duplicated still exist verbatim: `peft_models/mars/utils.py:1` and `peft_models/ablation/utils.py:1`, still consumed at `peft_models/mars/model.py:380-383` and `peft_models/ablation/model.py:178-181`. Encoder `target_modules` live only in the arch registry and are not wired into those tables.
- (d) The spike itself (export → artifacts → desktop step → device step → metric) is un-run, spike-gated, unowned.

**Drift / doubtful claims:**
- `trainer/builder.py:287-291` hardcodes `LoraConfig(..., task_type="CAUSAL_LM")` on every PEFT branch — an encoder run would build a causal-LM PEFT wrapper. This directly contradicts the encoder path and is not flagged anywhere.
- Registry row drift vs the plan's worked example: plan sketches `target_modules=["query","key","value","dense"]`, `attention_module_name="attention.self"`; code has `("query","value")` / `"attention"`. Small, but it is exactly the "MARS assumptions become registry data" contract, and nothing validates it.
- Registry key is `BertModel` (an `architectures[0]` value), so `AutoModelForSequenceClassification` variants (`BertForSequenceClassification`) would fail closed in `resolve_architecture` — no `text-classification` architecture is reachable.

**Cheapest next step (host-only, no device):** add `tests/encoder/test_task_registry.py` + a `TASK_REGISTRY` with the three task rows and the missing `TaskType` members, then a `tests/encoder/test_mapping.py` that joins the encoder `target_modules` against a synthetic handoff map. Everything up to "one desktop train step" is host-doable; only the Android one-step smoke needs a device.

---

### #34 — Charging-cycle training scheduler (`04_code_plans/02`) · checkpoint

**Required:** `LinearLRScheduler.stateDict()/loadFromState()` implemented **first**; NEW `scheduler/TrainingWorker.kt`, `scheduler/TrainingScheduler.kt`, `scheduler/TrainingScheduleConfig.kt`; FGS type + permissions declared in the **library** manifest; a shared session lock; `TrainingScheduleConfigTest.kt`, `ORTSchedulerTest.kt`, `SessionLockTest.kt`, a `training_state.json` resume parse test; thermal/energy traces.

**Verified present (prereq seams only):**
- `.../training/TrainingJobManager.kt:11-14` — `TrainingJobSpec(repoId, config)`, doc-commented as "the seam a future WorkManager `Worker` (#34) uses… **no WorkManager dependency is added**"; `:20-31` one `TrainingJob` per sanitized repo id.
- `.../training/TrainingJob.kt:34,43` + `ORTTrainerNative.kt:47,224,254` — cooperative `cancelRequested`.
- `ORTTrainerNative.kt:14,127,206-209` — `TrainingState.schedulerState` is saved via `scheduler.stateDict()` and restored via `scheduler!!.loadFromState(...)`; `:194` bounds `totalSteps` by `trainingConfig.maxSteps`.
- `androidx.work` **is already a library dependency** — `MobileTransformers/build.gradle.kts:88` (`libs.androidx.work.runtime.ktx`, version `2.9.1` at `gradle/libs.versions.toml:18,45`), used by `hub/PackageDownloadWorker.kt` (a working `CoroutineWorker` precedent for downloads).

**Gaps:**
- (a) **The blocking TODO is still open**: `ORTScheduler.kt:157` (`loadFromState`) and `:161` (`stateDict`) are literally `TODO("Not yet implemented")` inside `LinearLRScheduler` (`:122-163`). Cosine implements both (`:79`, `:111-120`). Because `ORTTrainerNative.kt:127` calls `scheduler.stateDict()` unconditionally on every checkpoint write, a **linear-schedule run throws `NotImplementedError` at the first checkpoint today** — this is worse than the plan's framing ("would restart the schedule"): it is a hard crash on the save path, not silent drift.
- (a) **No `scheduler/` package** — `find … -iname "*Scheduler*"` returns only `ORTScheduler.kt` and `constants/SchedulerType.kt`. No `TrainingWorker`, `TrainingScheduler`, `TrainingScheduleConfig`.
- (a) **No FGS declaration** — `MobileTransformers/src/main/AndroidManifest.xml` is an empty `<manifest>` element; the app manifest declares only activities. No `FOREGROUND_SERVICE` / `FOREGROUND_SERVICE_DATA_SYNC` permission, no service type.
- (a) **No session lock exists to reuse.** `grep synchronized|Mutex|ReentrantLock|Semaphore` over the library finds only `ORTVectorDatabase.kt:58,73` (instance-cache double-checked locking). Nothing guards train/merge/generate mutual exclusion.
- (a) No `ORTSchedulerTest.kt` / `TrainingScheduleConfigTest.kt` / `SessionLockTest.kt` (`src/test/**` has 20+ tests, none scheduler-related).
- (c)/(d) Doze/constraint/thermal/energy legs are device-only and correctly deferred.

**Drift / doubtful claims:**
- IMPLEMENTATION_ORDER.md:306 self-check for #18 is marked `[x]`: *"Are the session lock + cooperative cancellation defined for reuse by the scheduler (#34)?"* — the parenthetical only evidences `cancelRequested` + `TrainingJobManager` + `TrainingJobSpec`. **There is no session lock in the tree.** #34 step 4 and `SessionLockTest.kt` therefore have no prerequisite to build on; #34's cost is higher than the order file implies.
- HANDOFF's "#34 blocked on a `LinearLRScheduler` TODO" is accurate but under-states scope: closing the TODO is ~30 lines; the plan's other five deliverables are untouched.

**Cheapest next step (host-only, no device):** implement `LinearLRScheduler.stateDict()/loadFromState()` mapping `baseLr/startFactor/endFactor/totalIters` onto the existing cosine-shaped `SchedulerState` fields (additive only — do not change `training_state.json`), plus `ORTSchedulerTest.kt` asserting Linear and Cosine both reproduce the uninterrupted LR after a serialize/restore. Pure JVM unit test, no device, no emulator. Everything else in #34 needs a device.

---

### #35 — Federated codec & Python Flower simulation (`04_code_plans/03`) · checkpoint

**Required:** `FederatedAdapterRecord` over `TrainableTensorCodec` (no new ordering); `flower_client.py` with `get_parameters`/`fit`/`evaluate`; `flower_sim.py` FedAvg driver saving a **new global adapter per round**; `mobiletransformers federated simulate` CLI; flwr in a separate extra/group; five tests; bounded comm size; aggregated adapter loads in desktop inference and changes logits.

**Verified present:**
- `src/mobiletransformers/federated/adapter_record.py:56-61` — `codec_tensor_specs()` = `for entry in handoff._sorted_entries(): specs.extend(entry.tensor_specs())`. Ordering is `HandoffMap._sorted_entries` (`artifacts/handoff_map.py:171-178`, sorted by canonical weight name) → **the record genuinely invents no ordering** (F8 holds).
- `adapter_record.py:108-121` — `from_handoff` fails closed on array/spec count mismatch and sets `adapter_format_version = handoff.schema_version`.
- `adapter_record.py:123-130` — `check_format()` calls the **shared** `check_compat` (`artifacts/versioning.py:37-52`) on the record's own `schemaVersion`/`minReaderVersion`, then equality-checks `adapterFormatVersion == handoff.schema_version` → `HandoffError`.
- `adapter_record.py:137-172` — pinned serialization: `struct.pack("<I", len(header))` + `json.dumps(header, sort_keys=True)` UTF-8 + concatenated C-order LE payloads with `byteOffset`/`byteLength`. Deterministic (`sort_keys`), no padding, matches the plan text exactly.
- `tests/federated/fixtures/federated_record.golden.bin` **exists** (note: NOT at `tests/fixtures/`), regenerable via `tests/federated/gen_serialization_golden.py`, and `tests/federated/test_serialization_golden.py:24-27` does a real `assert blob == _GOLDEN.read_bytes()` plus a deserialize-back test.
- `flower_sim.py:126-154` — pure-numpy weighted `federated_average`, fails closed on no survivors / tensor-count mismatch / zero weight; `:157-163` `save_global_adapter` → `global_adapter_round<N>.mtfed`.
- `cli/federated.py:14-28,36-62` + `cli/main.py:16,25,37` — `mobiletransformers federated simulate` is wired into the dispatcher; `--help` works in the core env (verified by running it).
- `pyproject.toml:34-37` — flwr is **genuinely out of the lock** (`grep 'name = "flwr"' uv.lock` → 0 hits), with the reason (protobuf/rich/typer downgrade) and the out-of-band path `pip install "flwr[simulation]"` documented in the comment; `flower_sim.py:185-191` raises that same instruction as a `HandoffError` on `ImportError`.
- All 5 required test files exist and **13 tests pass in the core env** in 0.14s.

**Gaps:**
- (a) **`build_server_app` is a stub**: `flower_client.py:79-91` — `_ = (handoff, base_model_id, peft_method, rounds, output_dir); return ServerApp()`. No FedAvg strategy attached, no round loop, no `save_global_adapter` call. `run_simulation` (`flower_sim.py:194-204`) never calls `save_global_adapter` either → **the CLI's `--output` directory is never written to**, and the DoD's "saves a new global adapter artifact per round" and "metric improves versus round 0" are unreachable even with flwr installed. This is the single biggest overstatement behind "code-complete".
- (a) **`run_local_training_step` is not a real `fit`**: `flower_client.py:40-49` — the incoming global adapter is discarded (`_ = incoming`, with the comment *"(incoming global adapter would be copied into `state.parameters` here before training.)"*), and the loop is `for _ in range(max_steps): optimizer.step(); model.lazy_reset_grad()` — **no forward pass, no loss, no batch**, with the comment *"... feed the deterministic tiny dataset ..."* left in place. It returns hardcoded `{"numExamples": 1, "trainLoss": 0.0}`. It is an ORT-API-shaped skeleton, not a training seam that would produce a moving metric.
- (a) `get_parameters` / `evaluate` (plan step 2) do not exist — only an `@app.train()` handler.
- (b) **`check_format` is never called in production code** — `grep -rn check_format` finds the definition and the two test call sites only. `deserialize()` (`adapter_record.py:174-211`) does no compat check at all, so an incoming record from a wrong-schema peer is accepted by the only ingest path that matters. Fail-closed is implemented but not wired.
- (b) `test_comm_size.py:18-21` bounds a 36-byte, 2-tensor toy record (`size < raw_payload + 4096`). Plan step 5 ("measure and bound the LoRA payload", "for LoRA **and MARS**") is not met in substance — no MARS case, no real package.
- (c)/(d) Manual legs correctly deferred: N-client ORT-`fit` sim, multi-round metric trend, aggregated-adapter desktop logits smoke. All need the source-built ORT-training wheel + out-of-band flwr; **no device needed** — these are host-runnable under the ORT-training profile.

**Drift / doubtful claims (answering the four verification asks directly):**
1. *Ordering derived from the codec?* **YES** — verified at `adapter_record.py:56-61` → `handoff_map.py:171-178`, pinned by `test_codec_roundtrip.py:11-18`. (Nit: it reaches through the private `_sorted_entries`; a public accessor would be safer.)
2. *`adapterFormatVersion` gated against handoff `schemaVersion` via the shared `check_compat`?* **YES, but only if a caller opts in** — `check_format` uses the one shared helper and equality-checks the handoff version, and `test_format_version.py` covers both directions; however nothing in `deserialize`/`flower_sim`/`cli` calls it.
3. *Byte golden exists and is compared?* **YES** — `tests/federated/fixtures/federated_record.golden.bin`, compared byte-for-byte at `test_serialization_golden.py:26-27`. (Path differs from the audit brief's `tests/fixtures/…`.)
4. *Is `flower_client.py` a real ORT `fit` seam or a stub?* **A stub/skeleton** — see above. HANDOFF.md:390's "ORT `fit` seam" and IMPLEMENTATION_ORDER.md:100's "code-complete" overstate it.

Additional contract drift worth flagging:
- **Role vocabulary mismatch.** Plan/tier contract: `role: adapter | trainable_weight | head`. Code emits the codec's `TensorSpec.role` ∈ `{weight, weight_quantized, scale, zero_point}` (`handoff_map.py:52-60,106-120` → `adapter_record.py:111`). Serialized headers (and therefore the frozen #36 golden) carry `"role": "weight"`, not the documented set.
- **"Do not aggregate merged base weights"** (`04_tier3_reach_extensions.md`, Implementation Notes) vs. the code's own docstring at `adapter_record.py:14-16`: *"the federatable set is the codec's ordered trainable tensor specs … True A/B-factor-only exchange is a future refinement."* The names used are `inference_initializer_names` (merged tensor identity, `aggregation_role="merged_base_plus_adapter"`) at full weight shape — i.e. the v1 record exchanges **merged-weight-shaped** tensors, not LoRA A/B factors. It is honestly documented as a simplification, but it contradicts the tier doc's explicit instruction and invalidates the "LoRA-sized payload" comm-size claim.
- `FederatedTensor.aggregation` is set uniformly from one `from_handoff` kwarg; the codec's per-spec `aggregation_role` is ignored, so `server_only`/`average` per-tensor policy is unreachable.
- `metrics` contract: plan wants `numExamples/numTokens/trainLoss/peakMemoryMb/durationMs`; the record's `metrics` is a free `dict[str, Any]` with no schema and no producer.

**Cheapest next step (host-only, no device):** two host changes, both testable in the core env — (i) call `check_format` inside `deserialize`/`save_global_adapter` ingest and add a test; (ii) make `build_server_app` actually run rounds: attach FedAvg over `federated_average` and call `save_global_adapter` per round, so `--output` produces artifacts. Then the manual leg (flwr + ORT-training wheel, host, no device) becomes worth running.

---

### #36 — Federated Android client & gateway (`04_code_plans/04`)

**Required:** Kotlin `federated/FederatedTrainingRepository.kt` + `federated/AdapterTensorCodec.kt`; JNI `exportTrainableTensors`/`importTrainableTensors`; Python `federated/gateway.py`; `mobiletransformers federated server` CLI; `AdapterTensorCodecTest.kt` cross-language golden; `FederatedConfigTest.kt` consent/TLS gate; `pytest tests/federated/test_gateway_dropout.py`.

**Verified present:** essentially nothing of the plan's own deliverables.
- `grep -rn "TrainableTensors" --include=*.kt --include=*.cpp --include=*.h` → **0 matches**. `ORTTrainerNative.kt:618-632` declares `releaseTrainingSession`, `saveModel`, `createTrainingSession`, `performTraining`, `setLearningRate`, `optimizerStep`, `mergeExportWeights` — no tensor import/export pair. `mergeExportSessionWeights` (the reuse hook the plan names) exists at `ORTTrainerNative.kt:602`.
- No `src/mobiletransformers/federated/gateway.py`; no `federated server` subcommand (`cli/federated.py:16` registers only `{simulate}`).
- No `tests/federated/test_gateway_dropout.py`; no Kotlin `federated/` package; no consent/TLS config type anywhere.
- The one genuine head start: #35's `federated_record.golden.bin` + the pinned serialization already freeze the cross-language contract this plan must mirror.

**Gaps:** (d) everything, deliberately — the plan says *"Gated hard: do not start until the Option-A simulation (#35) passes"*, and #35's simulation leg has not been run. IMPLEMENTATION_ORDER.md:448 leaves the gate self-check unticked, which is consistent.

**Drift / doubtful claims:** none — no code, no claims. One forward-looking risk: the golden #36 must mirror already bakes in #35's role-vocabulary and merged-weight drift (above); freezing a wrong-vocabulary header now costs a golden regen later.

**Cheapest next step:** do not start. If forced, the only host-doable slice is `gateway.py` + `tests/federated/test_gateway_dropout.py` over canned records (no device, no flwr if the aggregation goes through `federated_average`). Everything Kotlin/JNI needs a device for the round-trip smoke, though the `AdapterTensorCodecTest.kt` golden-parity test is a pure JVM unit test.

---

### #37 — FunctionGemma architecture gate & intents (`04_code_plans/05`) · checkpoint

**Required:** a `Gemma3Model` inference class set as the Gemma-3 `ArchitectureSpec.inference_model_class` (gate 1); `tools/functiongemma/mobile_actions.py`; Kotlin `agent/FunctionCallValidator.kt` + `agent/IntentBinder.kt`; `pytest tests/inference/test_gemma3_registry.py`; `FunctionCallValidatorTest.kt`; `IntentBinderTest.kt`; `pytest tests/functiongemma/test_actions_dataset.py`; gate result recorded in the support matrix.

**Verified present:**
- `src/mobiletransformers/config/registry/architecture.py:71-74` — `"Gemma3ForCausalLM": ArchitectureSpec(..., GemmaOnnxConfig, ("q_proj","v_proj"), None)` with the comment *"Gemma3 export is supported (GemmaOnnxConfig); inference is the FunctionGemma gate (#37)."* The gate is correctly **expressed as data** (a `None` `inference_model_class`) and `ArchitectureSpec.load_inference_model_class` (`:53-55`) fails closed with `"{arch} has no inference builder yet"`.
- `tests/unit/test_registries.py:29,38-41` — Gemma-3 resolves and has an OnnxConfig dotted path (training-export half only; the test does not assert the `None` inference gate).

**Gaps:**
- (a) **No `Gemma3Model`** — `grep "Gemma3" inference/builder.py` → 0 hits; the ladder at `inference/builder.py:3236-3241` still handles only `GemmaForCausalLM` / `Gemma2ForCausalLM`. Gate 1 is unattempted.
- (a) `inference/builder.py:3235-3250` is still a literal `if/elif config.architectures[0] == ...` chain — the architecture registry is **not consumed** at the inference dispatch site (documented as gated by the inference migration in `architecture.py:9-11`, so this is planned drift, not accidental). Whoever adds `Gemma3Model` must decide whether to add an `elif` (violating the plan) or migrate the site first.
- (a) No `tools/functiongemma/` (`ls tools/` → `__init__.py`, `parser_config.py`, `tokenizer_export.py`, `utils.py`); no `mobile_actions.py`.
- (a) No Kotlin `agent/` package: no `FunctionCallValidator.kt`, no `IntentBinder.kt`, no action-schema type, no allowlist.
- (a) None of the four required tests exist (`tests/inference/` has no `test_gemma3_registry.py`; no `tests/functiongemma/`).
- (a) **Gate result not recorded in the support matrix** — `grep -rn "Gemma3|gemma-3" docs/ src/mobiletransformers/support/` → 0 hits. `04_code_plans/05` step 1 and the tier doc's acceptance criterion *"FunctionGemma has documented architecture and differentiation gate results"* are unmet: the gate is encoded in a registry `None` and a code comment, but there is no user-visible pass/fail record.
- (d) Everything downstream (dataset, on-device fine-tune, demo) is correctly gated behind gate 1 + gate 2.

**Drift / doubtful claims:** the tool-grammar gate (gate 3) is decidable *now* — HANDOFF.md:429-435 records Gate 0.1's GenAI verdict and the dual-`.so` outcome, so the validator strategy (grammar vs. post-hoc JSON) could be written down without touching Gemma-3. Nothing has been.

**Cheapest next step (host-only, no device):** the honest one is *not* to start the spike but to **record the gate as an explicit not-yet-passed row** in the support matrix / `02_code_plans/02` so the acceptance criterion "failed gates become explicit future work" is met — a docs-only change. If the spike is wanted, `tests/inference/test_gemma3_registry.py` asserting `inference_model_class is None` **today** (as a canary that flips when the gate passes) is ~10 lines, host-only. The `Gemma3Model` export itself is host-only too (Python export, no device); only the on-device fine-tune and intent demo need a device.

---

## Tier-doc requirements not picked up by any code plan

Read from `agent_docs/04_tier3_reach_extensions.md` against the five plan files:

1. **"Do not aggregate merged base weights. Aggregate adapter/trainable tensors only."** (§3.3 Implementation Notes) — no plan owns this as a testable acceptance line, and #35's implementation currently exchanges merged-weight tensor identities by explicit design note (`adapter_record.py:14-16`). Nothing will fail if this stays wrong.
2. **MARS federation** — §3.3 scope says "MARS tensors if their names/shapes/merge semantics are stable"; `04_code_plans/03` only mentions MARS inside the comm-size test. `test_comm_size.py` has no MARS case, and no plan owns the "are MARS tensor semantics stable enough to federate?" decision.
3. **`get_parameters` / `set_parameters` API gap** — §3.3 "What is not solved" explicitly names *"MobileTransformers lacks a stable `get_parameters`/`set_parameters` API for trainable adapter tensors."* `04_code_plans/03` lists `get_parameters` under step 2 only; nothing exists, and the Python-side public API (`tests/unit/test_public_api.py`) does not export one.
4. **Storage check before each chunk** (§3.2: "Check thermal state, battery state, and available storage") — `04_code_plans/02`'s chunk contract carries thermal + battery + storage in prose but its Definition of done only names `THERMAL_STATUS_SEVERE`. No storage precondition is testable.
5. **`docs/mobile_evaluation.md`-style energy/thermal export** — required by both the tier doc and #34's DoD; there is no owner for the *format*, and `docs/` was not verified to contain a `mobile_evaluation.md` template for scheduled runs.
6. **Encoder "intent classification for on-device mobile tasks"** (§3.1 scope, 4th bullet) — the natural bridge between #33 and #37 (`IntentBinder`); neither plan's Definition of done requires it.
7. **flwr install path is documented only in a `pyproject.toml` comment** (`:34-37`) and the runtime error string (`flower_sim.py:188-190`) — `grep -rn flwr docs/ README.md Makefile` → 0 hits. #31 (docs) does not own a federated page, so the out-of-band install is discoverable only by reading `pyproject.toml`.

## Remaining work, ordered

**Host-doable now (no device, no flwr, core `.venv`):**
1. `#34` `LinearLRScheduler.stateDict()/loadFromState()` + `ORTSchedulerTest.kt` — closes the only *actively crashing* defect in Tier 3 (`ORTTrainerNative.kt:127` calls `stateDict()` on every checkpoint). JVM unit test only.
2. `#35` wire `check_format` into `deserialize`/ingest + a fail-closed test.
3. `#35` make `build_server_app`/`run_simulation` actually aggregate and call `save_global_adapter` per round, so `--output` is populated; add a CLI test in `tests/cli/test_cli.py` (currently zero `federated` coverage there).
4. `#35` decide + record the role-vocabulary and merged-vs-A/B question **before** #36 mirrors the golden; regenerate the golden if the vocabulary changes.
5. `#33` `TASK_REGISTRY` + `TaskType.{TEXT_CLASSIFICATION,SIMILARITY}` + `tests/encoder/test_task_registry.py` and `test_mapping.py`.
6. `#37` record the Gemma-3 gate as an explicit unmet row in the support matrix (docs-only, satisfies the tier acceptance criterion without spiking).
7. `#36` (only if the #35 gate is declared passed) `gateway.py` + `tests/federated/test_gateway_dropout.py` over canned records.

**Host-doable but heavy profile (ORT-training wheel and/or out-of-band `flwr[simulation]`; still no device):**
8. `#35` manual leg: N-client sim with a real `fit`, multi-round metric trend, aggregated-adapter desktop logits smoke. Blocked behind items 2-3 above — the current `run_local_training_step` cannot move a metric.
9. `#33` encoder export smoke + one desktop train step (`all-MiniLM-L6-v2`).
10. `#37` gate-1 spike: `Gemma3Model` + inference-graph export.

**Device-required:**
11. `#34` foreground `dataSync` worker, Doze/charging behavior, multi-chunk resume, thermal/energy traces (CHECKPOINT #34).
12. `#33` Android encoder one-step training smoke (or a documented blocker).
13. `#36` JNI import/export round-trip instrumentation, one-client gateway round, on-device payload size.
14. `#37` on-device FunctionGemma fine-tune + train→tool-call→dry-run-intent demo (CHECKPOINT #37).

**Manual/user-run judgement calls (no code):**
15. Decide whether #35's box should be re-labelled from "code-complete" to "partial" in `IMPLEMENTATION_ORDER.md:100` and `HANDOFF.md:386-397`.
16. Correct `IMPLEMENTATION_ORDER.md:306`'s `[x]` for the #18 session lock, or build the lock — #34 step 4 depends on it existing.
