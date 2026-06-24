# Optimum ONNX Support Matrix

**Priority #19 | Prerequisites: #6 (`01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`) | Blocks: starter-zoo docs, contributor model-onboarding**

## Purpose

Produce a single machine-readable `model_support_matrix.json` that records, per candidate model, how far it gets along the MobileTransformers readiness pipeline — from "Optimum can export it" all the way to "Android can train it and RAG works." The matrix is the reporting layer over #6's `TasksManager`-based candidate detection. It uses **inherited statuses** (a later status can only be `true` if its predecessor is `true`) plus a `blockers[]` list so contributors can see exactly where a newly Optimum-supported architecture stalls. Only the *ready* statuses (`android_inference_ready`, `android_training_ready`, `rag_ready`) surface in user-facing starter-zoo docs; the earlier statuses are contributor-only.

This aligns MobileTransformers to Optimum ONNX as the first support filter (decision in `02_tier1_hf_integrated_core.md` §"Optimum ONNX Support Alignment Plan") rather than maintaining a hand-curated list. Today there is **no** support-matrix generation in the repo — PEFT target modules are hard-coded in `peft_models/mars/utils.py` (`TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING`), and `trainer/builder.py` imports concrete Optimum configs (`LlamaOnnxConfig, Qwen2OnnxConfig, Phi3OnnxConfig, …`) without querying `TasksManager`. This plan introduces the discovery + reporting layer.

**Verified facts this plan relies on:** optimum-onnx 0.1.0 exposes `optimum.exporters.tasks.TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")`; transformers ceiling is `<4.58`. (Note: the repo's current `requirements-*.txt` pin `optimum==1.23.x` — the legacy bundled package. #2/#6 migrate to standalone `optimum-onnx`; this plan targets the post-migration API and records the resolved versions at runtime rather than hard-coding them.)

## Touched / new files

New (Python, under the package scaffolded by #1 — e.g. `mobiletransformers/`):

- `mobiletransformers/support/matrix.py` — the generator: `SupportMatrixGenerator`, `build_matrix(...)`, `detect_candidate(...)`, status-inheritance logic, JSON writer.
- `mobiletransformers/support/statuses.py` — the ordered status enum + inheritance helper.
- `mobiletransformers/support/models.py` — dataclasses for `CandidateEntry` and the matrix envelope.
- `mobiletransformers/support/__init__.py`.
- `config/support_candidates.yml` (or reuse the commented model list in `config.yml`) — the input candidate list.
- `build/support/model_support_matrix.json` — generated output (gitignored build artifact; a committed copy may live at `docs/model_support_matrix.json` for the user docs build).

Touched:

- `config.yml` — promote the commented candidate families (TinyLlama, Phi-3, Qwen2, SmolLM2, DeepSeek-R1-Distill-Qwen, embedding: all-MiniLM-L6-v2, Qwen3-Embedding, multilingual-e5-small) into a structured `SUPPORT_MATRIX.candidates` block, or point the generator at `config/support_candidates.yml`.
- CLI wiring (#14 `one_command_export_cli`): add a `mobiletransformers support-matrix` subcommand that calls `build_matrix`.

Reused from #6: the `TasksManager` wrapper, the task-priority selection, and the export-report it emits (`optimum/export_report.json`, `optimum/supported_tasks.json`). This plan consumes those rather than re-querying when an export already ran.

## Data contracts / interfaces

### Status order and inheritance (`statuses.py`)

Six ordered statuses; each implies all earlier ones. The generator computes them in order and the moment one is `false`, every later one is forced `false` and a blocker is recorded.

```
optimum_exportable        # TasksManager has a supported ONNX task for this model_type
  -> mobile_package_exportable   # MobileTransformers can normalize files + tokenizer/gen config + manifest
    -> train_artifacts_exportable # ORT training artifacts + PEFT/MARS metadata generate
      -> android_inference_ready  # Android loads package + generates >=1 token (probe result)
        -> android_training_ready # Android runs >=1 train step + merge (probe result)
          -> rag_ready            # embedding artifacts + RAG config validate (probe result)
```

```python
class SupportStatus(str, Enum):
    OPTIMUM_EXPORTABLE = "optimum_exportable"
    MOBILE_PACKAGE_EXPORTABLE = "mobile_package_exportable"
    TRAIN_ARTIFACTS_EXPORTABLE = "train_artifacts_exportable"
    ANDROID_INFERENCE_READY = "android_inference_ready"
    ANDROID_TRAINING_READY = "android_training_ready"
    RAG_READY = "rag_ready"

STATUS_ORDER = [...]  # as above
USER_FACING_STATUSES = {ANDROID_INFERENCE_READY, ANDROID_TRAINING_READY, RAG_READY}
```

Inheritance helper: `apply_inheritance(statuses: dict[SupportStatus,bool]) -> dict` zeroes every status after the first `false` and returns it; `first_blocked(statuses)` returns the earliest failing status name for blocker attribution.

### Status evidence sources

| Status | How decided (input) |
| --- | --- |
| `optimum_exportable` | `TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")` returns a non-empty task list including a usable causal/feature-extraction task |
| `mobile_package_exportable` | dry-run of #6/#14 normalization succeeds (tokenizer via `AutoTokenizer`, `generation_config.json`, manifest writable); RAG models need `feature-extraction` |
| `train_artifacts_exportable` | ORT training-artifact generation succeeds and PEFT/MARS target modules resolve — checked against `TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING` for the `model_type` (e.g. `qwen2`, `llama` present; unknown type ⇒ blocker "MARS target module mapping not verified") |
| `android_inference_ready` | a recorded Android probe result (instrumentation/manual), keyed by `modelId`+`variant`; absent ⇒ `false` + blocker "no android inference probe" |
| `android_training_ready` | recorded Android train+merge probe result |
| `rag_ready` | embedding ONNX present + `rag_config.json` validates + (optional) embedding probe |

The generator does NOT run Android on its own. It reads a probe-results file (`build/support/android_probes.json`) that CI/instrumentation tests write; missing probes leave the ready statuses `false` honestly.

### Candidate detection flow (`detect_candidate`)

```python
def detect_candidate(model_id: str, requested_task: str | None = None,
                     trust_remote_code: bool = False, token: str | None = None) -> CandidateEntry:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=token)
    model_type = cfg.model_type                      # e.g. "qwen2", "llama", "phi3", "bert"
    architectures = cfg.architectures                # e.g. ["Qwen2ForCausalLM"]
    supported = TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")  # list[str]
    selected = _select_task(supported, requested_task)   # priority below
    # record versions resolved in *this* env
    optimum_onnx_version = importlib.metadata.version("optimum-onnx")
    transformers_version = importlib.metadata.version("transformers")
    ...
```

Task selection priority (matches `02_tier1_hf_integrated_core.md` step 3): prefer `text-generation-with-past` → `text-generation` → `feature-extraction` → `sentence-similarity`; if none present and no `requested_task`, mark `optimum_exportable=false` with blocker "no supported onnx task". `trust_remote_code` is captured from the call (a model that *requires* it is flagged so user docs can warn). `opset` defaults to the export opset (20, per `config.yml` `extra_options.opset` and `trainer/builder.py` default `opset=20`).

### JSON shape (`model_support_matrix.json`)

Envelope plus a list of per-model entries:

```json
{
  "schemaVersion": 1,
  "generatedAt": "2026-06-24T00:00:00Z",
  "toolchain": {
    "optimumOnnxVersion": "0.1.0",
    "transformersVersion": "4.57.1",
    "transformersCeiling": "<4.58",
    "onnxRuntimeTrainingVersion": "1.23.0+cpu"
  },
  "statusOrder": [
    "optimum_exportable","mobile_package_exportable","train_artifacts_exportable",
    "android_inference_ready","android_training_ready","rag_ready"
  ],
  "userFacingStatuses": ["android_inference_ready","android_training_ready","rag_ready"],
  "models": [
    {
      "modelId": "Qwen/Qwen2-0.5B",
      "modelType": "qwen2",
      "architectures": ["Qwen2ForCausalLM"],
      "optimumOnnxVersion": "0.1.0",
      "transformersVersion": "4.57.1",
      "opset": 20,
      "supportedTasks": ["feature-extraction", "text-generation", "text-generation-with-past"],
      "selectedTask": "text-generation-with-past",
      "trustRemoteCode": false,
      "marsTargetModulesKnown": true,
      "statuses": {
        "optimum_exportable": true,
        "mobile_package_exportable": true,
        "train_artifacts_exportable": true,
        "android_inference_ready": false,
        "android_training_ready": false,
        "rag_ready": false
      },
      "blockers": ["no android inference probe recorded for variant cpu-int4"]
    }
  ]
}
```

Per-entry fields are recorded even when a later status is false: `optimumOnnxVersion`, `transformersVersion`, `opset`, `selectedTask`, `supportedTasks`, `trustRemoteCode` are always populated from detection (the early stages run regardless). `blockers[]` is human-readable; it always includes the reason for the earliest `false` status and may include downstream notes (e.g. "MARS target module mapping not verified for model_type 'gemma'").

A second, **filtered** output for user docs (`docs/model_support_matrix.json` or rendered into a doc table) includes only models with at least one `USER_FACING_STATUSES` true and strips the contributor-only earlier statuses.

## Implementation steps

1. **Candidate input.** Read `SUPPORT_MATRIX.candidates` (model id, optional `task`, optional `trust_remote_code`, target `variant`) from `config.yml`/`config/support_candidates.yml`. Seed it from the families already listed as comments in `config.yml`.
2. **`detect_candidate` per model.** `AutoConfig.from_pretrained` → `model_type`, `architectures`; `TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")` → `supportedTasks`; `_select_task` → `selectedTask`. Record `optimumOnnxVersion`/`transformersVersion` via `importlib.metadata.version`, `opset`, `trustRemoteCode`. Set `optimum_exportable` from whether a usable task exists.
3. **`mobile_package_exportable`.** If a real export already ran (#6/#14 produced `optimum/export_report.json` + `optimum/supported_tasks.json` for this model), trust it. Otherwise dry-run the normalization checks (tokenizer load via `AutoTokenizer`, generation config availability, manifest writability) and set the flag.
4. **`train_artifacts_exportable`.** Check ORT training-artifact generation availability for the selected task and resolve PEFT/MARS target modules: look up `model_type` in `peft_models/mars/utils.py`'s `TRANSFORMERS_MODELS_TO_MARS_TARGET_MODULES_MAPPING`; set `marsTargetModulesKnown` and, if unknown, add blocker "MARS target module mapping not verified for model_type '<type>'" and force the status `false`.
5. **Read Android probes.** Load `build/support/android_probes.json` (written by instrumentation/CI), keyed by `(modelId, variant)`, carrying `inferenceOk`, `trainStepOk`, `mergeOk`, `ragOk`. Map to `android_inference_ready` (inferenceOk), `android_training_ready` (trainStepOk && mergeOk), `rag_ready` (ragOk && embedding artifacts present). Missing entry ⇒ ready statuses `false` + appropriate blocker.
6. **Apply inheritance.** Run `apply_inheritance` so no status outranks a failed predecessor; use `first_blocked` to guarantee a blocker is present for the earliest failure.
7. **Write outputs.** Emit the full `model_support_matrix.json` (contributor view) to `build/support/`. Emit the filtered user-facing JSON/table to `docs/`.
8. **CLI.** Add `mobiletransformers support-matrix --config config.yml --out build/support/model_support_matrix.json [--docs docs/model_support_matrix.json] [--probes build/support/android_probes.json]` to the #14 CLI.

## Interactions

- **#6 (`optimum_onnx_export_and_tasksmanager`):** the source of the `TasksManager` wrapper, task-priority logic, and `optimum/*.json` reports. This plan reuses #6's task selection rather than duplicating it; keep `_select_task` identical to #6's chooser.
- **#14 (one-command export CLI):** hosts the `support-matrix` subcommand; an export run can update a single model's entry in place.
- **`peft_models/mars/utils.py`:** authoritative source for `train_artifacts_exportable`'s MARS-mapping check — when a new architecture is added there, the matrix should flip `marsTargetModulesKnown` to true on regeneration.
- **`config.yml`:** candidate list and `opset`/quant defaults feed detection.
- **File 01 (HF facade) / Android probes:** `android_*_ready` and `rag_ready` come from instrumentation that exercises the facade smoke (`fromPretrained → train(maxSteps=1) → merge → generate`), writing `android_probes.json`. The matrix consumes those results; it does not run Android itself.
- **Hub package (#13) / docs:** user-facing starter-zoo docs render only the filtered matrix.

## Tests & smokes

- **Inheritance unit test:** `{optimum_exportable:true, mobile_package_exportable:false, train_artifacts_exportable:true,...}` → after `apply_inheritance`, everything from `mobile_package_exportable` onward is `false`; `first_blocked` returns `mobile_package_exportable`.
- **Task-selection test:** supported `["feature-extraction","text-generation","text-generation-with-past"]` → `selectedTask=="text-generation-with-past"`; supported `["feature-extraction"]` for an embedding model → that task selected; empty/unsupported list → `optimum_exportable=false` + blocker.
- **Detection test (mocked `TasksManager`/`AutoConfig`):** `qwen2` model_type yields populated `supportedTasks`, `modelType`, `architectures`; an unsupported synthetic model_type yields `optimum_exportable=false`.
- **MARS-mapping test:** `model_type="qwen2"`→`marsTargetModulesKnown=true`; `model_type="gemma"` (not in the mapping) → `train_artifacts_exportable=false` + the specific blocker string.
- **Version-capture test:** `optimumOnnxVersion`/`transformersVersion` recorded from `importlib.metadata`; `transformersCeiling=="<4.58"` present in envelope.
- **Probe-merge test:** given an `android_probes.json` with `inferenceOk=true, trainStepOk=false`, `android_inference_ready=true`, `android_training_ready=false`, `rag_ready=false`.
- **JSON-shape test:** generated matrix validates against the schema (required keys per entry: `modelId, modelType, supportedTasks, selectedTask, trustRemoteCode, opset, optimumOnnxVersion, transformersVersion, statuses, blockers`).
- **Filtered-docs test:** user-facing output contains only models with a `USER_FACING_STATUSES` true and omits the three contributor-only statuses.
- **CLI smoke:** `mobiletransformers support-matrix` over the `config.yml` candidate list (network-mocked) produces a well-formed file without crashing on a model that lacks any supported task.
