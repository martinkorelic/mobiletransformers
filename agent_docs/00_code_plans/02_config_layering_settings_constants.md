# Config Layering: Settings & Constants
**Priority (global #):** 4  |  **Prerequisites:** `00_code_plans/01_python_package_and_uv_scaffolding.md`  |  **Blocks:** `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`, `02_code_plans/05_one_command_export_cli.md`, and any export/CLI module that needs secrets or shared constants

## Purpose

Replace the two ad-hoc config surfaces with three explicit, ordered layers:

1. **`config/config.yml`** — user-editable, non-secret defaults (today's root `config.yml`).
2. **`config/settings.py`** — typed, env-driven runtime settings that **own all secrets and machine-specific paths**.
3. **`config/constants.py`** — non-secret shared constants (today's `tools/parser_config.py` plus a few literals scattered in code).

> **Scope split with `00_code_plans/09`.** This plan owns the **three layers + secrets** (`config.yml`, `Settings`, `constants.py`) and keeps `Settings` as a stdlib dataclass (secrets stay dependency-light). `00_code_plans/09` owns the **typed tunable-config layer** (Pydantic v2 models in `config/models.py`) and the **enum vocabulary** that is *added into* this plan's `config/constants.py`. So: secrets/paths → `Settings` (here); closed-set string choices → enums in `constants.py` (defined by 09); structured tunable config → Pydantic models (09). `pydantic>=2` becomes a core dependency (coordinated in `00_code_plans/03`).

After this plan, business logic never calls `os.environ[...]` directly for secrets. Today it does, in many places (verified):
- `os.environ['HF_TOKEN']` — `trainer/builder.py:255,257,258`, `trainer/validator.py:152`, `inference/validator.py:45,113,161,193`, `artifact/onnx_builder.py:125,331,495,617,646`, `inference/builder.py:3413`.
- `os.environ['HF_CACHE']` — `inference/builder.py:3412`.
- `os.environ["GEMINI_API_KEY"]` — `evaluation/openehr/openehr_eval.py:28`.
- Azure OpenAI vars via `from config import ...` — `evaluation/mobile/recommendation_eval.py:14,72-76`, sourced from `config.py:9-13`.
- Experiment constants via `from config import TASK_EPOCHS, BATCH_SIZE, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION, EXPERIMENT_RANKS` — `research/offline_train_eval.py:40`, defined in `config.py:17-31`.

## Touched / new files

New:
- `config/config.yml` — moved/copied from root `config.yml` (all sections: `TRAIN_BUILDER`, `INFERENCE_BUILDER`, `ARTIFACT_BUILDER`, `ARTIFACT_VALIDATOR`).
- `config/settings.py` — `Settings` model + `get_settings()`.
- `config/constants.py` — section names + lookup tables migrated from `tools/parser_config.py`, plus experiment constants from `config.py`.
- `src/mobiletransformers/utils/yaml.py` — `load_config_from_file(path)` consolidating the 4 duplicate copies (`trainer/builder.py:499`, `inference/builder.py:3397`, `artifact/onnx_builder.py:681`, `trainer/validator.py:1138`, `trainer/merge_validator.py:435`, `inference/validator.py:396`).
- `tests/unit/test_settings_precedence.py`.

Modified (shims, not full migration):
- Root `config.py` → thin re-export from `config.settings` + `config.constants` with a `DeprecationWarning`.
- `tools/parser_config.py` → thin re-export from `config.constants` with a `DeprecationWarning`.

## Data contracts / interfaces

### `config/settings.py`

`Settings` owns secrets + machine paths. Use `pydantic-settings` **only if** plan 03 accepts it as a dependency; otherwise a stdlib dataclass loader (doc 00 explicitly allows either). Stdlib shape below to keep core deps minimal:

```python
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os
from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    hf_token: str | None
    hf_cache: Path | None
    # Azure OpenAI (eval only)
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_deployment_name: str | None
    azure_model_name: str | None
    azure_api_version: str | None
    # Gemini (openehr eval)
    gemini_api_key: str | None

    def require_hf_token(self) -> str:
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN is not set (config/settings.py)")
        return self.hf_token

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()  # preserves current config.py:4 behavior
    def _p(v): return Path(v) if v else None
    return Settings(
        hf_token=os.environ.get("HF_TOKEN"),
        hf_cache=_p(os.environ.get("HF_CACHE")),
        azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME"),
        azure_model_name=os.environ.get("AZURE_MODEL_NAME"),
        azure_api_version=os.environ.get("AZURE_API_VERSION"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    )
```

Rule: `get_settings()` is the **only** place `os.environ` is read for secrets. Business logic calls `get_settings().hf_token` (or `.require_hf_token()`), never `os.environ['HF_TOKEN']`. The `DEEPEVAL_TELEMETRY_OPT_OUT=YES` writes (`evaluation/benchmark/*_eval.py`, `research/offline_train_eval.py:20`) are non-secret side-effect env writes and may stay, or move to a `configure_eval_env()` helper in `evaluation/__init__.py`.

### `config/constants.py`

Migrate verbatim from `tools/parser_config.py` (which is literally "all the section names of the config.yml file"):

```python
# Config section names (from tools/parser_config.py)
ARTIFACT_CONFIG = "ARTIFACT_BUILDER"
ARTIFACT_VALIDATOR_CONFIG = "ARTIFACT_VALIDATOR"
TRAIN_CONFIG = "TRAIN_BUILDER"
INFERENCE_CONFIG = "INFERENCE_BUILDER"
INFERENCE_ARTIFACT_CONFIG = "inference_config"
TEST_GENERATION_CONFIG = "test_generation_config"

TASK_NAME_TO_DATASET = {
    "logiqa": "data/logiqa_train",
    "hellaswag": "Rowan/hellaswag",
    "arc": "allenai/ai2_arc",
    "boolq": "google/boolq",
}

# Experiment constants (from config.py:17-31) — non-secret, so they belong here
TASK_EPOCHS = {"boolq": 2, "logiqa": 3, "arc_e": 4, "winogrande": 4,
               "arc_c": 4, "hellaswag": 1, "mini_personalqa": 6}
BATCH_SIZE = 32
PER_DEVICE_BATCH_SIZE = 6
GRADIENT_ACCUMULATION = 2
EXPERIMENT_RANKS = [2, 8, 32]

# Canonical artifact filenames (lift literals currently hard-coded in config.yml
# ARTIFACT_BUILDER, e.g. "quant_model.onnx")
DEFAULT_TRAIN_MODEL = "quant_model.onnx"
DEFAULT_INFERENCE_MODEL = "quant_model.onnx"
SUPPORTED_PEFT_METHODS = ("lora", "lora-xs", "mars", "all")  # from config.yml train_method comment
```

> **`00_code_plans/09` supersedes the closed-set tuples here with enums.** The `SUPPORTED_PEFT_METHODS` tuple becomes the `PEFTMethod(str, Enum)`, and 09 adds the rest of the vocabulary (`SamplingMethod`, `SchedulerType`, `ExecutionProvider`, `CoreConfigId`, `MemoryConfigId`, `SearchType`, `QuantizationType`, `TaskType`, `HandoffMode`, `MergerVariant`) into this same `config/constants.py`, mirrored 1:1 by Kotlin enums. Keep the section-name/filename/experiment constants here; treat the enum block as 09's to populate.

### `src/mobiletransformers/utils/yaml.py`

```python
import yaml
def load_config_from_file(config_file: str | Path) -> dict:
    with open(config_file) as f:
        return yaml.safe_load(f)
```

Replaces the six near-identical local definitions listed above.

## Precedence model

Effective value for any tunable resolves in this strict order (highest first):

| Rank | Source | Owner |
| --- | --- | --- |
| 1 | CLI flag (`argparse`) | `cli/*.py` |
| 2 | Environment variable | `config/settings.py` (`get_settings()`) |
| 3 | `config/config.yml` value | `utils/yaml.load_config_from_file` |
| 4 | Package default constant | `config/constants.py` |

Implementation idiom for a resolver (used by export/CLI modules):

```python
def resolve(cli_value, env_value, yaml_value, default):
    for v in (cli_value, env_value, yaml_value, default):
        if v is not None:
            return v
    return None
```

Secrets (HF_TOKEN, Azure, Gemini) skip ranks 1/3/4 — they live only at rank 2 (env via settings). YAML never holds secrets.

## Implementation steps

1. Add `config/__init__.py` if Python-import of the `config/` dir is desired; otherwise load `config.yml` by path and import `settings`/`constants` as modules. (Root `config.py` already shadows a `config` package name — see step 6 for the shim that resolves this.)
2. Copy root `config.yml` → `config/config.yml` unchanged (preserve every section so Android JSON-emission paths in `artifact/onnx_builder.py` do not drift — doc 00 risk: "Moving config may break Android artifact generation if JSON paths drift").
3. Write `config/constants.py` migrating `tools/parser_config.py` content + the `config.py` experiment constants + canonical filename/PEFT-method constants.
4. Write `config/settings.py` with the `Settings` dataclass + `lru_cache`d `get_settings()`.
5. Write `src/mobiletransformers/utils/yaml.py::load_config_from_file`.
6. Convert root `config.py` into a deprecation shim:
   ```python
   import warnings
   from config.settings import get_settings as _gs
   from config.constants import (TASK_EPOCHS, BATCH_SIZE, PER_DEVICE_BATCH_SIZE,
                                  GRADIENT_ACCUMULATION, EXPERIMENT_RANKS)
   _s = _gs()
   HF_TOKEN = _s.hf_token
   AZURE_OPENAI_ENDPOINT = _s.azure_openai_endpoint
   AZURE_OPENAI_API_KEY = _s.azure_openai_api_key
   AZURE_DEPLOYMENT_NAME = _s.azure_deployment_name
   AZURE_MODEL_NAME = _s.azure_model_name
   AZURE_API_VERSION = _s.azure_api_version
   warnings.warn("import from config.settings/config.constants instead of config.py",
                 DeprecationWarning, stacklevel=2)
   ```
   This keeps `evaluation/mobile/recommendation_eval.py:14` and `research/offline_train_eval.py:40` working unchanged.
7. Convert `tools/parser_config.py` into a re-export shim from `config.constants` (keeps `trainer/builder.py:46`, `artifact/onnx_builder.py:28`, `inference/validator.py:16`, `trainer/validator.py:25`, `trainer/merge_validator.py:7` working).
8. In migrated business modules (done per later plans), replace `os.environ['HF_TOKEN']` with `get_settings().require_hf_token()` and `os.environ['HF_CACHE']` with `get_settings().hf_cache`. List every call site from the Purpose section as the migration checklist.
9. Add a lint/CI guard: `grep -rn "os.environ\[.\(HF_TOKEN\|HF_CACHE\|GEMINI_API_KEY\|AZURE_\)" src/` must return nothing (ban direct secret reads in `src/`). Root legacy packages are exempt until they migrate.

## Interactions with other plans

- **Plan 01** created `src/mobiletransformers/utils/` and added `python-dotenv` as a core dep — `get_settings()` depends on both.
- **Plan 03** decides whether `pydantic-settings` is accepted; if yes, swap the dataclass for a `BaseSettings` subclass with the same field names and the same `get_settings()` accessor (callers unchanged).
- **Plan 05 / 13 (export CLI)** consume `get_settings()` for HF auth and `config/config.yml` for `TRAIN_BUILDER`/`INFERENCE_BUILDER`/`ARTIFACT_BUILDER` defaults; they own the CLI-flag layer (rank 1).
- Doc 00 Implementation Sequence steps 11–14 sequence this exactly: move `config.yml` (11), move secrets/constants (12), add compat wrappers (13), migrate `parser_config` consumers first (14).

## Tests & smokes

- `tests/unit/test_settings_precedence.py`: assert CLI > env > YAML > default using the `resolve()` idiom; monkeypatch `os.environ` and a temp YAML.
- `get_settings()` returns identical object across calls (lru_cache) and reads `HF_TOKEN` from a `.env` via `load_dotenv()`.
- Import-compat: `from tools.parser_config import TRAIN_CONFIG, TASK_NAME_TO_DATASET` and `from config import TASK_EPOCHS, AZURE_API_VERSION` both still resolve (and emit `DeprecationWarning`).
- CI guard grep over `src/` returns no direct secret env reads.
- `load_config_from_file("config/config.yml")` returns a dict containing all four top-level sections.
