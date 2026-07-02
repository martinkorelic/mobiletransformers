# Python Package and uv Scaffolding
**Priority (global #):** 1  |  **Prerequisites:** —  |  **Blocks:** `00_code_plans/02_config_layering_settings_constants.md`, `00_code_plans/03_dependency_profiles_and_ort_training_wheel.md`, and every later Python plan

## Purpose

Stand up the packageable `src/mobiletransformers/` layout, a standards-compliant `pyproject.toml` (hatchling build backend), uv-managed dependency surfaces (extras + groups), the local ORT-training wheel source declaration, and the `mobiletransformers` console entry point. This unblocks all subsequent Python work. The repo currently has **no** `pyproject.toml`, `uv.lock`, `setup.py`, or version tag (confirmed: `ls pyproject.toml uv.lock` → none), and code lives in flat root packages: `trainer/`, `artifact/`, `inference/`, `tools/`, `peft_models/`, `database/`, `evaluation/`, `research/`.

This pass adds scaffolding and **compatibility shims** only. It does NOT move implementation modules — those migrate subsystem-by-subsystem in later plans. Existing root imports such as `from tools.parser_config import TRAIN_CONFIG` (used in `trainer/builder.py:46`, `artifact/onnx_builder.py:28`, `inference/validator.py:16`, `trainer/validator.py:25`, `trainer/merge_validator.py:7`) MUST keep resolving throughout migration.

## Touched / new files

New:
- `pyproject.toml` (repo root) — build system, project metadata, extras, dependency groups, uv sources, scripts.
- `src/mobiletransformers/__init__.py` — exports `__version__`.
- `src/mobiletransformers/cli/__init__.py`, `src/mobiletransformers/cli/main.py` — console entry `main()`.
- `src/mobiletransformers/cli/export.py`, `validate.py`, `package_model.py` — stubs (real logic in later plans).
- Empty package skeletons mirroring doc 00 Target Hierarchy: `config/` (populated by plans 02/09: `constants.py`, `settings.py`, `models.py`, `registry/`), `export/`, `artifacts/`, `peft/` (`mars/`, `lora_xs/`, `ablation/`), `training/`, `inference/`, `rag/`, `hub/`, `evaluation/` (`benchmarks/`, `mobile/`, `smoke/`), `utils/` — each with `__init__.py`.
- `tests/unit/`, `tests/integration/`, `tests/smoke/` with `__init__.py` and one import-compat test.
- `requirements/` (empty dir; populated by plan 03 via `uv export`).
- `third_party/onnxruntime/` placeholder (manifest authored in plan 03).
- `scripts/` placeholder dir.
- `tests/unit/test_import_compat.py`.

Untouched (kept working via shims, migrated later): all existing root packages listed above.

## Data contracts / interfaces

### `pyproject.toml` (authoritative sketch)

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "mobiletransformers"
version = "0.1.0"
description = "Export and Android runtime tooling for on-device transformer training and inference."
readme = "README.md"
requires-python = ">=3.10,<3.14"
dependencies = [
  "huggingface-hub>=0.34",
  "numpy>=1.26",
  "onnx>=1.16",
  "pyyaml>=6.0",
  "python-dotenv>=1.0",
  "tokenizers>=0.20",
]

[project.scripts]
mobiletransformers = "mobiletransformers.cli.main:main"

[project.optional-dependencies]
# Public install surfaces. Exact pins resolved in plan 03.
export = ["optimum-onnx[onnxruntime]>=0.1.0", "transformers>=4.45,<4.58", "onnx", "onnxscript>=0.3"]
train  = ["transformers>=4.45,<4.58", "onnxscript>=0.3", "peft>=0.13", "onnx"]
rag    = ["langchain-community>=0.3", "langchain-huggingface>=0.3", "langchain-objectbox==0.1.0", "sentence-transformers>=5"]
eval   = ["deepeval>=3", "matplotlib>=3.10"]
hub    = ["huggingface-hub>=0.34"]

[dependency-groups]
# Local-only workflows. Never part of the published install surface.
dev               = ["pytest>=8", "ruff>=0.6", "mypy>=1.11"]
docs              = ["mkdocs>=1.6"]
smoke             = ["pytest>=8"]
# ORT-training wheel PROVIDES onnxruntime; never co-install with public onnxruntime/genai. See plan 03.
ort-training-local = ["onnxruntime-training==1.23.0+cpu", "optimum-onnx>=0.1.0", "peft>=0.13", "onnxscript>=0.3"]
android-build      = ["pyyaml>=6.0"]

[tool.uv.sources]
# Built by scripts/build_ort_training_wheel.sh; .whl is git-ignored (see plan 03).
onnxruntime-training = { path = "third_party/wheels/onnxruntime_training-1.23.0+cpu-local.whl" }

[tool.hatch.build.targets.wheel]
packages = ["src/mobiletransformers"]

[tool.hatch.metadata]
allow-direct-references = true   # required for the local path wheel source

[tool.uv]
# extras with a colliding onnxruntime import must not be synced together; see plan 03 conflict notes.
```

Notes:
- `requires-python = ">=3.10,<3.14"` (NOT `<3.13` — optimum-onnx 0.1.0 supports py 3.9–3.13, so 3.13 is in-window; cap below 3.14).
- `export` uses `optimum-onnx[onnxruntime]` (pulls public onnxruntime). `ort-training-local` uses `optimum-onnx` **without** the `[onnxruntime]` extra (the training wheel provides `onnxruntime`). Rationale and full conflict matrix in plan 03.
- License field intentionally omitted until the licensing decision (doc 00: target Apache-2.0; do not set yet).

### CLI entry contract (`cli/main.py`)

```python
def main(argv: list[str] | None = None) -> int:
    """Argparse dispatcher. Subcommands: export, validate, package-model.
    Returns process exit code. `mobiletransformers --help` and
    `python -m mobiletransformers.cli.main --help` both work."""
```

Subcommand stubs (`export.py`, `validate.py`, `package_model.py`) expose `add_parser(subparsers)` + `run(args) -> int`. In this plan they parse args and print a "not yet wired" notice, then return 0 under `--dry-run`. Real bodies land in plans 05 / 13.

> **This dispatcher contract is canonical for every later CLI plan.** `02_code_plans/05` (one-command export CLI) and `02_code_plans/04` (pull/install) register their subcommands into **this** `cli/main.py` argparse dispatcher via the same `add_parser`/`run` shape — they do not add a `cli/__main__.py`, do not change the console-script entry (`mobiletransformers.cli.main:main`), and do not introduce typer/click.

### Compatibility shim contract

Goal: a developer who runs `uv pip install -e .` can still execute `python -m trainer.builder` and `from tools.parser_config import TRAIN_CONFIG`. Two mechanisms, pick per-module as migration proceeds:

1. **Phase A (this plan): leave root packages in place.** The src layout is additive; root packages remain importable when cwd is the repo root. No shim needed yet for unmoved modules.
2. **Phase B (later plans, when a module physically moves to `src/`):** replace the old root module with a thin re-export shim, e.g. once `trainer/builder.py` becomes `src/mobiletransformers/export/training_export.py`:

```python
# trainer/builder.py  (shim, added in plan 05 — documented here for the contract)
import warnings
from mobiletransformers.export.training_export import *          # noqa: F401,F403
from mobiletransformers.export.training_export import main       # noqa: F401
warnings.warn("trainer.builder moved to mobiletransformers.export.training_export",
              DeprecationWarning, stacklevel=2)
```

The migration map (doc 00 "Migration Map") is the source of truth for old→new pairs.

## Implementation steps

1. Create `src/mobiletransformers/` and the full subpackage tree from doc 00 Target Hierarchy. Add `__init__.py` to every package and subpackage (`config`, `peft/mars`, `peft/lora_xs`, `peft/ablation`, `evaluation/benchmarks`, `evaluation/mobile`, `evaluation/smoke`, `cli`, `export`, `artifacts`, `training`, `inference`, `rag`, `hub`, `utils`).
2. Set `src/mobiletransformers/__init__.py` to `__version__ = "0.1.0"`.
3. Write `cli/main.py` with the argparse dispatcher and `if __name__ == "__main__": raise SystemExit(main())`. Wire `export`/`validate`/`package-model` subparsers to the stub modules.
4. Write the stub subcommand modules with `add_parser`/`run` and a `--dry-run` flag.
5. Author `pyproject.toml` per the sketch. Do NOT yet add real pins inside extras — plan 03 reconciles `requirements-or.txt` / `requirements-ort.txt` and pins. Leave loose `>=` ranges here so `uv sync` resolves during scaffolding.
6. Create `third_party/wheels/` and `.gitignore` it (`third_party/wheels/*.whl`). The `[tool.uv.sources]` path may dangle until plan 03 builds the wheel; gate it so core install (`uv sync` with no groups) does not require it (the wheel is only in the `ort-training-local` group).
7. Create empty `requirements/` and `scripts/` dirs (populated in plan 03).
8. Add `tests/unit/test_import_compat.py` asserting `from tools.parser_config import TRAIN_CONFIG, ARTIFACT_CONFIG, INFERENCE_CONFIG` and `import mobiletransformers` both succeed.
9. Run `uv sync --group dev` (no extras) to confirm the core env resolves and the `mobiletransformers` script installs.
10. Run `mobiletransformers --help` and `python -m mobiletransformers.cli.main --help`.

## Interactions with other plans

- **Plan 02 (config layering)** adds `src/mobiletransformers/config/` (`settings.py`, `constants.py` — **inside the package**, so the installed wheel is self-contained) plus repo-root `config/config.yml` (user-editable YAML only). The `config` subpackage created here in the skeleton (`src/mobiletransformers/config/__init__.py`) is plan 02's home. The `python-dotenv` core dep here exists because `config.py` currently calls `load_dotenv()` (`config.py:2-4`); plan 02 owns the migration of those secrets.
- **Plan 03 (dependency profiles + ORT wheel)** fills in exact pins inside the extras/groups defined here, builds the wheel that `[tool.uv.sources]` points at, authors `third_party/onnxruntime/manifest.json`, and generates the `requirements/*.lock.txt` files via `uv export`.
- **Plan 05 (optimum-onnx export)** and **plan 13 (one-command export CLI)** flesh out the CLI stubs and add the first real root→`src` shims following the Migration Map.

## Tests & acceptance

**Unit (automated)** — small, fast; prove the component wires together and compiles.
- `pytest tests/unit/test_import_compat.py` — both `mobiletransformers` and legacy `tools.parser_config` import.

**Integration (automated)** — runnable; produces a checkable expected output (tiny fixture in, asserted out).
- `uv sync --group dev` resolves with no extras (core stays small/platform-neutral); the `mobiletransformers` script installs.
- `mobiletransformers --help` exits 0; lists `export`, `validate`, `package-model`.
- `python -m mobiletransformers.cli.main --help` exits 0 (module-form parity).
- `mobiletransformers export --dry-run` exits 0 without importing the heavy export stack.
- `python -m build` (or `uv build`) produces a wheel containing only `src/mobiletransformers` (hatchling target check).

**Manual (user-run)** — long/intensive or device/emulator-specific; the **user** runs these.
- None for this plan (no source-built wheel or real export touched here; the ORT-training wheel arrives in plan 03).

**Definition of done** — explicit pass criteria + expected artifacts/behaviour when the plan is finished.
- `src/mobiletransformers/` subpackage tree (+ `__init__.py`) exists per doc 00 Target Hierarchy; `__version__ == "0.1.0"`.
- A standards-compliant `pyproject.toml` (hatchling backend, extras, dependency groups, `[tool.uv.sources]` for the local wheel, `mobiletransformers` console script) resolves under `uv sync --group dev` with no extras.
- Both `mobiletransformers --help` and `python -m mobiletransformers.cli.main --help` exit 0; legacy root imports (`from tools.parser_config import TRAIN_CONFIG`) still resolve.
- `uv build` emits a wheel containing only `src/mobiletransformers`.
