# Dependency Profiles and the ORT-Training Wheel
**Priority (global #):** 2  |  **Prerequisites:** `00_code_plans/01_python_package_and_uv_scaffolding.md`  |  **Blocks:** `01_code_plans/06_source_built_ort_training_pipeline.md`, `01_code_plans/05_optimum_onnx_export_and_tasksmanager.md`, and every plan that imports `onnxruntime` / `optimum`

## Purpose

Lock the toolchain before anything builds on it. The crux: **four dependency surfaces all collide on the single `onnxruntime` import name** and must never share one environment. This plan defines isolated profiles (uv extras + groups), the source-built `onnxruntime-training==1.23.0+cpu` wheel and its provenance manifest, the one CI unknown to resolve (the torch version the ORT-training wheel was compiled against), and the reconciliation of the two drifting requirements files into one resolved set.

Evidence the repo already has the collision: `requirements-ort.txt` pins `onnxruntime-training==1.23.0+cpu` while `requirements-or.txt` pins `onnxruntime-rocm==1.18.0` + `torch-ort==1.19.2`; the export path imports `onnxruntime as rt` and `from onnxruntime.training import onnxblock, artifacts` in the same module (`artifact/onnx_builder.py:20,24`). All three provide/shadow `import onnxruntime`.

## The collision (why isolation is mandatory)

`onnxruntime`, `onnxruntime-training`, `onnxruntime-rocm`, `onnxruntime-genai`, and `optimum-onnx[onnxruntime]` all install code under (or depend on) the **same top-level `onnxruntime` import package**. Installing two of them into one venv yields whichever wins the install order — silent, version-dependent breakage. uv treats them as the same distribution name in some cases and as conflicting providers in others, so the safe contract is: **one onnxruntime provider per synced environment**, enforced by never combining the relevant extra/group pairs in a single `uv sync`.

## Dependency profile table

| Profile | Key contents | onnxruntime source |
| --- | --- | --- |
| **train+export (unified)** | `onnxruntime-training==1.23.0+cpu` (local wheel, **PROVIDES `onnxruntime`**), `optimum-onnx==0.1.0` **WITHOUT `[onnxruntime]` extra**, `transformers>=4.45,<4.58`, `torch` (pinned to the ORT-training build — see "the one unknown"), `onnx`, `onnxscript`, `peft` | training wheel |
| **export-only** | `optimum-onnx[onnxruntime]`, `transformers<4.58`, `torch`, `onnx`, `onnxscript` | public `onnxruntime` |
| **genai-desktop-smoke** | `onnxruntime-genai>=0.14` | bundled by genai |
| **rag** | `langchain-community`, `langchain-huggingface`, `langchain-objectbox==0.1.0`, `sentence-transformers` | — |
| **eval** | `deepeval`, `matplotlib` | — |

Mapping to the `pyproject.toml` from plan 01:
- **train+export (unified)** → `[dependency-groups] ort-training-local` (+ the local wheel via `[tool.uv.sources]`). `optimum-onnx` here is **bare** (no `[onnxruntime]`) so it does not pull public onnxruntime over the training wheel.
- **export-only** → `[project.optional-dependencies] export` (uses `optimum-onnx[onnxruntime]`).
- **genai-desktop-smoke** → its own group (add `genai-smoke = ["onnxruntime-genai>=0.14"]`), or fold into `export` only if genai is the chosen engine; keep separate by default.
- **rag** → `[project.optional-dependencies] rag`.
- **eval** → `[project.optional-dependencies] eval`.

**Never co-sync** (mutually exclusive onnxruntime providers): `ort-training-local` ✕ `export`; `ort-training-local` ✕ `genai-smoke`; `export` ✕ `genai-smoke`; `export` ✕ `export-rocm`. Declare this with uv conflicts:

```toml
[tool.uv]
conflicts = [
  [{ group = "ort-training-local" }, { extra = "export" }],
  [{ group = "ort-training-local" }, { group = "genai-smoke" }],
  [{ extra = "export" }, { group = "genai-smoke" }],
]
```

rag/eval carry no onnxruntime provider and may combine with any one runtime profile.

### Core (always-synced) dependencies

Independent of the onnxruntime profiles, the **core** dependency set (alongside `python-dotenv` from plan 01) must include **`pydantic>=2`**. It carries no onnxruntime provider, so it combines with every profile. This is required by `00_code_plans/09`, which makes Pydantic v2 the typed config contract for every config object and the cross-boundary JSON schema (`schemas/*.schema.json`). Add it to the base `[project] dependencies` (not an extra/group) so `config/models.py` is importable in any synced environment, including the train+export and export profiles that emit `training_config.json` / `generation_config.json` / `rag_config.json` via `model_dump(by_alias=True)`.

## Verified facts to state in the build docs

- **optimum-onnx 0.1.0** requires `optimum~=2.1`, `transformers>=4.36,<4.58`, and Python 3.9–3.13. (This is why plan 01 sets `requires-python = ">=3.10,<3.14"` and every transformers pin is `<4.58`.)
- **Public `onnxruntime-training` on PyPI stalls at ~1.19.2.** The repo needs `1.23.0+cpu`, which therefore **MUST be source-built**. There is no public wheel to install.
- **Optimum v2's ORTTrainer/training-wrapper deprecation does NOT affect this repo.** The repo never used Optimum's `ORTTrainer`. It uses `onnxruntime.training.artifacts.generate_artifacts` (`artifact/onnx_builder.py:65`) + `optimum.exporters.onnx.export` (`trainer/builder.py:14`). The `ORTTrainer` class in `trainer/validator.py:335` is the repo's **own** class wrapping `onnxruntime.training.api` (`CheckpointState, Module, Optimizer` — `artifact/onnx_builder.py:21`), not `optimum.onnxruntime.ORTTrainer`. So upgrading to optimum 2.x for the export path is safe.

## The one unknown to lock in CI

**The exact `torch` version the source-built ORT-training wheel was compiled against is unknown and must be resolved, then pinned.** ORT-training links against a specific torch ABI; a mismatch crashes at import. Procedure:

1. Build a **clean** venv from the current pins in `requirements-ort.txt` (`onnxruntime-training==1.23.0+cpu`, `optimum==1.23.3`, `peft==0.13.2`, `onnxscript==0.3.1`, plus whatever transformers/torch resolve).
2. Record exactly what `torch` resolves to (`uv pip freeze | grep -i '^torch'`).
3. Pin that exact `torch==X.Y.Z` into the `ort-training-local` group and into `third_party/onnxruntime/manifest.json` (`torch_version` field).
4. CI re-builds from the pin and asserts `python -c "import torch, onnxruntime; from onnxruntime.training import artifacts"` succeeds.

Until step 2 is done, leave `torch` unpinned in `ort-training-local` and mark it `# TODO(plan03): pin to resolved ORT-training torch ABI`.

## Reconciling the two drifting requirements files

The files disagree on shared packages:

| Package | `requirements-or.txt` | `requirements-ort.txt` | Resolution |
| --- | --- | --- | --- |
| optimum | `1.23.2` | `1.23.3` | export profile moves to `optimum~=2.1` (required by optimum-onnx 0.1.0); legacy 1.x kept only if a frozen ROCm lock needs it |
| deepeval | `3.4.7` | `3.1.4` | take newer `3.4.7` in `eval` extra |
| langchain-community | `0.3.19` | `0.3.27` | take newer `0.3.27` in `rag` extra |
| peft | `0.13.2` | `0.13.2` | agree → `peft>=0.13,<0.14` |

Procedure (doc 00 Implementation Sequence step 2):

```bash
# seed groups from existing files (then split/clean — do NOT keep mixed shape long-term)
uv add -r requirements-ort.txt --group ort-training-local
uv add -r requirements-or.txt  --group export-rocm
# hand-resolve the conflicts above into extras/groups, drop notebook/CUDA/ROCm noise
# (bertviz, PyQt6, seaborn, llama-index, nvidia-*-cu12, triton, torchaudio/torchvision +rocm
#  go to export-rocm / a research group, NOT core)
```

ROCm/CUDA stack from `requirements-or.txt` (`onnxruntime-rocm`, `torch-ort`, `torchaudio==2.3.1+rocm6.0`, `torchvision==0.18.1+rocm6.0`, `nvidia-*-cu12`, `triton==3.1.0`) is its own `export-rocm` group, kept out of default sync (doc 00).

## Touched / new files

New:
- `scripts/build_ort_training_wheel.sh` — builds the CPU training wheel (`--enable_training_apis --build_wheel`).
- `scripts/build_ort_training_android.sh` — builds the Android AAR/headers/`.so` with matching ORT commit.
- `third_party/onnxruntime/manifest.json` — provenance (schema below).
- `third_party/onnxruntime/BUILD.md` — human build steps.
- `third_party/wheels/README.md` — explains the git-ignored `.whl`.
- `requirements/requirements-export.lock.txt`, `requirements-train-local.lock.txt`, `requirements-rag.lock.txt`, `requirements-dev.lock.txt`, `sbom-cyclonedx.json` — all generated, not hand-edited.

Modified:
- `pyproject.toml` — fill exact pins into extras/groups; add `[tool.uv] conflicts`; add `genai-smoke`/`export-rocm` groups.
- `.gitignore` — `third_party/wheels/*.whl`, `third_party/onnxruntime/build/`.

## Data contracts / interfaces

### `third_party/onnxruntime/manifest.json`

```json
{
  "ort_git_sha": "<exact onnxruntime commit SHA>",
  "ort_tag": "v1.23.0",
  "version": "1.23.0+cpu",
  "build_flags": ["--enable_training_apis", "--build_wheel", "--config", "Release"],
  "python_version": "3.11.x",
  "torch_version": "<RESOLVED in CI — the one unknown>",
  "cmake_version": "<x.y.z>",
  "ndk_version": "<r26x / 26.x.x>",
  "android_api_level": 24,
  "abis": ["arm64-v8a", "armeabi-v7a", "x86_64"],
  "wheel": {
    "filename": "onnxruntime_training-1.23.0+cpu-local.whl",
    "sha256": "<sha256 of the built wheel>"
  },
  "android": {
    "aar_sha256": "<sha256>",
    "so_sha256": {"arm64-v8a": "<sha256>", "armeabi-v7a": "<sha256>", "x86_64": "<sha256>"}
  }
}
```

`[tool.uv.sources] onnxruntime-training = { path = "third_party/wheels/onnxruntime_training-1.23.0+cpu-local.whl" }` (from plan 01) points at the artifact whose `sha256` matches `manifest.json.wheel.sha256`.

### Lock generation (doc 00 commands)

```bash
uv sync --group dev --extra export
uv export --extra export --format requirements.txt -o requirements/requirements-export.lock.txt
uv sync --group ort-training-local
uv export --group ort-training-local --format requirements.txt -o requirements/requirements-train-local.lock.txt
uv export --extra rag --format requirements.txt -o requirements/requirements-rag.lock.txt
uv export --format cyclonedx1.5 -o requirements/sbom-cyclonedx.json
```

## Implementation steps

1. Fill exact pins into the plan-01 `pyproject.toml` extras/groups using the reconciliation table; add `genai-smoke` and `export-rocm` groups; add `[tool.uv] conflicts`.
2. Author `scripts/build_ort_training_wheel.sh` and `scripts/build_ort_training_android.sh` (clone ORT at the manifest SHA, configure NDK/ABIs, build, emit wheel + AAR + `.so`, compute SHA256).
3. Write `third_party/onnxruntime/BUILD.md` and a `manifest.json` with all fields except the torch ABI placeholder.
4. **Resolve the one unknown:** clean-build from `requirements-ort.txt`, capture resolved `torch`, pin it in `ort-training-local` and `manifest.torch_version`.
5. Build the wheel; drop it at the `[tool.uv.sources]` path; record its SHA256 in the manifest; git-ignore the `.whl`.
6. Run the reconciliation `uv add -r ...` seed, then hand-split into clean extras/groups (drop notebook/CUDA/ROCm noise out of core into `export-rocm`/research).
7. Generate all `requirements/*.lock.txt` + `sbom-cyclonedx.json` via `uv export`.
8. Verify isolation: each profile syncs alone and imports its onnxruntime provider; the declared conflict pairs error out on co-sync.

## Interactions with other plans

- **Plan 01** owns the `pyproject.toml` skeleton, the `[tool.uv.sources]` path entry, the empty `requirements/`, `scripts/`, `third_party/` dirs, and `requires-python`/transformers caps this plan justifies.
- **Plan 06 (source-built ORT training pipeline)** consumes the `ort-training-local` profile and the built wheel to prove `generate_artifacts` runs end-to-end — it is the first thing to validate after this plan locks the toolchain.
- **Plan 05 (optimum-onnx export)** consumes the `export` profile (`optimum-onnx[onnxruntime]`, optimum 2.x) and relies on the verified fact that Optimum v2's ORTTrainer deprecation does not break this repo's `optimum.exporters.onnx.export` usage (`trainer/builder.py:14`).
- The Android side of `build_ort_training_android.sh` feeds the Gradle rename plan (`00_code_plans/04`) which links the local `onnxruntime-genai.aar` / training `.so`.
- **Plan 09 (typed models / enums / registries)** requires `pydantic>=2` in core (above); it is the first consumer to need it, ahead of the export/builder plans.

## Tests & smokes

- `uv sync --frozen --group ort-training-local` then `python -c "import onnxruntime; from onnxruntime.training import artifacts; from onnxruntime.training.api import CheckpointState, Module, Optimizer; import torch"` — train+export profile alive (mirrors `artifact/onnx_builder.py:20-21`).
- `uv sync --frozen --extra export` then `python -c "import onnxruntime; from optimum.exporters.onnx import export"` — export-only profile alive.
- `uv sync --extra export --group ort-training-local` **must fail** (declared conflict) — proves isolation is enforced.
- `sha256sum third_party/wheels/onnxruntime_training-1.23.0+cpu-local.whl` matches `manifest.json.wheel.sha256`.
- `uv export ...` regenerates every `requirements/*.lock.txt` deterministically (no diff on re-run).
- CI asserts `manifest.json.torch_version` is non-empty (the one unknown is resolved, not left TODO).
