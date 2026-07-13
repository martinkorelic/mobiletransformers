# ONNX Runtime Training — build provenance

This repo depends on a **source-built** `onnxruntime-training==1.23.0+cpu` Python wheel. Public PyPI
`onnxruntime-training` stalls around 1.19.2, so the 1.23.0 training APIs this project uses
(`onnxruntime.training.artifacts.generate_artifacts`, `onnxruntime.training.api.{CheckpointState,
Module, Optimizer}`) are only available from a local build.

The authoritative machine-readable provenance is [`manifest.json`](./manifest.json). This document is
the human-readable companion. **Nothing here is run automatically** — the wheel already exists (see
below); rebuild only when the ORT SHA or torch ABI must change.

## Current artifact (already built)

- **Wheel:** `onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl`
  (SHA256 `87e6f3c661b0a4c6bcaa347c3abcb9ebe05943e2b44cae04701fca89bd14c65d`)
- **Python:** 3.12.11 — the wheel is **cp312-only**; install it only under Python 3.12.
- **Paired torch ABI:** `torch==2.7.1` (observed build `2.7.1+cu126`). A mismatched torch crashes at
  import — keep this pin in lockstep with the `ort-training-local` group in `pyproject.toml`.
- **ORT source:** commit `9b25b6a838d83850300afeff37bcd18723f865e3`
  (`git describe` → `v1.19.0-1714-g9b25b6a838`), built 2025-07-08.
- Built in the sibling research repo `on_device_llm_finetune`; copied into
  `third_party/wheels/` (git-ignored) and referenced via `[tool.uv.sources]`.

## Verify the existing wheel is alive (Python 3.12)

```bash
uv sync --python 3.12 --group ort-training-local
uv run --python 3.12 python -c "import onnxruntime, torch; \
  from onnxruntime.training import artifacts; \
  from onnxruntime.training.api import CheckpointState, Module, Optimizer; \
  print(onnxruntime.__version__, torch.__version__)"
# expected: 1.23.0+cpu 2.7.1+...
```

## Rebuilding from source (reference — only if the pin must change)

Handled by [`../../scripts/build_ort_training_wheel.sh`](../../scripts/build_ort_training_wheel.sh).
Outline:

1. `git clone https://github.com/microsoft/onnxruntime && git checkout <manifest.ort_git_sha>`.
2. Build with `--enable_training_apis --build_wheel --config Release` under Python 3.12 in a venv
   whose `torch` matches `manifest.torch_version`.
3. Copy the emitted wheel into `third_party/wheels/`, recompute its SHA256, and update
   `manifest.json` (`wheel.sha256`, `ort_git_sha`, `python_version`, `torch_version`).

> A full ORT source build needs tens of GB of scratch space and many minutes. This machine's disk is
> near-full, so the build is intentionally **not** run here — the prebuilt wheel is reused as-is.

The Android training `.so`/AAR build (`build_ort_training_android.sh`) is a **later** plan
(`agent_docs/00_code_plans/04`) and its manifest fields (`ndk_version`, `abis`, `android.*`) are left
null until then.
