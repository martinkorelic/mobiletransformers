# third_party/wheels/

Holds local, **git-ignored** Python wheels that are not available on public PyPI — currently the
source-built ONNX Runtime Training wheel.

- `onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl` — referenced by
  `[tool.uv.sources] onnxruntime-training` in the root `pyproject.toml`. Provenance and SHA256 live in
  [`../onnxruntime/manifest.json`](../onnxruntime/manifest.json); build steps in
  [`../onnxruntime/BUILD.md`](../onnxruntime/BUILD.md).

`*.whl` here is ignored by git (see root `.gitignore`). To (re)obtain the wheel:

- **Reuse the existing build:** copy it from the sibling repo —
  `cp ../on_device_llm_finetune/dist/onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl .`
  then confirm `sha256sum` matches `manifest.json`.
- **Rebuild from source:** run `scripts/build_ort_training_wheel.sh` (see `BUILD.md`).

The `ort-training-local` uv group is **cp312-only** — sync it under Python 3.12:
`uv sync --python 3.12 --group ort-training-local`.
