# third_party/wheels/

Holds local, **git-ignored** Python wheels that are not available on public PyPI — currently the
source-built ONNX Runtime Training wheel.

- `onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl` — referenced by
  `[tool.uv.sources] onnxruntime-training` in the root `pyproject.toml`. Provenance and SHA256 live in
  [`../onnxruntime/manifest.json`](../onnxruntime/manifest.json); build steps in
  [`../onnxruntime/BUILD.md`](../onnxruntime/BUILD.md).

`*.whl` here is ignored by git (see root `.gitignore`). To obtain the wheel:

- **Fetch the published build** (what you almost certainly want) —
  `TRAINING=1 scripts/fetch_native_deps.sh`. It downloads the wheel and verifies its sha256 against
  [`../onnxruntime/manifest.json`](../onnxruntime/manifest.json) before installing it here.
- **Rebuild from source:** run `scripts/build_ort_training_wheel.sh` (see `BUILD.md`). Hours, and
  only necessary if you are changing ONNX Runtime itself.

The `ort-training-local` uv group is **cp312-only** — sync it under Python 3.12:
`uv sync --python 3.12 --group ort-training-local`.
