#!/usr/bin/env bash
# Build the source ONNX Runtime *training* CPU wheel that this repo depends on.
#
# This is REFERENCE/PROVENANCE tooling. The wheel already exists (see
# third_party/onnxruntime/manifest.json); only rebuild when the ORT SHA or torch ABI must change.
# A full build needs tens of GB of scratch space and several minutes.
#
# Usage:
#   ORT_SRC=/path/to/onnxruntime PYTHON=python3.12 scripts/build_ort_training_wheel.sh
#
# Reads the target commit + build flags from third_party/onnxruntime/manifest.json.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_ROOT/third_party/onnxruntime/manifest.json"
WHEELS_DIR="$REPO_ROOT/third_party/wheels"

ORT_SRC="${ORT_SRC:-$REPO_ROOT/../onnxruntime}"
PYTHON="${PYTHON:-python3.12}"

ORT_SHA="$("$PYTHON" -c "import json;print(json.load(open('$MANIFEST'))['ort_git_sha'])")"
TORCH_VER="$("$PYTHON" -c "import json;print(json.load(open('$MANIFEST'))['torch_version'])")"

echo "== ONNX Runtime training wheel build =="
echo "   ORT source : $ORT_SRC"
echo "   ORT commit : $ORT_SHA"
echo "   Python     : $($PYTHON --version)"
echo "   torch pin  : $TORCH_VER  (must match the built ABI)"

if [ ! -d "$ORT_SRC" ]; then
  echo "ERROR: ORT source not found at $ORT_SRC (set ORT_SRC=...)." >&2
  exit 1
fi

# Pin torch to the recorded ABI before building so training links against the right libtorch.
"$PYTHON" -m pip install "torch==$TORCH_VER"

git -C "$ORT_SRC" fetch --all --tags
git -C "$ORT_SRC" checkout "$ORT_SHA"

# CPU training build. See third_party/onnxruntime/BUILD.md for the full flag rationale.
"$ORT_SRC/build.sh" \
  --config Release \
  --enable_training_apis \
  --build_wheel \
  --parallel \
  --skip_tests

mkdir -p "$WHEELS_DIR"
BUILT_WHL="$(find "$ORT_SRC/build" -name 'onnxruntime_training-*.whl' | head -1)"
if [ -z "$BUILT_WHL" ]; then
  echo "ERROR: no onnxruntime_training wheel produced under $ORT_SRC/build." >&2
  exit 1
fi
cp "$BUILT_WHL" "$WHEELS_DIR/"
echo "Copied $(basename "$BUILT_WHL") -> $WHEELS_DIR/"
echo "SHA256: $(sha256sum "$WHEELS_DIR/$(basename "$BUILT_WHL")" | cut -d' ' -f1)"
echo "Now update third_party/onnxruntime/manifest.json (wheel.sha256, ort_git_sha, python_version, torch_version)."
