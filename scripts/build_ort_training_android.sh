#!/usr/bin/env bash
# Build the ONNX Runtime *training* Android AAR + native .so libraries, at the SAME ORT commit as the
# Python training wheel (third_party/onnxruntime/manifest.json).
#
# PLACEHOLDER / REFERENCE: the Android training native build belongs to a later plan
# (agent_docs/00_code_plans/04, Android Gradle rename/migration). It is NOT run in the foundation
# pass and the manifest's Android fields (ndk_version, abis, android.*) are still null. This script
# records the intended build shape so the provenance is reproducible when that plan lands.
#
# Usage:
#   ORT_SRC=/path/to/onnxruntime ANDROID_NDK_HOME=/path/to/ndk scripts/build_ort_training_android.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_ROOT/third_party/onnxruntime/manifest.json"

ORT_SRC="${ORT_SRC:-$REPO_ROOT/../onnxruntime}"
ORT_SHA="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['ort_git_sha'])")"
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to your NDK path}"

# ABIs / API level to be finalized by 00_code_plans/04 and written back into manifest.json.
ABIS="${ABIS:-arm64-v8a}"
ANDROID_API="${ANDROID_API:-24}"

echo "== ONNX Runtime training Android build (reference) =="
echo "   ORT source : $ORT_SRC"
echo "   ORT commit : $ORT_SHA"
echo "   NDK        : $ANDROID_NDK_HOME"
echo "   ABIs       : $ABIS   API: $ANDROID_API"

if [ ! -d "$ORT_SRC" ]; then
  echo "ERROR: ORT source not found at $ORT_SRC (set ORT_SRC=...)." >&2
  exit 1
fi

git -C "$ORT_SRC" checkout "$ORT_SHA"

for abi in $ABIS; do
  "$ORT_SRC/build.sh" \
    --android \
    --android_sdk_path "${ANDROID_HOME:-$HOME/Android/Sdk}" \
    --android_ndk_path "$ANDROID_NDK_HOME" \
    --android_abi "$abi" \
    --android_api "$ANDROID_API" \
    --enable_training_apis \
    --build_java \
    --config Release \
    --parallel \
    --skip_tests
done

echo "Build complete. Record NDK version, ABIs, AAR/.so SHA256 into third_party/onnxruntime/manifest.json (android.*)."
