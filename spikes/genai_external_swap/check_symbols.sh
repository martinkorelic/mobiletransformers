#!/usr/bin/env bash
# Gate 0.1 symbol check (#10): confirm OgaCreateModelWithInitializers is fork-only (ABSENT) and the stable
# OgaCreateModel is PRESENT in the *linked Android* onnxruntime-genai .so. Runs against the bundled AAR.
#
#   ./check_symbols.sh [path/to/onnxruntime-genai.aar]
#
# Defaults to the vendored aarLibs AAR. Exit 0 = PASS (fork-only confirmed), non-zero = FAIL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AAR="${1:-$REPO_ROOT/android/MobileTransformersApp/MobileTransformers/src/main/aarLibs/onnxruntime-genai.aar}"
ABI="${ABI:-arm64-v8a}"

if [[ ! -f "$AAR" ]]; then
  echo "FAIL: AAR not found at $AAR" >&2
  exit 2
fi

# Find an nm that reads ELF: prefer the NDK's llvm-nm, then any llvm-nm, then system nm.
NM="$(command -v llvm-nm || ls "$HOME"/Android/Sdk/ndk/*/toolchains/llvm/prebuilt/*/bin/llvm-nm 2>/dev/null | head -1 || command -v nm || true)"
if [[ -z "$NM" ]]; then
  echo "FAIL: no nm/llvm-nm available" >&2
  exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$AAR" "jni/$ABI/libonnxruntime-genai.so" -d "$WORK"
SO="$WORK/jni/$ABI/libonnxruntime-genai.so"

echo "AAR: $AAR"
echo "ABI: $ABI   nm: $NM"
echo "so:  $(du -h "$SO" | cut -f1)"

with_init="$("$NM" -D --defined-only "$SO" 2>/dev/null | grep -c 'OgaCreateModelWithInitializers' || true)"
create_model="$("$NM" -D --defined-only "$SO" 2>/dev/null | grep -c 'OgaCreateModel$' || true)"
generators="$("$NM" -D --defined-only "$SO" 2>/dev/null | grep -c 'OgaGenerator' || true)"

echo "OgaCreateModelWithInitializers : $with_init  (expect 0 -> fork-only)"
echo "OgaCreateModel                 : $create_model  (expect >=1 -> stable API)"
echo "OgaGenerator*                  : $generators  (expect >=1)"

fail=0
[[ "$with_init" -eq 0 ]] || { echo "FAIL: OgaCreateModelWithInitializers is present (not fork-only)" >&2; fail=1; }
[[ "$create_model" -ge 1 ]] || { echo "FAIL: OgaCreateModel missing (stable API absent)" >&2; fail=1; }
[[ "$generators" -ge 1 ]] || { echo "FAIL: no OgaGenerator symbols" >&2; fail=1; }

if [[ "$fail" -eq 0 ]]; then
  echo "PASS: OgaCreateModelWithInitializers is fork-only (absent); OgaCreateModel present."
fi
exit "$fail"
