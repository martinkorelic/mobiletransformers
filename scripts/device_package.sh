#!/usr/bin/env bash
# W6 (#1-29 device-test provisioning): export a real package, reshape it into the on-device cache layout,
# and `adb push` it so the instrumented suites (which assumeTrue-skip without it) can run.
#
#   MODEL=<hf-id> [VARIANT=cpu-int4] [TRAIN=1] [RAG=1] [EMBEDDING_MODEL=<hf-id>] scripts/device_package.sh
#
# Steps: (1) inference+genai export under the `export` profile; (1b) optional training stage under
# `ort-training-local` (TRAIN=1) for a train-capable package; (2) reshape build/pkg (#14 variants/ tree)
# into <sanitizedRepoId>/{inference,train,embedding,tokenizer}; (3) adb push.
#
# Push destination: the instrumentation app's *external files dir*, NOT /data/local/tmp. `adb push`
# can write both, but /data/local/tmp is SELinux-labelled `shell_data_file` — the app domain cannot
# read it on a modern Android, and cannot write it at all, which the merge and checkpoint legs
# (TrainMergeGenerateTest) require. DeviceModel.cacheRoot() already probes this path. Override with
# DEVICE_DEST=... if you know what you are doing.
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf-id>, e.g. HuggingFaceTB/SmolLM2-135M-Instruct}"
VARIANT="${VARIANT:-cpu-int4}"
TRAIN="${TRAIN:-0}"
RAG="${RAG:-1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-}"
PKG="build/pkg"
DEVICE_CACHE="build/device_cache"
TEST_PKG="${TEST_PKG:-com.martinkorelic.mobiletransformers.test}"
DEVICE_DEST="${DEVICE_DEST:-/sdcard/Android/data/$TEST_PKG/files/mt_pkg}"
# The #10 spike suite probes its own dir (mt_genai_spike/inference) and had no provisioning path, so
# Gate 0.1 #2/#3/#5 could only ever skip after a device-package run.
SPIKE_DEST="${SPIKE_DEST:-/sdcard/Android/data/$TEST_PKG/files/mt_genai_spike}"
# Rough floor: the package itself plus the merged/checkpoint bytes the training legs write beside it.
MIN_FREE_MB="${MIN_FREE_MB:-4096}"

# --- preflight: fail here with a diagnosis, not three minutes into an export -----------------------
command -v adb >/dev/null || { echo "adb not found on PATH (install platform-tools)" >&2; exit 1; }

mapfile -t DEVICES < <(adb devices | awk 'NR>1 && $2=="device" {print $1}')
if [[ "${#DEVICES[@]}" -eq 0 ]]; then
  echo "no authorized device. Connect one, enable USB debugging, and accept the RSA prompt:" >&2
  adb devices -l >&2
  exit 1
fi
if [[ "${#DEVICES[@]}" -gt 1 && -z "${ANDROID_SERIAL:-}" ]]; then
  echo "${#DEVICES[@]} devices attached; set ANDROID_SERIAL=<serial> to pick one:" >&2
  adb devices -l >&2
  exit 1
fi

ABI="$(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
if [[ "$ABI" != "arm64-v8a" ]]; then
  echo "device ABI is '$ABI' but only arm64-v8a ships a complete jniLibs set" >&2
  echo "(x86_64 is missing libonnxruntime.so / libtokenizers_{c,cpp}.a — see docs/ARCHITECTURE.md)" >&2
  exit 1
fi

# `df -m` is not portable across Android toybox versions (Android 15 rejects it), and this probe must
# never be the thing that fails the run — report in MB from 1K blocks, and treat "unknown" as "proceed".
FREE_MB="$(adb shell df -k /sdcard 2>/dev/null | awk 'NR>1 {print int($4/1024); exit}' | tr -d '\r' || true)"
if [[ -n "$FREE_MB" && "$FREE_MB" -lt "$MIN_FREE_MB" ]]; then
  echo "device has ${FREE_MB}MB free on /sdcard; need >= ${MIN_FREE_MB}MB" >&2
  exit 1
fi
echo ">> device $(adb shell getprop ro.product.model | tr -d '\r') ($ABI, API $(adb shell getprop ro.build.version.sdk | tr -d '\r'), ${FREE_MB:-?}MB free)"

# --- 1. inference + genai export (export profile) --------------------------------------------------
RAG_ARGS=()
if [[ "$RAG" == "1" ]]; then
  RAG_ARGS+=(--include-rag)
  [[ -n "$EMBEDDING_MODEL" ]] && RAG_ARGS+=(--embedding-model "$EMBEDDING_MODEL")
fi

echo ">> [1/4] inference+genai export ($MODEL) under the export profile"
# --validate re-reads the written package against the #13 manifest contract, so a broken export fails
# here rather than as an unexplained skip on device.
uv run --extra export --python 3.12 mobiletransformers export \
  --model "$MODEL" --output "$PKG" --genai --validate "${RAG_ARGS[@]}"

if [[ "$TRAIN" == "1" ]]; then
  echo ">> [1b] training stage under ort-training-local (train-capable package)"
  # An explicit `uv sync` first, then `uv run --no-sync`. Step 1 above installs the export profile's
  # stock `onnxruntime` into the shared .venv, and `uv run --group ort-training-local` does NOT displace
  # it — the source-built training wheel provides a distribution of the same name, so the resolver
  # considers the requirement already satisfied. The training import then finds a runtime with no
  # training APIs and dies with `ImportError: cannot import name 'PropagateCastOpsStrategy'`. Running
  # the training stage on its own works, which is why this only shows up on the TRAIN=1 path.
  uv sync --python 3.12 --group ort-training-local --no-default-groups --reinstall-package onnxruntime-training
  uv run --no-sync --python 3.12 \
    mobiletransformers export --model "$MODEL" --output "$PKG" --stages training
fi

# --- 2. reshape into the on-device cache layout ----------------------------------------------------
# sanitized repo id = HF id with '/' -> '__' (mirrors PackageFormat.sanitizeRepoId).
SANITIZED="${MODEL//\//__}"
echo ">> [2/4] reshape $PKG/variants/$VARIANT -> $DEVICE_CACHE/$SANITIZED"
if [[ ! -d "$PKG/variants/$VARIANT" ]]; then
  echo "no variant '$VARIANT' in $PKG/variants (have: $(ls "$PKG/variants" 2>/dev/null | tr '\n' ' '))" >&2
  echo "set VARIANT=<id> to match the exported quantization" >&2
  exit 1
fi
rm -rf "$DEVICE_CACHE"
DEST="$DEVICE_CACHE/$SANITIZED"
mkdir -p "$DEST"
for stage in inference train embedding; do
  [[ -d "$PKG/variants/$VARIANT/$stage" ]] && cp -r "$PKG/variants/$VARIANT/$stage" "$DEST/$stage"
done
[[ -d "$PKG/shared/tokenizer" ]] && cp -r "$PKG/shared/tokenizer" "$DEST/tokenizer"

echo ">> staged: $(cd "$DEST" && ls -d */ 2>/dev/null | tr -d '/' | tr '\n' ' ')"

# --- 3. push ---------------------------------------------------------------------------------------
echo ">> [3/4] adb push -> $DEVICE_DEST"
adb shell "rm -rf '$DEVICE_DEST' && mkdir -p '$DEVICE_DEST'"
adb push "$DEVICE_CACHE/." "$DEVICE_DEST" >/dev/null
# `adb push` writes into the app's external dir as the `shell` user with mode 0770, so "other" — which
# is what the app resolves to for shell-owned entries — gets nothing, and File.canRead()/listFiles()
# return false/null on every pushed subdirectory. Widen so the app can use its own package.
#
# 777, not 775: in production the cache tree is created by ModelPackageInstaller and is app-owned, so
# the app can write inside it — the RAG vector store creates `embedding/database/`, and the merge and
# checkpoint legs write beside the weights. A read-only fixture would fail those for a reason that does
# not exist on a real install. chmod may be refused when the target is already app-owned; harmless.
adb shell "chmod -R 777 '$DEVICE_DEST' 2>/dev/null" || true
adb shell "ls '$DEVICE_DEST'"

echo ">> [3b] staging the #10 GenAI spike dir -> $SPIKE_DEST"
adb shell "rm -rf '$SPIKE_DEST' && mkdir -p '$SPIKE_DEST'"
adb push "$DEST/inference" "$SPIKE_DEST/inference" >/dev/null
adb shell "chmod -R 777 '$SPIKE_DEST' 2>/dev/null" || true

echo ">> [4/4] done. Run the instrumented suites:"
echo "   make device-test"
