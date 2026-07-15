#!/usr/bin/env bash
# W6 (#1-29 device-test provisioning): export a real package, reshape it into the on-device cache layout,
# and `adb push` it so the instrumented suites (which assumeTrue-skip without it) can run.
#
#   MODEL=<hf-id> [VARIANT=cpu-int4] [TRAIN=1] scripts/device_package.sh
#
# Steps: (1) inference+genai export under the `export` profile; (2) optional training stage under
# `ort-training-local` (TRAIN=1) for a train-capable package; (3) reshape build/pkg (#14 variants/ tree)
# into <sanitizedRepoId>/{inference,train,embedding,tokenizer}; (4) adb push to /data/local/tmp/mt_pkg.
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf-id>, e.g. HuggingFaceTB/SmolLM2-135M-Instruct}"
VARIANT="${VARIANT:-cpu-int4}"
TRAIN="${TRAIN:-0}"
PKG="build/pkg"
DEVICE_CACHE="build/device_cache"
DEVICE_DEST="/data/local/tmp/mt_pkg"

echo ">> [1/4] inference+genai export ($MODEL) under the export profile"
uv run --extra export --python 3.12 mobiletransformers export --model "$MODEL" --output "$PKG" --genai

if [[ "$TRAIN" == "1" ]]; then
  echo ">> [1b] training stage under ort-training-local (train-capable package)"
  uv run --group ort-training-local --python 3.12 --no-default-groups \
    mobiletransformers export --model "$MODEL" --output "$PKG" --stages training
fi

# sanitized repo id = HF id with '/' -> '__' (mirrors PackageFormat.sanitizeRepoId).
SANITIZED="${MODEL//\//__}"
echo ">> [2/4] reshape $PKG/variants/$VARIANT -> $DEVICE_CACHE/$SANITIZED"
rm -rf "$DEVICE_CACHE"
DEST="$DEVICE_CACHE/$SANITIZED"
mkdir -p "$DEST"
for stage in inference train embedding; do
  [[ -d "$PKG/variants/$VARIANT/$stage" ]] && cp -r "$PKG/variants/$VARIANT/$stage" "$DEST/$stage"
done
[[ -d "$PKG/shared/tokenizer" ]] && cp -r "$PKG/shared/tokenizer" "$DEST/tokenizer"

echo ">> [3/4] adb push -> $DEVICE_DEST"
adb shell "rm -rf $DEVICE_DEST && mkdir -p $DEVICE_DEST"
adb push "$DEVICE_CACHE/." "$DEVICE_DEST"

echo ">> [4/4] done. Run the instrumented suites:"
echo "   (cd android/MobileTransformersApp && JAVA_HOME=/opt/android-studio/jbr ./gradlew :MobileTransformers:connectedDebugAndroidTest)"
