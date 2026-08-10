#!/usr/bin/env bash
# #36 device round-trip: export adapter factors on REAL hardware -> aggregate on the host -> import the
# aggregate back into the device checkpoint.
#
#   [PKG=build/pkg] [SCALE=3.0] scripts/federated_round_device.sh
#
# The middle of a federated round is a host process, so the seam cannot be crossed inside one
# instrumentation run. This drives both halves and the host step between them:
#
#   phase 1 (device)  export the local factors from a live ORT checkpoint  -> client_update.bin
#   adb pull          bring the record to the host
#   peer record       derive a synthetic SECOND client (scaled) — see federated_peer_record.py for why
#   federated serve   FedAvg the two into a global record                  -> global_record.bin
#   adb push          hand the aggregate back to the device
#   phase 2 (device)  import it, assert every checkpoint tensor now holds the aggregate's bytes,
#                     then train+export one more round on top of it
#
# `BuildConfig.FEDERATION_ENABLED` is FALSE by default and that refusal is the feature (#36 privacy
# gate), so the instrumentation runs are invoked with `-PmtFederationEnabled=true` — deliberately, per
# invocation, never persisted into the tree.
set -euo pipefail

PKG="${PKG:-build/pkg}"
OUT="${OUT:-build/federated_round}"
SCALE="${SCALE:-3.0}"
TEST_PKG="${TEST_PKG:-com.martinkorelic.mobiletransformers.test}"
DEVICE_FED="${DEVICE_FED:-/sdcard/Android/data/$TEST_PKG/files/mt_pkg/federated}"
JAVA_HOME="${JAVA_HOME:-/opt/android-studio/jbr}"
TEST_CLASS="com.martinkorelic.mobiletransformers.FederatedRoundDeviceTest"

gradle_test() {
  local method="$1"
  (cd android/MobileTransformers && JAVA_HOME="$JAVA_HOME" ./gradlew \
    :MobileTransformers:connectedDebugAndroidTest \
    -PmtFederationEnabled=true \
    "-Pandroid.testInstrumentationRunnerArguments.class=$TEST_CLASS#$method")
}

# --- preflight ------------------------------------------------------------------------------------
command -v adb >/dev/null || { echo "adb not found on PATH (install platform-tools)" >&2; exit 1; }
mapfile -t DEVICES < <(adb devices | awk 'NR>1 && $2=="device" {print $1}')
if [[ "${#DEVICES[@]}" -eq 0 ]]; then
  echo "no authorized device; connect one and accept the RSA prompt" >&2
  exit 1
fi
if [[ ! -f "$PKG/mobiletransformers_manifest.json" ]]; then
  echo "no package at $PKG (set PKG=<dir>). The gateway needs the SAME package the device holds —" >&2
  echo "its weight_handoff_map.json is the authority on tensor names/shapes." >&2
  exit 1
fi

mkdir -p "$OUT"
# A stale record from an earlier run would let a SKIPPED phase look like a passing round.
rm -f "$OUT/client_update.bin" "$OUT/peer_update.bin" "$OUT/global_record.bin"
adb shell "rm -rf $DEVICE_FED" >/dev/null 2>&1 || true

# --- 1. device: export -----------------------------------------------------------------------------
echo ">> [1/5] phase 1 on device: export adapter factors from the live checkpoint"
gradle_test phase1ExportsAnUpdateFromTheRealCheckpoint

echo ">> [2/5] pull the client record"
adb pull "$DEVICE_FED/client_update.bin" "$OUT/client_update.bin" >/dev/null || {
  echo "phase 1 wrote no record. It skips (rather than fails) without a train-capable package —" >&2
  echo "run: make device-package MODEL=<hf-id> TRAIN=1" >&2
  exit 1
}
echo "   client record: $(stat -c%s "$OUT/client_update.bin") B"

# --- 2. host: second client + aggregate ------------------------------------------------------------
echo ">> [3/5] derive a synthetic peer record (x$SCALE) and aggregate both"
uv run python scripts/federated_peer_record.py \
  --package "$PKG" --input "$OUT/client_update.bin" --output "$OUT/peer_update.bin" --scale "$SCALE"

# Example weights 1 and 3 -> the FedAvg mean is (1*v + 3*SCALE*v)/4, a value NEITHER client submitted.
uv run mobiletransformers federated serve \
  --package "$PKG" \
  --updates "device:$OUT/client_update.bin:1" "peer:$OUT/peer_update.bin:3" \
  --min-clients 2 --round 1 --output "$OUT/global_record.bin"

# --- 3. device: import -----------------------------------------------------------------------------
echo ">> [4/5] push the global record back"
adb shell "mkdir -p $DEVICE_FED"
adb push "$OUT/global_record.bin" "$DEVICE_FED/global_record.bin" >/dev/null
adb shell "chmod -R 777 $DEVICE_FED" || true

echo ">> [5/5] phase 2 on device: import the aggregate, then train+export on top of it"
gradle_test phase2ImportsTheAggregateIntoTheRealCheckpoint

echo
echo "round complete. Payload sizes:"
ls -l "$OUT"/*.bin | awk '{print "  " $NF ": " $5 " B"}'
echo "Device-side measurements are in the instrumentation log:"
echo "  adb logcat -d -s FederatedRoundDeviceTest"
