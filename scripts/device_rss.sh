#!/usr/bin/env bash
# Gate 0.1 #4 + Gate 0.2 (#10/#12): collect the four-point RSS table and evaluate both gates.
#
#   scripts/device_rss.sh            # needs a device + a pushed package (make device-package)
#
# The two knobs live outside the test process, so the 2x2 table is four instrumented runs:
#   engine        -> MemoryRssTest.nativeFourPointTable / .genAiFourPointTable
#   weight load   -> `adb shell setprop debug.mtf.mmap_weights {0,1}` (an instrumented test cannot set
#                    an environment variable in the process it is measuring)
# Each run writes one JSON row into the app's external files dir; this pulls them and applies the
# thresholds ratified in agent_docs/01_tier0_foundation_decisions.md.
set -euo pipefail

GRADLE_ROOT="android/MobileTransformers"
TEST_PKG="${TEST_PKG:-com.martinkorelic.mobiletransformers.test}"
RSS_REMOTE="/sdcard/Android/data/$TEST_PKG/files/mt_rss"
OUT="${OUT:-build/rss}"
export JAVA_HOME="${JAVA_HOME:-/opt/android-studio/jbr}"

command -v adb >/dev/null || { echo "adb not found on PATH" >&2; exit 1; }
test "$(adb devices | grep -c 'device$')" -ge 1 || { echo "no authorized device" >&2; exit 1; }

adb shell "rm -rf '$RSS_REMOTE'" || true

for mode in 0 1; do
  echo ">> weight-load path: $([ "$mode" = 1 ] && echo mmap || echo copy)"
  adb shell setprop debug.mtf.mmap_weights "$mode"
  (cd "$GRADLE_ROOT" && ./gradlew :MobileTransformers:connectedDebugAndroidTest \
      -Pandroid.testInstrumentationRunnerArguments.class=com.martinkorelic.mobiletransformers.MemoryRssTest) \
    || echo ">> (run reported failures; rows already written are still collected)"
done
# Leave the device on the shipping default.
adb shell setprop debug.mtf.mmap_weights 0

rm -rf "$OUT"; mkdir -p "$OUT"
adb pull "$RSS_REMOTE" "$OUT" >/dev/null 2>&1 || true
find "$OUT" -name '*.json' -exec mv {} "$OUT"/ \; 2>/dev/null || true

python3 - "$OUT" <<'PY'
import json, pathlib, sys

rows = {}
for p in pathlib.Path(sys.argv[1]).rglob("*.json"):
    r = json.loads(p.read_text())
    rows[(r["engine"], bool(r["mmapWeights"]))] = r

if not rows:
    sys.exit("no RSS rows collected — did MemoryRssTest skip? (needs a pushed package)")

print(f"\n{'engine':8} {'load':6} {'pre':>10} {'postLoad':>10} {'post1tok':>10} {'postRel':>10} {'peak':>10}")
for (engine, mmap), r in sorted(rows.items()):
    print(f"{engine:8} {'mmap' if mmap else 'copy':6} {r['preLoadKb']:>10} {r['postWeightLoadKb']:>10} "
          f"{r['postFirstTokenKb']:>10} {r['postReleaseKb']:>10} {r['peakKb']:>10}   (kB)")

failures = []

# Gate 0.1 #4 — GenAI peak vs the Native baseline, on the shipping (copy) path.
nat, gen = rows.get(("native", False)), rows.get(("genai", False))
if nat and gen:
    allowed = max(nat["peakKb"] * nat["acceptedRssDeltaRatio"], nat["acceptedRssDeltaFloorKb"])
    delta = gen["peakKb"] - nat["peakKb"]
    ok = delta <= allowed
    print(f"\nGate 0.1 #4: GenAI peak - Native peak = {delta} kB, allowed {int(allowed)} kB -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("Gate 0.1 #4")
else:
    print("\nGate 0.1 #4: NOT EVALUATED (need both engines on the copy path)")

# Gate 0.2 — mmap must cut peak RSS by the ratified margin, per engine.
for engine in ("native", "genai"):
    copy_row, mmap_row = rows.get((engine, False)), rows.get((engine, True))
    if not (copy_row and mmap_row):
        print(f"Gate 0.2 [{engine}]: NOT EVALUATED (need both weight-load paths)")
        continue
    required = copy_row["gate02RequiredReduction"]
    reduction = 1 - (mmap_row["peakKb"] / copy_row["peakKb"])
    ok = reduction >= required
    print(f"Gate 0.2 [{engine}]: peak reduction {reduction:.1%}, required {required:.0%} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"Gate 0.2 [{engine}]")

# A failed gate is a real result to record, not a broken run — the mmap experiment is explicitly
# allowed to come back negative (01_code_plans/04). Report, do not exit non-zero.
print("\n" + ("all evaluated gates PASS" if not failures else "gates not met: " + ", ".join(failures)))
PY

echo ">> rows in $OUT"
