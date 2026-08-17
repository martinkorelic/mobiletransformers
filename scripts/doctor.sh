#!/usr/bin/env bash
# One preflight report: every prerequisite, whether it is present, and the command that fixes it.
#
#   make doctor        (or: scripts/doctor.sh)
#
# Read-only. It downloads nothing, installs nothing and syncs no profile — safe to run at any time,
# including while an export is in flight.
#
# WHY. The prerequisites are spread across uv, two Python versions, a JDK, the Android SDK, ~180 MB of
# gitignored native binaries, a 662 MB source-built wheel and a `.env` token, and each one fails in a
# different tool with a message that names neither the prerequisite nor how to get it. A fresh clone
# could not build the Android SDK and the reason was undiscoverable. This is the single place that
# answers "what is missing".
#
# It always exits 0. A missing prerequisite is normal — most workflows need only a few of these — so
# this reports rather than gates. The `[--]` rows tell you what you cannot do yet.
set -uo pipefail

cd "$(dirname "$0")/.."

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'
MISSING=0

section() { printf '\n%s%s%s\n' "$BOLD" "$1" "$OFF"; }
ok()      { printf '  %s[ok]%s %-26s %s\n' "$GREEN" "$OFF" "$1" "${2:-}"; }
bad()     { MISSING=$((MISSING+1)); printf '  %s[--]%s %-26s %s\n' "$RED" "$OFF" "$1" "${2:-}"; \
            printf '       %sfix:%s %s\n' "$YELLOW" "$OFF" "$3"; }
note()    { printf '       %s%s%s\n' "$DIM" "$1" "$OFF"; }

section "Host toolchain"

if command -v uv >/dev/null 2>&1; then
  ok "uv" "$(uv --version 2>/dev/null)"
else
  bad "uv" "not on PATH" "curl -LsSf https://astral.sh/uv/install.sh | sh   (usually lands in ~/.local/bin)"
fi

for v in 3.10 3.12; do
  if command -v "python$v" >/dev/null 2>&1; then
    ok "python$v" "$(python$v -V 2>&1)"
  elif uv python find "$v" >/dev/null 2>&1; then
    ok "python$v" "available via uv"
  else
    case "$v" in
      3.10) bad "python3.10" "not found" "uv python install 3.10   — the core/dev profile (make check) targets it" ;;
      3.12) bad "python3.12" "not found" "uv python install 3.12   — required by the export and ORT-training profiles" ;;
    esac
  fi
done

if [ -d .venv ]; then
  ok ".venv" "$(.venv/bin/python -V 2>&1 || echo 'present but unusable')"
  # Which profile the shared venv is currently on. The single most common way to "break" the repo is
  # running `make check` on a leftover export/training profile.
  if .venv/bin/python -c "import onnxruntime.training" >/dev/null 2>&1; then
    note "profile: ort-training-local — reset before make check: uv sync --frozen --group dev --python 3.10"
  elif .venv/bin/python -c "import onnxruntime" >/dev/null 2>&1; then
    note "profile: export (or genai) — reset before make check: uv sync --frozen --group dev --python 3.10"
  else
    note "profile: core/dev — ready for make check"
  fi
else
  bad ".venv" "no environment yet" "make setup"
fi

section "Python: training profile (on-device fine-tuning exports)"

WHEEL=third_party/wheels/onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl
if [ -f "$WHEEL" ]; then
  ok "ORT-training wheel" "$(du -h "$WHEEL" | cut -f1)"
else
  bad "ORT-training wheel" "$WHEEL" \
      "TRAINING=1 scripts/fetch_native_deps.sh   (or source-build it: third_party/onnxruntime/BUILD.md)"
  note "cp312 + linux_x86_64 only. macOS/Windows must rebuild it before any training export."
  note "Without it: exports are inference-only; make test-train and make publish-catalog cannot run."
fi

section "Android: JDK + SDK"

if (. scripts/lib/java_home.sh) >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . scripts/lib/java_home.sh >/dev/null 2>&1
  ok "JAVA_HOME" "$JAVA_HOME ($("$JAVA_HOME/bin/java" -version 2>&1 | head -1))"
else
  bad "JAVA_HOME" "no JDK 17+ found" "export JAVA_HOME=/path/to/jdk17   (Android Studio ships one)"
fi

SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}"
if [ -d "$SDK" ]; then
  ok "Android SDK" "$SDK"
else
  bad "Android SDK" "not at $SDK" "install it via Android Studio, then: export ANDROID_HOME=/path/to/Sdk"
fi

LOCAL_PROPS=android/MobileTransformers/local.properties
if [ -f "$LOCAL_PROPS" ]; then
  ok "local.properties" "present"
elif [ -d "$SDK" ]; then
  ok "local.properties" "absent, but ANDROID_HOME resolves — Gradle will manage"
else
  bad "local.properties" "absent and no SDK found" "echo \"sdk.dir=/path/to/Sdk\" > $LOCAL_PROPS"
fi

if command -v adb >/dev/null 2>&1; then
  DEVICES="$(adb devices 2>/dev/null | grep -c 'device$')"
  if [ "$DEVICES" -ge 1 ]; then
    ok "adb" "$DEVICES authorized device(s)"
  else
    ok "adb" "on PATH, no device attached"
    note "Device targets (device-package / device-test / device-rss) need one; host gates do not."
  fi
else
  bad "adb" "not on PATH" "export PATH=\"\$PATH:$SDK/platform-tools\""
  note "Only the device targets need it."
fi

section "Android: vendored native dependencies"

# Delegated to the fetch script's own verifier so there is ONE definition of "provisioned" — a second
# copy of this list here is exactly how the two would drift.
if NATIVE_REPORT="$(scripts/fetch_native_deps.sh 2>&1)" && \
   printf '%s' "$NATIVE_REPORT" | grep -q "nothing to do"; then
  ok "jniLibs / aarLibs / includes" "all present and verified against third_party/android/manifest.json"
else
  N_MISSING="$(printf '%s' "$NATIVE_REPORT" | grep -c '^MISSING\|^CORRUPT')"
  bad "jniLibs / aarLibs / includes" "$N_MISSING artifact(s) missing or corrupt" \
      "scripts/fetch_native_deps.sh   (see third_party/android/manifest.json)"
  printf '%s\n' "$NATIVE_REPORT" | grep '^MISSING\|^CORRUPT' | head -12 | sed 's/^/       /'
  note "These are gitignored and are the ONLY thing a git clone does not bring."
  note "Without them the Android SDK cannot be built at all (make android-build / build-aar)."
fi

section "Hugging Face credentials"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
  ok ".env" "present"
else
  bad ".env" "absent" "cp .env.example .env   then fill in the tokens you need"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  ok "HF_TOKEN" "set (personal)"
else
  bad "HF_TOKEN" "unset" "add HF_TOKEN=… to .env  (see .env.example)"
  note "Needed for: exporting a gated base model, pulling a private package, the app's Install button."
fi

if [ -n "${HF_TOKEN_ORG:-}" ]; then
  ok "HF_TOKEN_ORG" "set (organisation)"
else
  bad "HF_TOKEN_ORG" "unset" "add HF_TOKEN_ORG=… to .env  (see .env.example)"
  note "Needed for: make publish-catalog. A personal token scoped to one repo cannot see the others."
fi

section "Summary"
if [ "$MISSING" -eq 0 ]; then
  printf '  %sEverything this repo knows how to check is present.%s\n\n' "$GREEN" "$OFF"
else
  printf '  %s%d prerequisite(s) missing%s — each is listed with its fix above.\n' "$YELLOW" "$MISSING" "$OFF"
  printf '  Most workflows need only some of them; see docs/ARCHITECTURE.md ▸ Native dependencies.\n\n'
fi
exit 0
