#!/usr/bin/env bash
# Download and install the Android native dependencies a `git clone` does not bring.
#
#   scripts/fetch_native_deps.sh              # the natives bundle (what you need to build)
#   TRAINING=1 scripts/fetch_native_deps.sh   # also the source-built ORT-training wheel (~632 MB)
#   SYMBOLS=1 scripts/fetch_native_deps.sh    # also the unstripped debug symbols (~260 MB)
#   URL=file:///path/to/dir scripts/fetch_native_deps.sh   # a local mirror, or an already-downloaded copy
#   FORCE=1 scripts/fetch_native_deps.sh      # re-install even if every file already verifies
#
# `third_party/android/manifest.json` is the source of truth for what to fetch, where it goes and what
# it must hash to. This script does not hardcode a single filename.
#
# WHY THIS EXISTS. ~180 MB of prebuilt binaries and vendored headers are gitignored, so a fresh clone
# cannot build the Android SDK at all — and before this script the failure was undiscoverable: CMake
# reported a missing link input naming no provenance, and `android_build_aar.sh` pointed at a docs
# page that never mentioned jniLibs. See docs/ARCHITECTURE.md ▸ Native dependencies.
#
# TWO HASH CHECKS, DELIBERATELY. The archive sha256 proves the download; the per-file sha256s prove
# the unpack. A half-populated jniLibs/ is the failure this guards: the link then fails naming a
# symbol, not a missing file, and that is an afternoon.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
MANIFEST="third_party/android/manifest.json"
SYMBOLS="${SYMBOLS:-0}"
TRAINING="${TRAINING:-0}"
FORCE="${FORCE:-0}"

log()  { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m   %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m!! %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f "$MANIFEST" ]] || fail "$MANIFEST not found — are you in the repo root?"
command -v python3 >/dev/null || fail "python3 is required (it reads the manifest and verifies hashes)"
command -v curl    >/dev/null || fail "curl is required"
command -v tar     >/dev/null || fail "tar is required"
tar --zstd --help >/dev/null 2>&1 || fail "your tar cannot read .zst — install zstd (apt install zstd)"

q() { python3 -c "import json,sys;print(json.load(open('$MANIFEST'))$1)"; }

UNPACK_ROOT="$(q "['unpackRoot']")"
BASE_URL="${URL:-$(q "['baseUrl'] or ''")}"

# --- is anything actually missing? -----------------------------------------------------------------
# Reported before downloading, so a no-op run costs nothing and says so.
verify_installed() {
  python3 - "$REPO_ROOT" "$MANIFEST" <<'PY'
import hashlib, json, sys
from pathlib import Path

root, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
base = root / manifest["unpackRoot"]

missing, corrupt = [], []
for art in manifest["artifacts"]:
    path = base / art["path"]
    if not path.is_file():
        missing.append(art["path"])
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != art["sha256"]:
        corrupt.append((art["path"], art["sha256"], digest))

for d in manifest.get("directories", []):
    if not (base / d["path"]).is_dir():
        missing.append(d["path"] + "/")

for name in missing:
    print(f"MISSING {name}")
for name, want, got in corrupt:
    print(f"CORRUPT {name}\n        expected {want}\n        actual   {got}")
sys.exit(1 if (missing or corrupt) else 0)
PY
}

# Resolve the wheel's identity up front — needed both to decide whether to fetch it and to verify a
# copy that is already there. Its filename and sha256 live in the ORT build's own provenance record.
WHEEL_MANIFEST="third_party/onnxruntime/manifest.json"
WHEEL_FILE="$(python3 -c "import json;print(json.load(open('$WHEEL_MANIFEST'))['wheel']['filename'])")"
WHEEL_WANT="$(python3 -c "import json;print(json.load(open('$WHEEL_MANIFEST'))['wheel']['sha256'])")"
WHEEL_DEST="third_party/wheels/$WHEEL_FILE"

log "checking what is already installed under $UNPACK_ROOT"
NEED_NATIVES=0
verify_installed || NEED_NATIVES=1
[[ "$FORCE" == "1" || "$SYMBOLS" == "1" ]] && NEED_NATIVES=1

NEED_WHEEL=0
if [[ "$TRAINING" == "1" ]]; then
  if [[ -f "$WHEEL_DEST" ]] && [[ "$(sha256sum "$WHEEL_DEST" | cut -d' ' -f1)" == "$WHEEL_WANT" ]]; then
    log "[wheel] already present and verifies"
  else
    NEED_WHEEL=1
  fi
fi

if [[ "$NEED_NATIVES" == "0" && "$NEED_WHEEL" == "0" ]]; then
  log "everything requested is present and verifies — nothing to do (FORCE=1 to reinstall)"
  exit 0
fi

# A URL is only required once something actually has to be downloaded. Checked here, AFTER deciding,
# so a fully-provisioned tree reports itself green instead of failing on a baseUrl it never needed —
# which is the state of every developer machine that predates this script.
if [[ -z "$BASE_URL" ]]; then
  fail "something is missing (above), and there is no download URL.

  third_party/android/manifest.json has \`baseUrl: null\` — the artifacts have been built but not yet
  hosted anywhere, so there is nothing for this script to fetch.

  If you have the files already:
      URL=file:///path/to/their/directory scripts/fetch_native_deps.sh

  If you are the maintainer: host build/dist/*.tar.zst (and the ORT-training wheel, for TRAINING=1)
  and set \`baseUrl\` in the manifest to the directory they live under. A GitHub Release keeps the
  URL stable across tags."
fi

# --- fetch ------------------------------------------------------------------------------------------
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

bundle_count="$(q "['bundles'].__len__()")"
[[ "$NEED_NATIVES" == "1" ]] && for i in $(seq 0 $((bundle_count - 1))); do
  NAME="$(q "['bundles'][$i]['name']")"
  FILE="$(q "['bundles'][$i]['filename']")"
  WANT="$(q "['bundles'][$i]['sha256']")"
  REQUIRED="$(q "['bundles'][$i]['required']")"

  if [[ "$REQUIRED" != "True" && "$SYMBOLS" != "1" ]]; then
    warn "skipping optional bundle '$NAME' (SYMBOLS=1 to fetch it)"
    continue
  fi

  log "[$NAME] downloading $FILE"
  curl -fL --progress-bar -o "$STAGE/$FILE" "${BASE_URL%/}/$FILE" \
    || fail "download failed: ${BASE_URL%/}/$FILE"

  log "[$NAME] verifying archive sha256"
  GOT="$(sha256sum "$STAGE/$FILE" | cut -d' ' -f1)"
  # Refuse rather than half-populate: unpacking an archive that failed its hash would leave the tree
  # in a state this script's own verify step then blames on the unpack.
  [[ "$GOT" == "$WANT" ]] || fail "sha256 mismatch for $FILE
    expected $WANT
    actual   $GOT
  Nothing was unpacked. Re-download, or check that URL points at the right release."

  log "[$NAME] unpacking into $UNPACK_ROOT"
  mkdir -p "$UNPACK_ROOT"
  tar --zstd -xf "$STAGE/$FILE" -C "$UNPACK_ROOT"
done

# --- the ORT-training wheel (optional; only the training side needs it) -----------------------------
#
# Not part of the natives bundle and not an Android artifact: it is a source-built CPython wheel
# (cp312, linux_x86_64) that the EXPORT host needs to emit a training stage. Its filename and sha256
# are recorded in third_party/onnxruntime/manifest.json, which is the provenance record for the ORT
# build itself, so they are read from there rather than duplicated.
if [[ "$NEED_WHEEL" == "1" ]]; then
  log "[wheel] downloading $WHEEL_FILE (~632 MB)"
  mkdir -p third_party/wheels
  curl -fL --progress-bar -o "$STAGE/$WHEEL_FILE" "${BASE_URL%/}/$WHEEL_FILE" \
    || fail "download failed: ${BASE_URL%/}/$WHEEL_FILE"
  GOT="$(sha256sum "$STAGE/$WHEEL_FILE" | cut -d' ' -f1)"
  [[ "$GOT" == "$WHEEL_WANT" ]] || fail "sha256 mismatch for $WHEEL_FILE
    expected $WHEEL_WANT
    actual   $GOT
  Nothing was installed."
  # Move only after the hash passes: a partially-written wheel makes every `uv run` fail with a
  # metadata error that names the cache, not the download.
  mv "$STAGE/$WHEEL_FILE" "$WHEEL_DEST"
  log "[wheel] installed -> $WHEEL_DEST"
fi

# --- verify the unpack ------------------------------------------------------------------------------
log "verifying every artifact against the manifest"
if verify_installed; then
  log "done — $UNPACK_ROOT is fully provisioned. Next: make android-build"
else
  fail "the unpack did not produce every artifact the manifest declares (listed above).
  This means the bundle and the manifest disagree — report it rather than working around it."
fi
