#!/usr/bin/env bash
# Set up ORT engine separation so the Native/training engine and the GenAI engine coexist in one app
# (#10/#11). GenAI 0.14 needs stock ORT >=1.26; the Native/training engine needs the source-built
# ORT-training 1.23 — different version AND build, both with SONAME `libonnxruntime.so`. This script gives
# GenAI its own stock ORT under a distinct name so the two never collide:
#
#   • training ORT  -> stays  jniLibs/<abi>/libonnxruntime.so   (linked by libmobiletransformers.so)
#   • stock ORT     -> shipped jniLibs/<abi>/libort_gen.so      (SONAME raw-patched to libort_gen.so)
#   • genai .so     -> dlopen string raw-patched  libonnxruntime.so -> libort_gen.so
#
# Both export only ~3 symbols (hidden visibility) and genai resolves ORT via dlsym on its own handle, so
# there is no interposition. Idempotent; re-run to refresh. Requires network (Maven) + Python3.
set -euo pipefail

ORT_VER="${ORT_VER:-1.27.0}"          # stock ORT paired with onnxruntime-genai 0.14 (needs >=1.26)
ABI="${ABI:-arm64-v8a}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JNI="$REPO_ROOT/android/MobileTransformersApp/MobileTransformers/src/main/jniLibs/$ABI"
GENAI_SO="$JNI/libonnxruntime-genai.so"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

[[ -f "$GENAI_SO" ]] || { echo "FAIL: $GENAI_SO missing (extract it from the genai AAR first)" >&2; exit 2; }

echo ">> fetching stock onnxruntime-android $ORT_VER"
curl -fsSL -o "$WORK/ort.aar" \
  "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VER/onnxruntime-android-$ORT_VER.aar"
unzip -qo "$WORK/ort.aar" "jni/$ABI/libonnxruntime.so" -d "$WORK"

echo ">> raw-patching stock ORT SONAME -> libort_gen.so (no patchelf; preserves verneed/offsets)"
python3 - "$WORK/jni/$ABI/libonnxruntime.so" "$JNI/libort_gen.so" <<'PY'
import sys
b = bytearray(open(sys.argv[1],'rb').read())
old = b"libonnxruntime.so\x00"; new = b"libort_gen.so\x00" + b"\x00"*(len(old)-len(b"libort_gen.so\x00"))
assert len(new)==len(old)
open(sys.argv[2],'wb').write(b.replace(old,new))
PY

echo ">> raw-patching genai dlopen target: libonnxruntime.so -> libort_gen.so"
python3 - "$GENAI_SO" <<'PY'
import sys
p = sys.argv[1]; b = bytearray(open(p,'rb').read())
def patch(old,new):
    ob=old.encode()+b'\x00'; nb=new.encode()+b'\x00'; nb+=b'\x00'*(len(ob)-len(nb)); assert len(nb)==len(ob)
    n=b.count(ob); b[:]=b.replace(ob,nb); return n
n1=patch("libonnxruntime.so.1","libort_gen.so.1"); n2=patch("libonnxruntime.so","libort_gen.so")
open(p,'wb').write(b); print(f"   genai dlopen patched (.so.1 x{n1}, .so x{n2})")
PY

echo ">> done. jniLibs/$ABI now has:"
ls -la "$JNI" | grep -iE "libort_gen|libonnxruntime(-genai)?\.so$"
echo ">> NOTE: the genai AAR is NOT a Gradle dependency (its Java is unused); the patched .so ships from jniLibs."
