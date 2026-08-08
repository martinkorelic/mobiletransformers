#!/usr/bin/env bash
# Assemble the release AAR for the :MobileTransformers SDK module (#30).
#
#   scripts/android_build_aar.sh [-Pversion=<x>]
#
# Requires JDK 17 and the Android SDK/NDK. The native build also needs the git-ignored vendored
# libraries under MobileTransformers/src/main/jniLibs/<abi>/ — they are provisioned out-of-band (same
# story as the ORT-training wheel), so this checks for them up front and says so rather than failing
# deep inside CMake.
set -euo pipefail

GRADLE_ROOT="android/MobileTransformers"
MODULE_DIR="${GRADLE_ROOT}/MobileTransformers"
JNI_LIBS="${MODULE_DIR}/src/main/jniLibs"

# Libraries the CMake link line requires from jniLibs/<abi>/ (see cpp/CMakeLists.txt).
REQUIRED_LIBS=(libonnxruntime.so libonnxruntime-genai.so libtokenizers_c.a libtokenizers_cpp.a)
# v1 ships arm64-v8a only, matching build.gradle.kts's abiFilters — jniLibs/x86_64 lacks
# libonnxruntime.so and the tokenizers archives, so libmobiletransformers.so cannot be built for it.
# Set ABIS="arm64-v8a x86_64" once those are vendored; the completeness check below then enforces it.
ABIS="${ABIS:-arm64-v8a}"

if [ ! -d "${JNI_LIBS}" ]; then
  echo "error: ${JNI_LIBS} is absent." >&2
  echo "       The native build needs the vendored ONNX Runtime / tokenizers libraries per ABI." >&2
  echo "       See docs/ARCHITECTURE.md and agent_docs/ for the provisioning story." >&2
  exit 1
fi

incomplete=0
for abi in ${ABIS}; do
  for lib in "${REQUIRED_LIBS[@]}"; do
    # .a/.so are interchangeable for some of these depending on how they were vendored.
    stem="${lib%.*}"
    if ! compgen -G "${JNI_LIBS}/${abi}/${stem}."* > /dev/null; then
      echo "error: ${JNI_LIBS}/${abi}/ is missing ${stem}.(so|a)" >&2
      incomplete=1
    fi
  done
done
if [ "${incomplete}" -ne 0 ]; then
  echo >&2
  echo "The vendored native libraries are incomplete for the requested ABIs (${ABIS})." >&2
  echo "A release AAR must ship arm64-v8a AND x86_64. To build a partial artifact for local" >&2
  echo "work only:  ABIS=arm64-v8a scripts/android_build_aar.sh -Pandroid.injected.build.abi=arm64-v8a" >&2
  exit 1
fi

: "${JAVA_HOME:=}"
if [ -z "${JAVA_HOME}" ] && [ -d /opt/android-studio/jbr ]; then
  export JAVA_HOME=/opt/android-studio/jbr
fi

echo "==> assembling the release AAR"
(cd "${GRADLE_ROOT}" && ./gradlew :MobileTransformers:assembleRelease "$@")

AAR="$(find "${MODULE_DIR}/build/outputs/aar" -name '*-release.aar' -print -quit 2>/dev/null || true)"
if [ -z "${AAR}" ]; then
  echo "error: assembleRelease reported success but produced no AAR." >&2
  exit 1
fi

echo "==> ${AAR}"

# An ABI directory in the AAR proves nothing on its own: the VENDORED libraries are packaged from
# jniLibs/<abi>/ regardless of whether our own library was built for that ABI. An AAR carrying
# jni/x86_64/ without libmobiletransformers.so fails at System.loadLibrary on the consumer's device.
# So verify the PROJECT's library per ABI, not just the directories.
present_abis="$(unzip -l "${AAR}" | awk '/\.so$/ {print $4}' | sed 's|/[^/]*$||;s|^jni/||' | sort -u)"
own_abis="$(unzip -l "${AAR}" | awk '/libmobiletransformers\.so$/ {print $4}' | sed 's|/[^/]*$||;s|^jni/||' | sort -u)"
echo "    ABI directories:            $(echo "${present_abis}" | paste -sd, -)"
echo "    libmobiletransformers.so:   $(echo "${own_abis}" | paste -sd, -)"

missing_own="$(comm -23 <(echo "${present_abis}") <(echo "${own_abis}"))"
if [ -n "${missing_own}" ]; then
  echo >&2
  echo "error: the AAR ships ABI directories with NO libmobiletransformers.so: $(echo "${missing_own}" | paste -sd, -)" >&2
  echo "       A consumer on that ABI would fail at System.loadLibrary(\"mobiletransformers\")." >&2
  echo "       Build every shipped ABI, or restrict the packaged ABIs (abiFilters) to what was built." >&2
  exit 1
fi
