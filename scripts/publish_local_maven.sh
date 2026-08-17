#!/usr/bin/env bash
# Publish the :MobileTransformers SDK to the local Maven repository (~/.m2) (#30).
#
#   scripts/publish_local_maven.sh [-Pversion=<x>]
#
# Publishes com.martinkorelic.mobiletransformers:mobiletransformers-android:<version> — AAR, sources
# jar and POM — so `examples/consumer-app` (or any external project with mavenLocal()) can resolve it.
set -euo pipefail

GRADLE_ROOT="android/MobileTransformers"

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/java_home.sh"

# Same ABI caveat as android_build_aar.sh: pass -Pandroid.injected.build.abi=<abi> when the
# vendored libraries are only complete for one ABI.
echo "==> publishing to mavenLocal"
(cd "${GRADLE_ROOT}" && ./gradlew :MobileTransformers:publishToMavenLocal "$@")

COORD_DIR="${HOME}/.m2/repository/com/martinkorelic/mobiletransformers/mobiletransformers-android"
if [ ! -d "${COORD_DIR}" ]; then
  echo "error: publish reported success but ${COORD_DIR} does not exist." >&2
  exit 1
fi

echo "==> published:"
find "${COORD_DIR}" -type f \( -name '*.aar' -o -name '*.pom' -o -name '*-sources.jar' \) \
  -printf '    %p\n' | sort
