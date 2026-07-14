#!/usr/bin/env bash
# Assemble the release AAR for the :MobileTransformers SDK module.
#
# STUB (created by #28 / 05_code_plans/01). The real body — release-variant assembly, native-lib
# packaging (jniLibs/aarLibs), and any signing — is owned by #30 (05_code_plans/03, AAR & Maven).
# Until then this fails closed rather than silently producing nothing.
set -euo pipefail

GRADLE_ROOT="android/MobileTransformersApp"

echo "scripts/android_build_aar.sh is a stub; the AAR assembly body is owned by #30 (05_code_plans/03)." >&2
echo "Interim: run '(cd ${GRADLE_ROOT} && ./gradlew :MobileTransformers:assembleRelease)' manually." >&2
exit 1
