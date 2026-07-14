#!/usr/bin/env bash
# Publish the :MobileTransformers SDK library to the local Maven repository (~/.m2).
#
# STUB (created by #28 / 05_code_plans/01). The real body — the `maven-publish` Gradle wiring and
# `publishToMavenLocal` invocation — is owned by #30 (05_code_plans/03, AAR & Maven).
set -euo pipefail

GRADLE_ROOT="android/MobileTransformersApp"

echo "scripts/publish_local_maven.sh is a stub; the mavenLocal publication is owned by #30 (05_code_plans/03)." >&2
echo "Interim: once maven-publish is wired, run '(cd ${GRADLE_ROOT} && ./gradlew publishToMavenLocal)'." >&2
exit 1
