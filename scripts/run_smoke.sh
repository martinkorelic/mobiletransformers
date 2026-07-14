#!/usr/bin/env bash
# Tiny export/package/manifest wiring smoke — a thin wrapper over `make test-smoke`.
#
# Created by #28 (05_code_plans/01) so CI (#29) and `make test-smoke` share one entrypoint. Runs the
# core-runnable subset (export/package/manifest wiring). The full generate_artifacts leg is env-gated
# (ort-training-local, cp312) and lives in `make test-train`.
set -euo pipefail

exec make test-smoke
