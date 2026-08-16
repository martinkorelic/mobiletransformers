#!/usr/bin/env bash
# Export, verify and publish the catalog of models the showcase app offers.
#
#   scripts/publish_catalog.sh                 # all entries: export + verify + push
#   ONLY=smollm2 scripts/publish_catalog.sh    # one entry
#   PUSH=0 scripts/publish_catalog.sh          # export + verify, publish nothing
#   KEEP=1 scripts/publish_catalog.sh          # do not re-export an entry whose package already exists
#
# Every entry ships BOTH an inference and a training stage: fine-tuning on device is the point of the
# project, and a shelf entry that cannot be trained demonstrates half of it.
#
# Two things here are not obvious and are the reason this is a script rather than a list of commands:
#
# 1. THE TWO-PROFILE DANCE. The inference export needs the `export` extra; the training stage needs the
#    source-built `ort-training-local` wheel. They collide on the `onnxruntime` import and must never
#    co-install. `uv run --group ort-training-local` alone does NOT displace the stock onnxruntime the
#    export profile just installed — the training wheel provides a distribution of the same name, so
#    the resolver considers the requirement satisfied, and the training import then dies with
#    `ImportError: cannot import name 'PropagateCastOpsStrategy'`. An explicit `uv sync
#    --reinstall-package` followed by `uv run --no-sync` is what actually works.
#
# 2. TASK AND ENGINE FLAGS ARE PER-MODEL, and getting them wrong fails late or, worse, silently:
#    - `--task text-classification` is what makes an ENCODER trainable at all. `TaskSpec.default_stages`
#      emits a train/ stage exactly when the task is `trainable`, and FEATURE_EXTRACTION is declared
#      trainable=False — so exporting an encoder the "natural" way yields an inference-only package.
#      Task auto-selection never picks text-classification.
#    - `--genai` is DECODER-only. A classification/feature-extraction graph has no KV cache, and the
#      export refuses to write a genai_config.json describing a cache the graph does not have.
set -euo pipefail

cd "$(dirname "$0")/.."

PUSH="${PUSH:-1}"
ONLY="${ONLY:-}"
KEEP="${KEEP:-0}"
ORG="${ORG:-mobiletransformers}"
OUT_ROOT="${OUT_ROOT:-build/catalog}"

# key | base model | repo name | task ("" = auto) | rag (1/0) | genai (1/0)
#
# Keep this table in sync with the app's assets/model_catalog.json — the app claims sizes and features
# per entry, and a catalog that disagrees with what was pushed is worse than no catalog.
ENTRIES=(
  "smollm2|HuggingFaceTB/SmolLM2-135M-Instruct|SmolLM2-135M-Instruct||1|1"
  "qwen25|Qwen/Qwen2.5-0.5B-Instruct|Qwen2.5-0.5B-Instruct||1|1"
  "minilm|sentence-transformers/all-MiniLM-L6-v2|all-MiniLM-L6-v2|text-classification|1|0"
  "distilbert|distilbert-base-uncased-finetuned-sst-2-english|distilbert-sst2-english|text-classification|0|0"
)

log() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# The token is read the same way `mobiletransformers push` reads it, so an org push cannot silently
# authenticate as a different identity than the one this script reports.
if [[ "$PUSH" == "1" ]]; then
  [[ -f .env ]] && { set -a; . ./.env; set +a; }
  [[ -n "${HF_TOKEN:-}" ]] || fail "PUSH=1 but no HF_TOKEN (put it in .env, or run with PUSH=0)"
fi

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r KEY MODEL REPO TASK RAG GENAI <<<"$entry"
  [[ -n "$ONLY" && "$ONLY" != "$KEY" ]] && continue

  PKG="$OUT_ROOT/$KEY"
  log "[$KEY] $MODEL -> $ORG/$REPO"

  if [[ "$KEEP" == "1" && -f "$PKG/mobiletransformers_manifest.json" ]]; then
    log "[$KEY] KEEP=1 and a package already exists — skipping the export"
  else
    rm -rf "$PKG"

    TASK_ARGS=(); [[ -n "$TASK" ]] && TASK_ARGS=(--task "$TASK")
    RAG_ARGS=();  [[ "$RAG" == "1" ]] && RAG_ARGS=(--include-rag --embedding-model sentence-transformers/all-MiniLM-L6-v2)
    GENAI_ARGS=(); [[ "$GENAI" == "1" ]] && GENAI_ARGS=(--genai)

    log "[$KEY] 1/3 inference export (export profile)"
    uv run --extra export --python 3.12 mobiletransformers export \
      --model "$MODEL" --output "$PKG" --validate \
      "${TASK_ARGS[@]}" "${RAG_ARGS[@]}" "${GENAI_ARGS[@]}"

    log "[$KEY] 2/3 training stage (ort-training-local profile)"
    # See note 1 in the header: the explicit sync is load-bearing, not belt-and-braces.
    uv sync --python 3.12 --group ort-training-local --no-default-groups \
      --reinstall-package onnxruntime-training
    uv run --no-sync --python 3.12 mobiletransformers export \
      --model "$MODEL" --output "$PKG" --stages training "${TASK_ARGS[@]}"
  fi

  log "[$KEY] 3/3 verify"
  # Re-validate after the training stage: step 2 rewrites the manifest, and a package that validated
  # before the train/ stage was added says nothing about the one that will actually be pushed.
  uv run --group dev --python 3.10 mobiletransformers validate --package "$PKG"

  uv run --group dev --python 3.10 python - "$PKG" <<'PY'
import json, sys
from pathlib import Path

pkg = Path(sys.argv[1])
manifest = json.loads((pkg / "mobiletransformers_manifest.json").read_text())
features = {f for v in manifest.get("variants", []) for f in v.get("features", [])}

# The user's requirement, asserted rather than assumed: every published entry must be trainable.
if "train" not in features:
    raise SystemExit(
        f"{pkg}: no `train` group (features={sorted(features)}). An encoder needs an explicit "
        "--task text-classification: FEATURE_EXTRACTION is declared trainable=False, so the export "
        "emits no training stage and the package cannot be fine-tuned on device."
    )
if not (pkg / "variants").glob("*/train/training_config.json"):
    raise SystemExit(f"{pkg}: the train group is declared but training_config.json is missing")

size_mb = sum(manifest.get("fileSizes", {}).values()) / 1e6
inference_mb = sum(
    s for name, s in manifest.get("fileSizes", {}).items() if "/inference/" in name or "/tokenizer" in name
) / 1e6
print(f"    base            {manifest.get('baseModelId')}")
print(f"    task            {manifest.get('selectedTask')}")
print(f"    peft            {manifest.get('peftMethods')}")
print(f"    features        {sorted(features)}")
print(f"    total           {size_mb:.0f} MB")
print(f"    inference group {inference_mb:.0f} MB   <- approxSizeMb for the app catalog")
PY

  if [[ "$PUSH" == "1" ]]; then
    log "[$KEY] push -> $ORG/$REPO"
    # No --create: the repos are expected to exist. A mistyped id must fail rather than quietly make
    # a stray repo under the organisation.
    uv run --group dev --python 3.10 mobiletransformers push \
      --package "$PKG" --repo "$ORG/$REPO" --token "$HF_TOKEN"
  else
    log "[$KEY] PUSH=0 — rendering the card only"
    uv run --group dev --python 3.10 mobiletransformers push \
      --package "$PKG" --repo "$ORG/$REPO" --dry-run
  fi
done

log "done. Reset the profile before running the host suite: uv sync --frozen --group dev --python 3.10"
