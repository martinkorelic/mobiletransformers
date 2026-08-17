#!/usr/bin/env bash
# Build a tiny GenAI-loadable model for the #10 spike (a real small decoder LM in ONNX Runtime GenAI
# format: model.onnx + model.onnx.data + genai_config.json + tokenizer). Standalone venv so it does NOT
# touch the uv-managed profiles. Output feeds desktop_spike.py and the device GenAISpikeTest.
#
#   ./build_tiny_genai_model.sh [out_dir] [hf_model_id] [precision]
#
# Defaults: out=build/genai_spike_model, model=HuggingFaceTB/SmolLM2-135M-Instruct, precision=int4, cpu.
set -euo pipefail

OUT="${1:-build/genai_spike_model}"
MODEL="${2:-HuggingFaceTB/SmolLM2-135M-Instruct}"
PREC="${3:-int4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv-genai-spike}"

cd "$REPO_ROOT"
if [[ ! -d "$VENV" ]]; then
  python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
# CPU torch + the genai model builder deps. onnxruntime-genai>=0.14 matches the device AAR.
python -m pip install --quiet \
  "onnxruntime-genai>=0.14" "onnxruntime>=1.20" "torch>=2.2" "transformers>=4.45" "onnx" \
  "onnx_ir" "onnxscript" "huggingface-hub>=0.24"

mkdir -p "$OUT"
CACHE="$REPO_ROOT/build/genai_spike_cache"
mkdir -p "$CACHE"
echo ">> building $MODEL ($PREC, cpu) -> $OUT"
python -m onnxruntime_genai.models.builder -m "$MODEL" -o "$OUT" -p "$PREC" -e cpu -c "$CACHE"

echo ">> done. GenAI package contents:"
ls -la "$OUT"
echo ">> genai_config.json present: $([[ -f "$OUT/genai_config.json" ]] && echo yes || echo NO)"
