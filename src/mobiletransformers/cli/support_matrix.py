"""`mobiletransformers support-matrix` — generate model_support_matrix.json (#20).

Reads a candidate list (``--candidates`` JSON, else built-ins), optionally merges an
``android_probes.json`` (``--probes``), and writes the full matrix (``--out``) plus a filtered
user-facing view (``--docs``). Detection needs the export profile (transformers/optimum); without it,
pass ``--candidates`` with pre-resolved tasks is not supported — run under the export profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Fallback candidate families if no --candidates file is given (mirrors config.yml SUPPORT_MATRIX).
_DEFAULT_CANDIDATES = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-0.5B",
    "HuggingFaceTB/SmolLM2-135M",
]


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("support-matrix", help="Generate the model support matrix (#20).")
    parser.add_argument("--candidates", default=None, help="JSON file: list of ids or {modelId,...} objects.")
    parser.add_argument(
        "--probes", default=None, help="android_probes.json produced by device/CI instrumentation."
    )
    parser.add_argument(
        "--out", default="build/support/model_support_matrix.json", help="Full matrix output path."
    )
    parser.add_argument("--docs", default=None, help="Optional filtered user-facing matrix output path.")
    parser.add_argument("--generated-at", default=None, help="ISO timestamp to stamp into the matrix.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    from mobiletransformers.support.matrix import build_matrix, write_filtered_docs, write_matrix

    if args.candidates:
        candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    else:
        candidates = list(_DEFAULT_CANDIDATES)

    matrix = build_matrix(candidates, probes_path=args.probes, generated_at=args.generated_at)
    out = write_matrix(matrix, args.out)
    print(f"wrote support matrix ({len(matrix.models)} models) -> {out}")
    if args.docs:
        docs = write_filtered_docs(matrix, args.docs)
        print(f"wrote filtered user-facing matrix -> {docs}")
    return 0
