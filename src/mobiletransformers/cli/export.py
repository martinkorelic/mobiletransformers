"""`mobiletransformers export` — HF model -> device-ready #14 package (one-command export, #15).

Thin CLI over ``export.pipeline``. ``--dry-run`` resolves the plan + prints a manifest skeleton without
touching large files or heavy deps; a real run is env-gated (export + ORT-training profiles).
"""

from __future__ import annotations

import argparse
import json

from mobiletransformers.exceptions import MobileTransformersError


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("export", help="Export an HF model to a device-ready package.")
    parser.add_argument("--model", required=True, help="HF repo id to export.")
    parser.add_argument("--output", required=True, help="Output package directory.")
    parser.add_argument("--task", default=None, help="Optimum task (auto-selected if omitted).")
    parser.add_argument("--peft", default="lora", help="lora | lora-xs | mars | mars-opt0..mars-opt4.")
    parser.add_argument("--rank", type=int, default=8, help="LoRA/MARS rank (default 8).")
    parser.add_argument("--quant", default="int4", help="qint8 | int4 | fp16 (default int4).")
    parser.add_argument("--variant", default=None, help="Variant id (default cpu-<quant>).")
    parser.add_argument(
        "--include-rag", action="store_true", help="Also emit the embedding/RAG variant subtree."
    )
    parser.add_argument("--embedding-model", default=None, help="Embedding model id for RAG.")
    parser.add_argument("--genai", action="store_true", help="Declare GenAI engine support for the variant.")
    parser.add_argument("--config", default=None, help="Path to config YAML (overlay).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve + print the plan; write nothing.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    from mobiletransformers.export.pipeline import (
        ExportPlan,
        export_package,
        manifest_skeleton,
    )

    engines = ("native", "genai") if getattr(args, "genai", False) else ("native",)
    try:
        result = export_package(
            model=args.model,
            output=args.output,
            task=args.task,
            peft=args.peft,
            rank=args.rank,
            quant=args.quant,
            variant=args.variant,
            include_rag=args.include_rag,
            embedding_model=args.embedding_model,
            engines=engines,
            dry_run=args.dry_run,
        )
    except MobileTransformersError as exc:
        print(f"export failed: {exc}")
        return 1
    except NotImplementedError as exc:
        print(f"export (real run) is env-gated: {exc}")
        return 2

    if args.dry_run:
        assert isinstance(result, ExportPlan)
        skeleton = manifest_skeleton(result)
        print(
            f"[dry-run] would export {result.model_id} (task={result.task}, "
            f"peft={result.peft_method.value}, variant={result.variant_id}) -> {result.output_dir}"
        )
        print(json.dumps(skeleton, indent=2, sort_keys=True))
        return 0

    assert not isinstance(result, ExportPlan)
    print(f"exported package -> {result.output_dir} (manifest: {result.manifest_path})")
    return 0
