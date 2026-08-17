"""`mobiletransformers export` — HF model -> device-ready #14 package (one-command export, #15).

Thin CLI over ``export.pipeline``. ``--dry-run`` resolves the plan + prints a manifest skeleton without
touching large files or heavy deps; a real run is env-gated (export + ORT-training profiles).

``--config`` supplies defaults for any knob not given on the command line (precedence: CLI > YAML >
default, as documented in ``docs/EXPORT.md``). ``--validate`` re-reads the package that was just
written and runs the #13 manifest validation over it, so a broken export fails the command rather than
being discovered later on device.
"""

from __future__ import annotations

import argparse
import json

from mobiletransformers.exceptions import MobileTransformersError


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("export", help="Export an HF model to a device-ready package.")
    # Not argparse-required: --config may supply either. run() enforces that one of the two
    # sources provided them, with a message naming the missing flag.
    parser.add_argument("--model", default=None, help="HF repo id to export.")
    parser.add_argument("--output", default=None, help="Output package directory.")
    parser.add_argument("--task", default=None, help="Optimum task (auto-selected if omitted).")
    parser.add_argument("--peft", default="lora", help="lora | lora-xs | mars | mars-opt0..mars-opt4.")
    parser.add_argument("--rank", type=int, default=8, help="LoRA/MARS rank (default 8).")
    parser.add_argument("--quant", default="int4", help="qint8 | int4 | fp16 (default int4).")
    parser.add_argument("--variant", default=None, help="Variant id (default cpu-<quant>).")
    parser.add_argument(
        "--peft-target",
        default=None,
        help=(
            "Comma-separated modules PEFT adapts (e.g. 'q_proj,v_proj'). Omit to use the architecture "
            "registry's row for the model, which is the per-model default and the place to add a new "
            "architecture."
        ),
    )
    parser.add_argument(
        "--include-rag", action="store_true", help="Also emit the embedding/RAG variant subtree."
    )
    parser.add_argument("--embedding-model", default=None, help="Embedding model id for RAG.")
    parser.add_argument("--genai", action="store_true", help="Declare GenAI engine support for the variant.")
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages to build: inference,training,embedding (default: auto by profile).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML supplying defaults for any flag not passed (CLI > YAML > default).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the written package against the manifest contract before returning.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve + print the plan; write nothing.")
    parser.set_defaults(func=run)
    return parser


#: CLI dest -> its argparse default. A dest still holding its default is "unset", so the YAML overlay
#: may supply it; anything the user typed wins.
_OVERLAYABLE: dict[str, object] = {
    "model": None,
    "output": None,
    "task": None,
    "peft": "lora",
    "rank": 8,
    "quant": "int4",
    "variant": None,
    "peft_target": None,
    "include_rag": False,
    "embedding_model": None,
    "genai": False,
    "stages": None,
}


def _apply_config_overlay(args: argparse.Namespace) -> None:
    """Fill unset knobs from ``--config``'s ``export:`` block (or its top level).

    The flag was previously accepted and silently ignored while ``docs/EXPORT.md`` documented it as a
    working overlay — so a user's YAML had no effect and nothing said so.
    """
    if not getattr(args, "config", None):
        return
    from mobiletransformers.utils.yaml import load_config_from_file

    document = load_config_from_file(args.config) or {}
    if not isinstance(document, dict):
        raise MobileTransformersError(f"{args.config}: expected a YAML mapping at the top level")
    section = document.get("export", document)
    if not isinstance(section, dict):
        raise MobileTransformersError(f"{args.config}: 'export' must be a mapping")

    unknown = sorted(set(section) - set(_OVERLAYABLE))
    if unknown:
        raise MobileTransformersError(
            f"{args.config}: unknown export key(s) {unknown}; supported: {sorted(_OVERLAYABLE)}"
        )
    for dest, default in _OVERLAYABLE.items():
        if dest in section and getattr(args, dest, default) == default:
            setattr(args, dest, section[dest])


def run(args: argparse.Namespace) -> int:
    from mobiletransformers.export.pipeline import (
        ExportPlan,
        export_package,
        manifest_skeleton,
    )

    try:
        _apply_config_overlay(args)
    except MobileTransformersError as exc:
        print(f"export failed: {exc}")
        return 1
    for required in ("model", "output"):
        if not getattr(args, required, None):
            print(f"export failed: --{required} is required (pass it, or set it in --config)")
            return 1

    engines = ("native", "genai") if getattr(args, "genai", False) else ("native",)
    stages = (
        {s.strip() for s in args.stages.split(",") if s.strip()} if getattr(args, "stages", None) else None
    )
    # Empty tuple -> the architecture registry decides. Accepts a comma-separated string (CLI) or an
    # already-split list (YAML), so a config file can write it as a natural list.
    raw_targets = getattr(args, "peft_target", None)
    if isinstance(raw_targets, str):
        peft_targets = tuple(t.strip() for t in raw_targets.split(",") if t.strip())
    else:
        peft_targets = tuple(raw_targets or ())
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
            peft_targets=peft_targets,
            dry_run=args.dry_run,
            stages=stages,
        )
    except MobileTransformersError as exc:
        print(f"export failed: {exc}")
        return 1

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

    if getattr(args, "validate", False):
        from mobiletransformers.cli.validate import validate_package

        try:
            validate_package(result.output_dir)
        except MobileTransformersError as exc:
            print(f"export wrote a package that does not validate: {exc}")
            return 1
        print(f"validated package at {result.output_dir}")
    return 0
