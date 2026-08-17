"""`mobiletransformers push-adapter` (#22) — export a trained adapter from the cache and publish it.

Gate (``adapter.convert.to_peft_layout``): a clean LoRA maps to a **PEFT-compatible** adapter (Mode 1,
`adapter_config.json` at root); everything else (all MARS) becomes a **MobileTransformers-native**
adapter (Mode 2, merged tensors + handoff map + `mobiletransformers_adapter.json`). Mirrors `cli/push.py`
(injectable `uploader`, fail-closed card assertions, Hub upload lazy-imported). `--peft-only` errors
instead of falling back to native.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mobiletransformers.adapter.convert import materialize_peft_weights, to_peft_layout
from mobiletransformers.adapter.export import AdapterPackage, export_adapter_from_cache
from mobiletransformers.adapter.model_card import assert_required_sections, render_adapter_card
from mobiletransformers.artifacts.package_paths import PackagePaths
from mobiletransformers.exceptions import ExportError, MobileTransformersError


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("push-adapter", help="Publish a trained adapter from the device cache.")
    parser.add_argument(
        "--cache-repo", required=True, help="Materialized cache repo dir (train/ + inference/)."
    )
    parser.add_argument("--repo-id", required=True, help="Target HF adapter repo id.")
    parser.add_argument(
        "--out", default=None, help="Build dir for the adapter (default <cache-repo>/.adapter-push)."
    )
    parser.add_argument(
        "--base-license", default="see upstream", help="Exact upstream base-model license string."
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument(
        "--peft-only", action="store_true", help="Error instead of falling back to native mode."
    )
    parser.add_argument("--dry-run", action="store_true", help="Build + validate the card; do not upload.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace, *, uploader: Callable[..., Any] | None = None) -> int:
    try:
        pkg = export_adapter_from_cache(args.cache_repo)
    except MobileTransformersError as exc:
        print(f"push-adapter failed: {exc}")
        return 1

    layout = to_peft_layout(pkg)
    if args.peft_only and layout is None:
        print(
            f"push-adapter --peft-only: {pkg.peft_method!r} does not map to a PEFT adapter "
            "(would be native mode)"
        )
        return 1
    mode = "peft" if layout is not None else "native"

    out = Path(args.out) if args.out else Path(args.cache_repo) / ".adapter-push"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if mode == "peft":
        assert layout is not None
        (out / "adapter_config.json").write_text(
            json.dumps(layout.adapter_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # A Mode-1 repo without adapter_model.safetensors is unusable — `PeftModel.from_pretrained`
        # fails on it — so this must NOT be skipped silently (it was, which shipped weightless pushes).
        # Materializing needs the `train` extra (torch + safetensors) and onnxruntime-training for the
        # default factor reader, so `materialize_peft_weights` raises ExportError outside that profile.
        # A --dry-run stays runnable in the core env and reports what is missing instead of failing.
        try:
            materialize_peft_weights(pkg, layout, str(out))
        except ExportError as exc:
            if not args.dry_run:
                raise MobileTransformersError(
                    f"cannot publish a Mode-1 (PEFT) adapter without adapter_model.safetensors: {exc}"
                ) from exc
            print(f"[dry-run] adapter_model.safetensors NOT materialized: {exc}")
    else:
        _build_native_subtree(pkg, out)

    card = render_adapter_card(pkg, mode=mode, base_model_license=args.base_license)
    assert_required_sections(card, pkg)
    (out / "README.md").write_text(card, encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] built {mode} adapter at {out}; card validated; not uploading.")
        return 0

    if uploader is None:
        from huggingface_hub import create_repo, upload_folder

        create_repo(args.repo_id, exist_ok=True, private=args.private)
        uploader = upload_folder
    uploader(repo_id=args.repo_id, folder_path=str(out))
    print(f"pushed {mode} adapter {out} -> {args.repo_id}")
    return 0


def _build_native_subtree(pkg: AdapterPackage, out: Path) -> None:
    """Mode 2: merged per-tensor .bin(s) + weight_handoff_map.json + training_config.json + header."""
    cache = Path(pkg.cache_repo_dir)
    paths = PackagePaths.for_cache(cache.parent, cache.name)
    inference = paths.inference
    for t in pkg.tensors:
        src = inference / t.external_data_location
        if src.is_file():
            shutil.copy2(src, out / t.external_data_location)
            sha = src.with_suffix(src.suffix + ".sha256")
            if sha.is_file():
                shutil.copy2(sha, out / sha.name)
    for name in ("weight_handoff_map.json",):
        src = paths.train / name
        if src.is_file():
            shutil.copy2(src, out / name)
    tc = paths.train / "training_config.json"
    if tc.is_file():
        shutil.copy2(tc, out / "training_config.json")
    (out / "mobiletransformers_adapter.json").write_text(
        json.dumps(
            {
                "mode": "native",
                "baseModelId": pkg.base_model_id,
                "peftMethod": pkg.peft_method,
                "marsOptimizationLevel": pkg.mars_optimization_level,
                "handoffMode": pkg.handoff_mode,
                "weightHandoff": "weight_handoff_map.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
