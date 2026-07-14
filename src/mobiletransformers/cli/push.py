"""`mobiletransformers push` — validate a package (#13 gate), render a model card, upload to the Hub (#15).

Fails closed before any upload: the package must pass the #13 manifest validator. ``--dry-run`` renders
the card + writes ``README.md`` without uploading. The Hub upload lazy-imports ``huggingface_hub``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.exceptions import MobileTransformersError
from mobiletransformers.export.model_card import render_model_card
from mobiletransformers.hub.package_format import MANIFEST_FILENAME


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("push", help="Validate + publish a package to the Hugging Face Hub.")
    parser.add_argument("--package", required=True, help="Package directory to publish.")
    parser.add_argument("--repo", required=True, help="Target HF repo id.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--dry-run", action="store_true", help="Validate + render card; do not upload.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace, *, uploader: Callable[..., Any] | None = None) -> int:
    """``uploader`` is injectable for tests (defaults to huggingface_hub.upload_folder)."""
    package_dir = Path(args.package)
    try:
        manifest = MobileTransformersManifest.load(package_dir / MANIFEST_FILENAME)
        manifest.validate(package_dir)  # #13 gate — fail closed before upload
    except MobileTransformersError as exc:
        print(f"push aborted — package failed validation: {exc}")
        return 1

    card = render_model_card(manifest.to_dict(), str(package_dir))
    (package_dir / "README.md").write_text(card, encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] validated {package_dir}; wrote README.md ({len(card)} chars); not uploading.")
        return 0

    if uploader is None:
        from huggingface_hub import create_repo, upload_folder

        create_repo(args.repo, exist_ok=True, private=args.private)
        uploader = upload_folder
    uploader(repo_id=args.repo, folder_path=str(package_dir))
    print(f"pushed {package_dir} -> {args.repo}")
    return 0
