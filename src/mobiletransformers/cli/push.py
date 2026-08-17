"""`mobiletransformers push` — validate a package (#13 gate), render a model card, upload to the Hub (#15).

Fails closed before any upload: the package must pass the #13 manifest validator. ``--dry-run`` renders
the card + writes ``README.md`` without uploading. The Hub upload lazy-imports ``huggingface_hub``.

The target repo must already exist unless ``--create`` is passed. Creating on demand was the previous
default, which meant a mistyped repo id produced a new repo rather than an error — the wrong trade for
a command that normally runs against an organisation account.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mobiletransformers.artifacts.manifest import MobileTransformersManifest
from mobiletransformers.config.settings import get_settings
from mobiletransformers.exceptions import MobileTransformersError
from mobiletransformers.export.model_card import BANNER_FILENAME, render_model_card
from mobiletransformers.hub.package_format import MANIFEST_FILENAME


def _repo_root() -> Path:
    """The checkout this package was installed from, for build-time-only assets like the banner.

    `src/mobiletransformers/cli/push.py` -> up four. Returns a path that simply will not exist for a
    wheel installed outside a checkout, which the caller already handles by skipping the banner —
    an installed wheel has no `docs/` and should publish a card without a header image rather than
    fail the push.
    """
    return Path(__file__).resolve().parents[3]


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("push", help="Validate + publish a package to the Hugging Face Hub.")
    parser.add_argument("--package", required=True, help="Package directory to publish.")
    parser.add_argument("--repo", required=True, help="Target HF repo id.")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the repo if it does not exist. OFF by default: pushing to an existing repo is the "
        "normal case, and creating on demand means a mistyped id silently makes a new repo instead of "
        "failing — under an org account that is a stray public repo nobody asked for.",
    )
    parser.add_argument("--private", action="store_true", help="With --create, create the repo as private.")
    parser.add_argument(
        "--token",
        default=None,
        help="Hub token. Defaults to $HF_TOKEN (huggingface_hub's own fallback). Pass this when the "
        "target is an ORGANISATION repo and the org token lives in a differently-named variable — "
        "otherwise the upload silently authenticates as the wrong identity or not at all.",
    )
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

    # Stage the header image INTO the package so the card can reference it relatively. Hot-linking
    # the framework repository would tie a published page to that repo's visibility, default branch
    # and directory layout — and it renders as a broken image for as long as any of those disagree.
    # Copied here rather than at export time because it is a publishing concern: a package pulled to
    # a device has no use for it.
    banner_source = _repo_root() / "docs" / "assets" / BANNER_FILENAME
    banner: str | None = None
    if banner_source.is_file():
        shutil.copyfile(banner_source, package_dir / BANNER_FILENAME)
        banner = BANNER_FILENAME
    else:
        # Referencing a name we did not upload is worse than having no banner at all.
        print(f"note: {banner_source} not found — publishing the card without its header image")

    card = render_model_card(manifest.to_dict(), str(package_dir), banner=banner, repo_id=args.repo)
    (package_dir / "README.md").write_text(card, encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] validated {package_dir}; wrote README.md ({len(card)} chars); not uploading.")
        return 0

    # Explicit beats ambient: `huggingface_hub` falls back to $HF_TOKEN and then to the cached CLI
    # login, so an org push with no token argument can succeed as the WRONG identity — into a personal
    # namespace, or with the cached user's permissions — and look exactly like success.
    #
    # The fallback goes through `config.settings` rather than `os.environ` directly: that is the one
    # sanctioned credential-read site (`test_guards.py::test_no_direct_secret_environment_reads`), and
    # it also picks up `.env`, which a bare environ read would not.
    token = getattr(args, "token", None) or get_settings().hf_token

    if uploader is None:
        from huggingface_hub import create_repo, upload_folder

        if getattr(args, "create", False):
            create_repo(args.repo, exist_ok=True, private=args.private, token=token)
        uploader = upload_folder
    uploader(repo_id=args.repo, folder_path=str(package_dir), token=token)
    print(f"pushed {package_dir} -> {args.repo}")
    return 0
