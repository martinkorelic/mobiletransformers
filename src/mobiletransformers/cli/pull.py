"""`mobiletransformers pull` + `install-package` (#21) — download a Hub package and materialize the cache.

`pull` fetches manifest-first, selects a variant, downloads only the requested feature groups
(sha256-verified). `install-package` reshapes a staged package into the `LLMRepository` cache layout.
Both are thin CLIs over `hub.pull`.
"""

from __future__ import annotations

import argparse

from mobiletransformers.exceptions import MobileTransformersError

# The dispatcher registers each module's add_parser; expose two here via separate modules is overkill,
# so this single module contributes both subcommands through add_parser.


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    pull = subparsers.add_parser("pull", help="Download a Hub package (manifest-first, sha256-verified).")
    pull.add_argument("--repo-id", required=True, help="HF repo id of the package.")
    pull.add_argument("--revision", default="main", help="Git revision (default main).")
    pull.add_argument("--variant", default=None, help="Variant id (auto-selected if omitted).")
    pull.add_argument(
        "--features", default="inference", help="Comma-separated feature groups (e.g. inference,train,rag)."
    )
    pull.add_argument("--out", default=None, help="Staging output dir (default ./.mt-pull/<repo>).")
    pull.set_defaults(func=run_pull)

    inst = subparsers.add_parser(
        "install-package", help="Materialize a staged package into the cache layout."
    )
    inst.add_argument("--staging", required=True, help="Staged package directory (from `pull`).")
    inst.add_argument("--repo-id", required=True, help="HF repo id (used for the sanitized cache dir name).")
    inst.add_argument("--cache-root", required=True, help="Cache root to install into.")
    inst.add_argument("--variant", default=None, help="Variant id (default = manifest defaultVariant).")
    inst.set_defaults(func=run_install)
    return pull


def run_pull(args: argparse.Namespace) -> int:
    from mobiletransformers.hub.pull import pull_package

    features = tuple(f.strip() for f in args.features.split(",") if f.strip())
    try:
        staging = pull_package(
            args.repo_id, revision=args.revision, variant=args.variant, features=features, dest=args.out
        )
    except MobileTransformersError as exc:
        print(f"pull failed: {exc}")
        return 1
    print(f"pulled {args.repo_id} -> {staging}")
    return 0


def run_install(args: argparse.Namespace) -> int:
    from mobiletransformers.hub.pull import install_package

    try:
        target = install_package(args.staging, args.cache_root, args.repo_id, variant=args.variant)
    except MobileTransformersError as exc:
        print(f"install-package failed: {exc}")
        return 1
    print(f"installed -> {target}")
    return 0
