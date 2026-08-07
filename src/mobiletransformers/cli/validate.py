"""`mobiletransformers validate` — check a written package against the #13/#14 contract.

Also backs ``mobiletransformers export --validate``. Previously a stub that printed "not yet wired"
and returned 0, i.e. it reported success for anything — including a package that did not exist.

Validation is deliberately read-only and dependency-light (JSON + the filesystem), so it runs in the
core env against a package produced under the export or ORT-training profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mobiletransformers.exceptions import MobileTransformersError


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("validate", help="Validate a device-ready package.")
    parser.add_argument("--package", default=None, help="Package directory to validate.")
    parser.add_argument("--config", default=None, help="Path to config YAML (validates it parses).")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without running.")
    parser.set_defaults(func=run)
    return parser


def validate_package(package_dir: str | Path) -> None:
    """Fail closed unless ``package_dir`` is a valid #14 package.

    Checks, in order: the directory exists; the manifest is present and parses; the manifest's own
    invariants hold (#13 ``MobileTransformersManifest.validate``, which resolves every declared file,
    the selected variant's subtrees and the weight-handoff reference).
    """
    from mobiletransformers.artifacts.manifest import MobileTransformersManifest
    from mobiletransformers.hub.package_format import MANIFEST_FILENAME

    package = Path(package_dir)
    if not package.is_dir():
        raise MobileTransformersError(f"not a package directory: {package}")

    manifest_path = package / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise MobileTransformersError(f"no {MANIFEST_FILENAME} in {package}")

    manifest = MobileTransformersManifest.load(manifest_path)
    manifest.validate(package)


def validate_config(config_path: str | Path) -> None:
    """Fail closed unless ``config_path`` is a readable YAML mapping."""
    from mobiletransformers.utils.yaml import load_config_from_file

    path = Path(config_path)
    if not path.is_file():
        raise MobileTransformersError(f"config not found: {path}")
    document = load_config_from_file(path)
    if not isinstance(document, dict):
        raise MobileTransformersError(f"{path}: expected a YAML mapping at the top level")


def run(args: argparse.Namespace) -> int:
    package = getattr(args, "package", None)
    config = getattr(args, "config", None)
    if not package and not config:
        print("validate: pass --package and/or --config")
        return 2

    try:
        if config:
            validate_config(config)
            print(f"config OK: {config}")
        if package:
            if getattr(args, "dry_run", False):
                print(f"[dry-run] would validate the package at {package}")
            else:
                validate_package(package)
                print(f"package OK: {package}")
    except MobileTransformersError as exc:
        print(f"validation failed: {exc}")
        return 1
    return 0
