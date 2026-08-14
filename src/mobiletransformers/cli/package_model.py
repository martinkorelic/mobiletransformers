"""`mobiletransformers package-model` — re-emit a package's manifest + checksums from its tree.

Previously a stub that printed "not yet wired" and **returned 0**, i.e. it reported success for
anything — including a package that did not exist. `make package-model` and `docs/PUBLIC_API.md`
advertised it regardless.

Scope: this does NOT run an export (that is `mobiletransformers export`, #15). It re-derives the
integrity half of `mobiletransformers_manifest.json` — `fileSizes`, `sha256`, `requiredFiles`,
per-variant `paths` and the `downloadPlan` — by stream-hashing the on-disk tree, reusing the existing
manifest for the descriptive half (base model, variants, provenance). That is what you need after a
package tree changes underneath its manifest: a merged checkpoint copied back off a device, a
hand-swapped tokenizer, a stage directory added.

Fails closed: no directory, no manifest, or an unparseable manifest is a non-zero exit with a typed
error, never a silent 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mobiletransformers.exceptions import MobileTransformersError
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)

#: Manifest keys that `build_manifest` reads out of its ``report`` argument. Re-emitting from an
#: existing manifest means feeding these back in from the manifest's own top level.
_REPORT_KEYS = (
    "mobiletransformersVersion",
    "architectures",
    "supportedTasks",
    "selectedTask",
    "trustRemoteCode",
    "optimumOnnxVersion",
    "transformersVersion",
    "onnxRuntimeTrainingVersion",
    "onnxRuntimeGenAIVersion",
    "peftMethods",
    "quantization",
    "trainableParameterCount",
    "trainingParameterCount",
    "androidRuntime",
    "license",
)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "package-model",
        help="Re-emit a package's manifest + checksums from its on-disk tree.",
    )
    parser.add_argument("--package", default=None, help="Package directory to re-emit.")
    parser.add_argument("--config", help="Path to config YAML (validates it parses).")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
    parser.set_defaults(func=run)
    return parser


def repackage(package_dir: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Re-derive and (unless ``dry_run``) rewrite the manifest for an existing package.

    Returns the freshly built manifest dict. Raises :class:`MobileTransformersError` if the directory
    or its manifest is missing or unusable — never returns a partially-built manifest.
    """
    from mobiletransformers.hub.package_format import (
        MANIFEST_FILENAME,
        build_manifest,
        write_manifest,
        write_variant_checksums,
    )

    package = Path(package_dir)
    if not package.is_dir():
        raise MobileTransformersError(f"not a package directory: {package}")

    manifest_path = package / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise MobileTransformersError(
            f"no {MANIFEST_FILENAME} in {package} — run `mobiletransformers export` to create a package"
        )
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MobileTransformersError(f"{manifest_path} is not valid JSON: {exc}") from exc

    variants = existing.get("variants")
    if not variants:
        raise MobileTransformersError(f"{manifest_path} declares no variants")
    base_model_id = existing.get("baseModelId")
    if not base_model_id:
        raise MobileTransformersError(f"{manifest_path} has no baseModelId")

    report = {key: existing[key] for key in _REPORT_KEYS if key in existing}

    # `write_variant_checksums` adds files to the tree, so hash twice — the same two-pass shape the
    # export pipeline uses, so a re-emit is byte-identical to a fresh export of the same tree.
    def _build() -> dict[str, Any]:
        return build_manifest(
            package,
            variants,
            base_model_id=base_model_id,
            report=report,
            default_variant=existing.get("defaultVariant"),
            exported_at=existing.get("exportedAt"),
        )

    def _without_derived(entry: Any) -> Any:
        """Drop `checksums.json` keys — `sha256`/`fileSizes` are maps, `requiredFiles` is a list."""
        if isinstance(entry, dict):
            return {rel: v for rel, v in entry.items() if not str(rel).endswith("/checksums.json")}
        return [rel for rel in entry if not str(rel).endswith("/checksums.json")]

    if dry_run:
        rebuilt = _build()
        # `checksums.json` is derived from the first pass, so on a re-emit it is present where a fresh
        # export's first pass had nothing. Compare the real content, not that self-reference.
        changed = sorted(
            key
            for key in ("fileSizes", "sha256", "requiredFiles")
            if _without_derived(rebuilt.get(key, {})) != _without_derived(existing.get(key, {}))
        )
        logger.info(
            "package-model --dry-run: %s would %s",
            package,
            f"change {', '.join(changed)}" if changed else "be unchanged",
        )
        return rebuilt

    # Reproduce a fresh export's initial condition: the first `build_manifest` must not see the
    # previous run's `checksums.json`, or each variant's checksum file would grow an entry for itself
    # and a re-emit would never reach a fixpoint.
    for variant in variants:
        stale = package / "variants" / str(variant["id"]) / "checksums.json"
        stale.unlink(missing_ok=True)

    write_variant_checksums(package, _build())
    rebuilt = _build()
    write_manifest(package, rebuilt)
    return rebuilt


def run(args: argparse.Namespace) -> int:
    package = getattr(args, "package", None)
    config = getattr(args, "config", None)
    dry_run = bool(getattr(args, "dry_run", False))

    if not package:
        print("package-model: pass --package <dir> (the package to re-emit)")
        return 2

    try:
        if config:
            from mobiletransformers.cli.validate import validate_config

            validate_config(config)
        manifest = repackage(package, dry_run=dry_run)
    except MobileTransformersError as exc:
        print(f"package-model: {exc}")
        return 1

    verb = "would re-emit" if dry_run else "re-emitted"
    print(f"package-model: {verb} {len(manifest['sha256'])} files for {manifest['baseModelId']}")
    return 0
