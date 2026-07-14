"""MobileTransformers CLI entry point.

Canonical argparse dispatcher. Every later CLI plan (one-command export, hub
pull/install, ...) registers its subcommand into *this* dispatcher via the
``add_parser(subparsers)`` / ``run(args) -> int`` shape used by the stub modules
below. Do not add a ``cli/__main__.py`` or switch to typer/click.
"""

from __future__ import annotations

import argparse

from mobiletransformers import __version__
from mobiletransformers.cli import export, package_model, push, support_matrix, validate

_SUBCOMMANDS = (export, validate, package_model, push, support_matrix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mobiletransformers",
        description="Export and Android runtime tooling for on-device transformers.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(
        dest="command", metavar="{export,validate,package-model,push,support-matrix}"
    )
    for module in _SUBCOMMANDS:
        module.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Argparse dispatcher. Subcommands: export, validate, package-model.

    Returns a process exit code. Both ``mobiletransformers --help`` and
    ``python -m mobiletransformers.cli.main --help`` work.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
