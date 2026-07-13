"""`mobiletransformers validate` subcommand — stub.

Real body lands in a later plan. Parses args and reports that it is not yet
wired, returning 0 under ``--dry-run``.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("validate", help="Validate a device-ready package or config.")
    parser.add_argument("--config", help="Path to config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without running.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    print("mobiletransformers validate: not yet wired (stub).")
    if getattr(args, "dry_run", False):
        return 0
    return 0
