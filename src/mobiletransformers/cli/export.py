"""`mobiletransformers export` subcommand — stub.

Real body lands in the export-CLI plan (02_code_plans/05). For now it parses
args and reports that it is not yet wired, returning 0 under ``--dry-run``.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("export", help="Export an HF model to a device-ready package.")
    parser.add_argument("--config", help="Path to config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without running.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    print("mobiletransformers export: not yet wired (stub — real logic lands in 02_code_plans/05).")
    if getattr(args, "dry_run", False):
        return 0
    return 0
