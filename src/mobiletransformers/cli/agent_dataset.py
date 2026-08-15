"""`mobiletransformers agent-dataset` — build the #37 tool-call training set.

Two sources, one output shape:

* ``--source google/mobile-actions`` (or any Hub id / local JSONL in that format) imports a real
  function-calling corpus;
* ``--source generated --allowlist actions.json`` synthesises a per-user set from an app's own
  allowlist (`agent/mobile_actions.py`).

Both write ``<output>/<name>.jsonl`` — the `{"prompt", "completion"}` rows `ORTDataCurator` reads on
device under the `mobile_actions` task — plus ``<output>/action_schema.json``, the allowlist
`FunctionCallValidator` is constructed from. Emitting both together is the point: the training targets
and the validator's boundary come out of one command, so they cannot drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobiletransformers.exceptions import MobileTransformersError
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "agent-dataset",
        help="Build the #37 tool-call training set from a corpus or an app allowlist.",
    )
    parser.add_argument(
        "--source",
        default="google/mobile-actions",
        help="Hub dataset id, a local .jsonl, or 'generated' to synthesise from --allowlist.",
    )
    parser.add_argument("--output", default="build/agent", help="Directory to write into.")
    parser.add_argument("--name", default="mobile_actions", help="Basename of the emitted .jsonl.")
    parser.add_argument(
        "--allowlist",
        default=None,
        help="Action-schema JSON. Required for --source generated; ignored otherwise (the corpus "
        "declares its own tools).",
    )
    parser.add_argument("--split", default="train", help="Corpus split to keep ('train'/'eval'/'all').")
    parser.add_argument("--limit", type=int, default=None, help="Keep at most N rows (after shuffling).")
    parser.add_argument("--per-action", type=int, default=8, help="Rows per action for 'generated'.")
    parser.add_argument(
        "--templates",
        default=None,
        help="JSON {actionName: [prompt template, ...]} overriding the built-in phrasings for "
        "'generated'. Slots name the action's OWN parameters. Needed whenever your allowlist declares "
        "different parameters than the built-in demo actions — the generator refuses a template that "
        "references a parameter the action does not declare, so without this an app whose 'set_alarm' "
        "takes only 'time' cannot use the built-in 'set_alarm' phrasings at all.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Determinism for 'generated' and --limit.")
    parser.add_argument(
        "--prompt-style",
        default="context",
        choices=("context", "user"),
        help="'context' prepends the corpus's date/day preamble (relative dates need it); "
        "'user' is the bare instruction.",
    )
    parser.add_argument(
        "--multi-call",
        default="skip",
        choices=("skip", "first"),
        help="Records with several tool calls: drop them (default), or keep the first.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    import random

    from mobiletransformers.agent.mobile_actions_import import write_action_schema

    out = Path(args.output)
    dataset_path = out / f"{args.name}.jsonl"
    schema_path = out / "action_schema.json"

    try:
        if args.source == "generated":
            from mobiletransformers.agent.mobile_actions import (
                generate_examples,
                load_allowlist,
                write_jsonl,
            )

            if not args.allowlist:
                raise MobileTransformersError("--source generated requires --allowlist <action_schema.json>")
            specs = load_allowlist(args.allowlist)
            templates = None
            if args.templates:
                try:
                    templates = {
                        action: tuple(phrasings)
                        for action, phrasings in json.loads(
                            Path(args.templates).read_text(encoding="utf-8")
                        ).items()
                    }
                except (OSError, ValueError, AttributeError) as exc:
                    raise MobileTransformersError(
                        f"--templates {args.templates} is not a readable "
                        f'{{"actionName": ["phrasing", ...]}} JSON object: {exc}'
                    ) from exc
            rows = generate_examples(specs, per_action=args.per_action, seed=args.seed, templates=templates)
        else:
            from mobiletransformers.agent.mobile_actions import write_jsonl
            from mobiletransformers.agent.mobile_actions_import import (
                extract_allowlist,
                read_records,
                resolve_source,
                to_training_rows,
            )

            path = resolve_source(args.source)
            # Read once into memory: the allowlist is the union over every record, so a single
            # streaming pass cannot produce both halves, and the corpus is ~25 MB.
            records = list(read_records(path))
            specs = extract_allowlist(records)
            rows = to_training_rows(
                records,
                split=None if args.split == "all" else args.split,
                prompt_style=args.prompt_style,
                multi_call=args.multi_call,
            )

        if not rows:
            raise MobileTransformersError(
                f"no rows produced from {args.source!r} (split={args.split!r}) — nothing to train on"
            )
        if args.limit is not None and args.limit < len(rows):
            random.Random(args.seed).shuffle(rows)
            rows = rows[: args.limit]
    except MobileTransformersError as exc:
        print(f"agent-dataset: {exc}")
        return 1

    actions = ", ".join(sorted(s.action_name for s in specs))
    if args.dry_run:
        print(f"[dry-run] {len(rows)} rows, {len(specs)} actions ({actions})")
        print(f"[dry-run] would write {dataset_path} and {schema_path}")
        return 0

    write_jsonl(rows, dataset_path)
    write_action_schema(specs, schema_path)
    unbindable = sorted(s.action_name for s in specs if not s.allowed_intent)
    print(f"agent-dataset: wrote {len(rows)} rows -> {dataset_path}")
    print(f"agent-dataset: wrote {len(specs)} actions -> {schema_path} ({actions})")
    if unbindable:
        # Trainable and validatable, but IntentBinder can never fire them. Said out loud because a
        # silent empty intent would look like a binder bug much later.
        print(f"agent-dataset: no Android intent mapped for {unbindable} — validated but never bound")
    return 0
