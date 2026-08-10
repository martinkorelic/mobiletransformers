"""`mobiletransformers federated simulate` (#35) — Option-A Flower adapter-aggregation simulation.

Thin CLI over `federated.flower_sim.run_simulation`. The heavy deps (flwr + ORT-training) are imported
lazily inside the runner, so `--help` and the dispatcher work in the core env.
"""

from __future__ import annotations

import argparse

from mobiletransformers.artifacts.package_paths import PackagePaths
from mobiletransformers.exceptions import MobileTransformersError


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    fed = subparsers.add_parser("federated", help="Federated adapter experiments (Flower simulation).")
    fed_sub = fed.add_subparsers(dest="federated_command", metavar="{simulate,serve}")

    sim = fed_sub.add_parser("simulate", help="Run an N-client FedAvg adapter-aggregation simulation.")
    sim.add_argument("--package", required=True, help="Path to a MobileTransformers package dir.")
    sim.add_argument("--strategy", default="fedavg", help="Aggregation strategy (v1: fedavg).")
    sim.add_argument("--clients", type=int, default=4, help="Number of simulated clients.")
    sim.add_argument("--rounds", type=int, default=3, help="Number of federated rounds.")
    sim.add_argument("--local-max-steps", type=int, default=2, help="Local ORT steps per client per round.")
    sim.add_argument("--output", required=True, help="Directory for per-round global adapter artifacts.")
    sim.set_defaults(func=run_simulate)

    serve = fed_sub.add_parser(
        "serve", help="Aggregate one round of client adapter records into a global record."
    )
    serve.add_argument("--package", required=True, help="Path to a MobileTransformers package dir.")
    serve.add_argument("--strategy", default="fedavg", help="Aggregation strategy (v1: fedavg).")
    serve.add_argument(
        "--updates",
        required=True,
        nargs="+",
        help="Client record files (`<client_id>:<path>:<num_examples>`, or just a path).",
    )
    serve.add_argument("--min-clients", type=int, default=2, help="Fewest accepted clients per round.")
    serve.add_argument("--round", type=int, default=0, help="Round number to stamp on the global record.")
    serve.add_argument("--output", required=True, help="Path for the aggregated global record.")
    serve.set_defaults(func=run_serve)

    fed.set_defaults(func=_no_subcommand)
    return fed


def _no_subcommand(args: argparse.Namespace) -> int:
    print("usage: mobiletransformers federated {simulate,serve} ...")
    return 2


def _parse_update(spec: str) -> tuple[str, str, int]:
    """`<client_id>:<path>:<num_examples>`, or a bare path (client id = the filename, weight 1).

    The weight matters: FedAvg is example-weighted, so passing bare paths silently makes every client
    count equally. Accepted for convenience, but it is a different aggregation and worth knowing.
    """
    parts = spec.rsplit(":", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], int(parts[2])
    from pathlib import Path as _P

    return _P(spec).stem, spec, 1


def run_serve(args: argparse.Namespace) -> int:
    from pathlib import Path

    from mobiletransformers.artifacts.handoff_map import HandoffMap
    from mobiletransformers.artifacts.manifest import MobileTransformersManifest
    from mobiletransformers.federated.gateway import FederatedGateway

    if args.strategy != "fedavg":
        print(f"unsupported strategy {args.strategy!r} (v1 supports: fedavg)")
        return 2

    try:
        pkg = Path(args.package)
        manifest = MobileTransformersManifest.load(pkg / "mobiletransformers_manifest.json")
        handoff = HandoffMap.load(pkg / manifest.data["weightHandoff"])
        gateway = FederatedGateway(
            handoff,
            base_model_id=manifest.data.get("baseModelId", "unknown"),
            peft_method=(manifest.data.get("peftMethods") or ["lora"])[0],
            min_clients=args.min_clients,
        )

        submissions = []
        for spec in args.updates:
            client_id, path, num_examples = _parse_update(spec)
            submissions.append((client_id, Path(path).read_bytes(), num_examples))

        result = gateway.aggregate(submissions, round_number=args.round)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(result.blob)

        print(result.describe())
        for client_id, reason in result.rejected:
            print(f"  rejected {client_id}: {reason}")
        print(f"global record -> {out}")
    except MobileTransformersError as exc:
        print(f"error: {exc}")
        return 1
    return 0


def run_simulate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from mobiletransformers.artifacts.handoff_map import HandoffMap
    from mobiletransformers.artifacts.manifest import MobileTransformersManifest
    from mobiletransformers.federated.flower_sim import run_simulation

    try:
        pkg = Path(args.package)
        manifest = MobileTransformersManifest.load(pkg / "mobiletransformers_manifest.json")
        handoff = HandoffMap.load(pkg / manifest.data["weightHandoff"])
        peft_methods = manifest.data.get("peftMethods") or ["lora"]

        # The manifest is the single source of truth for where a stage lives. A hub package puts the
        # train stage at `variants/<variantId>/train`; only the on-device CACHE layout has it flat at
        # `<root>/train`. Resolving it here (rather than appending "train" in the client) is what
        # makes `--package` accept a real exported package.
        variant = manifest.select_variant(
            abis=("arm64-v8a",),
            requested_features=("train",),
        )
        paths = PackagePaths.for_hub(pkg, variant)
        train_dir = paths.train
        tokenizer_dir = paths.tokenizer
        if not (train_dir / "checkpoint").exists():
            raise MobileTransformersError(
                f"no training checkpoint at {train_dir} — export the package with the training stage "
                "(`mobiletransformers export --stages training`) before simulating"
            )
        run_simulation(
            handoff,
            base_model_id=manifest.data.get("baseModelId", "unknown"),
            peft_method=peft_methods[0],
            clients=args.clients,
            rounds=args.rounds,
            local_max_steps=args.local_max_steps,
            output_dir=args.output,
            train_dir=train_dir,
            tokenizer_dir=tokenizer_dir,
            strategy=args.strategy,
        )
    except MobileTransformersError as exc:
        print(f"federated simulate failed: {exc}")
        return 1
    print(f"federated simulation complete -> {args.output}")
    return 0
