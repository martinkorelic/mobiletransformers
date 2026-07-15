"""`mobiletransformers federated simulate` (#35) — Option-A Flower adapter-aggregation simulation.

Thin CLI over `federated.flower_sim.run_simulation`. The heavy deps (flwr + ORT-training) are imported
lazily inside the runner, so `--help` and the dispatcher work in the core env.
"""

from __future__ import annotations

import argparse

from mobiletransformers.exceptions import MobileTransformersError


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    fed = subparsers.add_parser("federated", help="Federated adapter experiments (Flower simulation).")
    fed_sub = fed.add_subparsers(dest="federated_command", metavar="{simulate}")

    sim = fed_sub.add_parser("simulate", help="Run an N-client FedAvg adapter-aggregation simulation.")
    sim.add_argument("--package", required=True, help="Path to a MobileTransformers package dir.")
    sim.add_argument("--strategy", default="fedavg", help="Aggregation strategy (v1: fedavg).")
    sim.add_argument("--clients", type=int, default=4, help="Number of simulated clients.")
    sim.add_argument("--rounds", type=int, default=3, help="Number of federated rounds.")
    sim.add_argument("--local-max-steps", type=int, default=2, help="Local ORT steps per client per round.")
    sim.add_argument("--output", required=True, help="Directory for per-round global adapter artifacts.")
    sim.set_defaults(func=run_simulate)

    fed.set_defaults(func=_no_subcommand)
    return fed


def _no_subcommand(args: argparse.Namespace) -> int:
    print("usage: mobiletransformers federated simulate ...")
    return 2


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
        run_simulation(
            handoff,
            base_model_id=manifest.data.get("baseModelId", "unknown"),
            peft_method=peft_methods[0],
            clients=args.clients,
            rounds=args.rounds,
            local_max_steps=args.local_max_steps,
            output_dir=args.output,
            strategy=args.strategy,
        )
    except MobileTransformersError as exc:
        print(f"federated simulate failed: {exc}")
        return 1
    print(f"federated simulation complete -> {args.output}")
    return 0
