#!/usr/bin/env python3
"""Derive a synthetic PEER record from a real device record (#36 device round-trip).

## Why this exists

`FederatedGateway` refuses to publish an aggregate below `min_clients` (2 by default), and one phone is
available. Submitting the device's own record twice would satisfy the count while producing an
"aggregate" numerically identical to what the device already holds — an import that writes back exactly
what was there cannot distinguish a working round from a no-op, which is the whole thing the device
round-trip is meant to prove.

So the second client is explicitly synthetic and explicitly *different*: every factor scaled by
``--scale``. FedAvg then produces a value the device did **not** produce, and the device-side assertion
"the checkpoint now holds the aggregate's bytes" has something to fail against.

What this does NOT claim: it is not a second training run, and the round-trip therefore proves the
transport/aggregation/import seam, not multi-device convergence (that is #35's simulation, which trains
real clients).

Usage::

    python scripts/federated_peer_record.py --package build/pkg \
        --input build/federated_round/client_update.bin \
        --output build/federated_round/peer_update.bin --scale 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, help="Path to the exported package dir.")
    parser.add_argument("--input", required=True, help="A real client record (from the device).")
    parser.add_argument("--output", required=True, help="Where to write the synthetic peer record.")
    parser.add_argument("--scale", type=float, default=3.0, help="Factor applied to every tensor.")
    args = parser.parse_args(argv)

    from mobiletransformers.artifacts.handoff_map import HandoffMap
    from mobiletransformers.artifacts.manifest import MobileTransformersManifest
    from mobiletransformers.federated.adapter_record import FederatedAdapterRecord

    pkg = Path(args.package)
    manifest = MobileTransformersManifest.load(pkg / "mobiletransformers_manifest.json")
    handoff = HandoffMap.load(pkg / manifest.data["weightHandoff"])

    record = FederatedAdapterRecord.deserialize(Path(args.input).read_bytes())
    # check_format is what would catch a device record built against a different package; running it
    # here means a mismatch is reported next to the two files rather than as a gateway rejection.
    record.check_format(handoff)

    peer = FederatedAdapterRecord.from_handoff(
        handoff,
        [array * args.scale for array in record.arrays],
        base_model_id=record.base_model_id,
        peft_method=record.peft_method,
        round=record.round,
        package_revision=record.mobiletransformers_package_revision,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(peer.serialize())

    print(
        f"peer record: {len(peer.tensors)} tensors x{args.scale} -> {out} "
        f"({out.stat().st_size} B)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
