"""FedAvg aggregation + the Option-A in-process simulation driver (#35).

The aggregation math (:func:`federated_average`) and artifact save (:func:`save_global_adapter`) are pure
numpy + stdlib — testable with canned client updates, no Flower and no ORT-training runtime. The
:func:`run_simulation` driver imports ``flwr`` lazily and is the manual workflow leg.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mobiletransformers.exceptions import HandoffError

if TYPE_CHECKING:
    import numpy as np

    from mobiletransformers.artifacts.handoff_map import HandoffMap
    from mobiletransformers.federated.adapter_record import FederatedAdapterRecord


@dataclass
class ClientUpdate:
    """One client's contribution to a round: updated tensors (codec order) weighted by ``num_examples``."""

    arrays: list[np.ndarray]
    num_examples: int


def federated_average(updates: Sequence[ClientUpdate | None]) -> list[np.ndarray]:
    """Weighted (by ``num_examples``) mean of per-client tensor lists, in codec order.

    Dropped clients (``None`` entries) are skipped — aggregation still completes over the survivors.
    Fails closed if no client survived, if survivors disagree on tensor count, or if total weight is zero.
    """
    import numpy as np

    survivors = [u for u in updates if u is not None]
    if not survivors:
        raise HandoffError("no surviving client updates to aggregate")

    n_tensors = len(survivors[0].arrays)
    for u in survivors:
        if len(u.arrays) != n_tensors:
            raise HandoffError(f"client tensor-count mismatch: expected {n_tensors}, got {len(u.arrays)}")
    total = sum(u.num_examples for u in survivors)
    if total <= 0:
        raise HandoffError("total num_examples across surviving clients is zero")

    aggregated: list[np.ndarray] = []
    for i in range(n_tensors):
        acc = None
        for u in survivors:
            contrib = np.asarray(u.arrays[i], dtype=np.float64) * (u.num_examples / total)
            acc = contrib if acc is None else acc + contrib
        # cast back to the survivors' dtype for a stable global artifact
        aggregated.append(np.asarray(acc, dtype=survivors[0].arrays[i].dtype))
    return aggregated


def save_global_adapter(record: FederatedAdapterRecord, output_dir: str | Path) -> Path:
    """Write a round's global adapter record to ``<output_dir>/global_adapter_round<N>.mtfed``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"global_adapter_round{record.round}.mtfed"
    path.write_bytes(record.serialize())
    return path


def run_simulation(
    handoff: HandoffMap,
    *,
    base_model_id: str,
    peft_method: str,
    clients: int,
    rounds: int,
    local_max_steps: int,
    output_dir: str | Path,
    strategy: str = "fedavg",
    **backend_config: Any,
) -> Path:  # pragma: no cover - manual workflow leg (needs flwr + ORT-training)
    """Option-A in-process Flower simulation. Manual/user-run leg — imports ``flwr`` lazily.

    Automated coverage lives in :func:`federated_average` (aggregation) and the record round-trip tests;
    this driver wires them into Flower's ``run_simulation`` over ORT-backed clients.
    """
    if strategy != "fedavg":
        raise HandoffError(f"unsupported strategy {strategy!r} (v1 supports 'fedavg')")
    try:
        from flwr.simulation import run_simulation as flwr_run_simulation  # noqa: F401,PLC0415
    except ImportError as exc:
        raise HandoffError(
            "running the Flower simulation requires flwr (install out-of-band: "
            'pip install "flwr[simulation]"), plus the ORT-training runtime for real client fit'
        ) from exc
    from mobiletransformers.federated.flower_client import build_client_app, build_server_app  # noqa: PLC0415

    client_app = build_client_app(handoff, base_model_id=base_model_id, local_max_steps=local_max_steps)
    server_app = build_server_app(
        handoff, base_model_id=base_model_id, peft_method=peft_method, rounds=rounds, output_dir=output_dir
    )
    flwr_run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=clients,
        backend_config=backend_config or {"client_resources": {"num_cpus": 1}},
    )
    return Path(output_dir)


__all__ = ["ClientUpdate", "federated_average", "save_global_adapter", "run_simulation"]
