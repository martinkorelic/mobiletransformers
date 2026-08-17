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


def aggregate_round(
    handoff: HandoffMap,
    updates: Sequence[ClientUpdate | None],
    *,
    base_model_id: str,
    peft_method: str,
    round_index: int,
    output_dir: str | Path,
    specs: Sequence[Any] | None = None,
) -> tuple[list[np.ndarray], Path]:
    """FedAvg one round's client updates, wrap them in a record, and persist it.

    The whole server side of a round, minus the Flower messaging: aggregate -> build the
    :class:`FederatedAdapterRecord` in codec order -> :func:`save_global_adapter`. Pure (no ``flwr``,
    no ORT), so the round semantics are unit-tested rather than only exercised in the manual sim.

    Returns ``(aggregated arrays, saved artifact path)``. The arrays are fed back to the clients as the
    next round's global adapter.
    """
    from mobiletransformers.federated.adapter_record import (  # noqa: PLC0415
        FederatedAdapterRecord,
        codec_tensor_specs,
    )

    aggregated = federated_average(updates)
    tensor_specs = list(specs) if specs is not None else codec_tensor_specs(handoff)
    if len(tensor_specs) != len(aggregated):
        # Naming the counts is not enough here: the two numbers come from two different tensor
        # VOCABULARIES, and the next person needs to know which. The codec describes the inference
        # initializers (one merged weight per adapted layer); an ORT training checkpoint holds the
        # rank-r factors (lora_A + lora_B per adapted layer), i.e. exactly twice as many, of a
        # different shape. See the `merged_base_plus_adapter` note in docs/FEDERATED.md.
        raise HandoffError(
            f"codec declares {len(tensor_specs)} tensors but the round aggregated {len(aggregated)}. "
            "The codec's vocabulary is MERGED inference initializers (one per adapted layer, full "
            "weight shape); a client returning raw ORT checkpoint trainables sends rank-r LoRA "
            "factors (lora_A + lora_B per layer) instead. v1 declares the merged shape "
            "(aggregation_role='merged_base_plus_adapter'), so a client must merge locally before "
            "exchanging — switching the record to rank-r adapters is the open v2 decision and would "
            "break the cross-language byte golden."
        )

    survivors = [u for u in updates if u is not None]
    record = FederatedAdapterRecord.from_handoff(
        handoff,
        aggregated,
        base_model_id=base_model_id,
        peft_method=peft_method,
        round=round_index,
        metrics={
            "clients": float(len(survivors)),
            "dropped": float(len(updates) - len(survivors)),
            "numExamples": float(sum(u.num_examples for u in survivors)),
        },
    )
    return aggregated, save_global_adapter(record, output_dir)


def run_simulation(
    handoff: HandoffMap,
    *,
    base_model_id: str,
    peft_method: str,
    clients: int,
    rounds: int,
    local_max_steps: int,
    output_dir: str | Path,
    train_dir: str | Path,
    tokenizer_dir: str | Path,
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
        handoff,
        base_model_id=base_model_id,
        peft_method=peft_method,
        rounds=rounds,
        output_dir=output_dir,
        # The clients need this and `run_simulation` has no node_config to carry it, so the server
        # puts it in each round's message. ABSOLUTE: each client runs inside its own Ray actor with
        # its own working directory, so a relative path resolves somewhere else entirely — which
        # surfaces as ORT's opaque `Invalid fd was supplied: -1`, naming nothing.
        train_dir=Path(train_dir).resolve(),
        tokenizer_dir=Path(tokenizer_dir).resolve(),
    )
    flwr_run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=clients,
        backend_config=backend_config or {"client_resources": {"num_cpus": 1}},
    )
    # The ServerApp saves one artifact per round via aggregate_round; fail closed if none appeared,
    # rather than reporting success for an empty --output (which is what used to happen).
    out = Path(output_dir)
    produced = sorted(out.glob("global_adapter_round*.mtfed"))
    if not produced:
        raise HandoffError(
            f"simulation finished but wrote no global adapter to {out} — no round completed aggregation"
        )
    return out


__all__ = [
    "ClientUpdate",
    "federated_average",
    "aggregate_round",
    "save_global_adapter",
    "run_simulation",
]
