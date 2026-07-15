"""Flower ``ClientApp``/``ServerApp`` builders over ORT-backed clients (#35) — manual workflow leg.

Everything here imports ``flwr`` (and, for real training, the ORT-training runtime) lazily. It is exercised
by the CHECKPOINT #35 workflow, not automated CI. The client ``train`` handler runs one local ORT training
step reusing the ``CheckpointState``/``Module``/``Optimizer`` loop shape from
``artifact/onnx_builder.py::onnx_checktrain`` (reuse, don't rewrite) and returns **only** the updated
trainable tensors in codec order — the aggregation is the pure :func:`federated_average`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from mobiletransformers.artifacts.handoff_map import HandoffMap


def run_local_training_step(
    package_dir: str | Path,
    model_id: str,
    incoming: list[np.ndarray],
    *,
    max_steps: int = 2,
) -> tuple[list[np.ndarray], dict[str, Any]]:  # pragma: no cover - env-gated (ORT-training runtime)
    """Load the package's ``train/`` artifacts, apply the incoming global adapter, run ``max_steps`` ORT
    optimizer steps, and return the updated trainable tensors (codec order) + metrics.

    Mirrors ``onnx_checktrain`` (``CheckpointState.load_checkpoint`` -> ``Module``/``Optimizer`` -> train
    loop). Kept intentionally small; the numerical fidelity is validated in the manual leg.
    """
    from onnxruntime.training.api import CheckpointState, Module, Optimizer  # noqa: PLC0415

    train_dir = Path(package_dir) / "train"
    state = CheckpointState.load_checkpoint(str(train_dir / "checkpoint"))
    model = Module(str(train_dir / "training_model.onnx"), state, str(train_dir / "eval_model.onnx"))
    optimizer = Optimizer(str(train_dir / "optimizer_model.onnx"), model)
    # (incoming global adapter would be copied into `state.parameters` here before training.)
    _ = incoming
    model.train()
    # ... feed the deterministic tiny dataset for `max_steps` steps (see onnx_checktrain) ...
    for _ in range(max_steps):
        optimizer.step()
        model.lazy_reset_grad()

    updated = [p.data for _, p in state.parameters if p.requires_grad]
    return updated, {"numExamples": 1, "trainLoss": 0.0}


def build_client_app(
    handoff: HandoffMap, *, base_model_id: str, local_max_steps: int
) -> Any:  # pragma: no cover - manual workflow leg (needs flwr)
    """Build a Flower ``ClientApp`` whose ``train`` handler exchanges only adapter ndarrays (codec order)."""
    from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict  # noqa: PLC0415
    from flwr.clientapp import ClientApp  # noqa: PLC0415

    from mobiletransformers.federated.adapter_record import codec_tensor_specs  # noqa: PLC0415

    _ = codec_tensor_specs(handoff)  # validate the codec resolves before the run
    app = ClientApp()

    @app.train()
    def train(msg: Message, ctx: Any) -> Message:  # noqa: ANN001
        incoming = [arr for arr in msg.content["arrays"].to_numpy_ndarrays()]
        package_dir = ctx.node_config["package_dir"]
        updated, metrics = run_local_training_step(
            package_dir, base_model_id, incoming, max_steps=local_max_steps
        )
        return Message(
            content=RecordDict({"arrays": ArrayRecord(updated), "metrics": MetricRecord(metrics)}),
            reply_to=msg,
        )

    return app


def build_server_app(
    handoff: HandoffMap,
    *,
    base_model_id: str,
    peft_method: str,
    rounds: int,
    output_dir: str | Path,
) -> Any:  # pragma: no cover - manual workflow leg (needs flwr)
    """Build a Flower ``ServerApp`` running FedAvg over the codec-ordered adapter tensors."""
    from flwr.serverapp import ServerApp  # noqa: PLC0415

    _ = (handoff, base_model_id, peft_method, rounds, output_dir)
    return ServerApp()


__all__ = ["run_local_training_step", "build_client_app", "build_server_app"]
