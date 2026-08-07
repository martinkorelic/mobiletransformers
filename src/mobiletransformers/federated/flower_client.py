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

from mobiletransformers.exceptions import HandoffError

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
    import numpy as np  # noqa: PLC0415
    from onnxruntime.training.api import CheckpointState, Module, Optimizer  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from mobiletransformers.config.settings import get_settings  # noqa: PLC0415

    train_dir = Path(package_dir) / "train"
    state = CheckpointState.load_checkpoint(str(train_dir / "checkpoint"))
    model = Module(str(train_dir / "training_model.onnx"), state, str(train_dir / "eval_model.onnx"))
    optimizer = Optimizer(str(train_dir / "optimizer_model.onnx"), model)

    # Apply the incoming GLOBAL adapter before training. Skipping this made every round start from the
    # client's own checkpoint, so aggregation had no effect whatsoever on the next round's clients.
    trainable = [(name, param) for name, param in state.parameters if param.requires_grad]
    if incoming:
        if len(incoming) != len(trainable):
            raise HandoffError(
                f"incoming global adapter has {len(incoming)} tensors, "
                f"but the checkpoint has {len(trainable)} trainable parameters"
            )
        for (name, param), array in zip(trainable, incoming, strict=True):
            if tuple(param.data.shape) != tuple(array.shape):
                raise HandoffError(
                    f"incoming tensor for {name!r} has shape {tuple(array.shape)}, "
                    f"expected {tuple(param.data.shape)}"
                )
            param.data = np.ascontiguousarray(array, dtype=param.data.dtype)

    # Deterministic tiny batch, mirroring artifact/onnx_builder.py::onnx_checktrain's input shaping.
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=get_settings().require_hf_token())
    tokenizer.pad_token_id = 0
    batch = tokenizer(
        ["This is a test, hello from world.", "This is a test, hello to world."],
        return_tensors="np",
        padding=True,
    )
    input_ids = np.asarray(batch["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(batch["attention_mask"], dtype=np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    labels = np.copy(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # ignore the final position in the loss

    model.train()
    losses: list[float] = []
    for _ in range(max_steps):
        # The FORWARD pass was missing entirely: optimizer.step() ran against zero/stale gradients, so
        # the "updated" tensors were not a function of any data and trainLoss was hardcoded 0.0.
        forward = model(input_ids, attention_mask, position_ids, labels)
        loss = float(np.asarray(forward[0] if isinstance(forward, (tuple, list)) else forward).mean())
        losses.append(loss)
        optimizer.step()
        model.lazy_reset_grad()

    updated = [param.data for _, param in state.parameters if param.requires_grad]
    return updated, {
        "numExamples": int(input_ids.shape[0]),
        "trainLoss": losses[-1] if losses else 0.0,
    }


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
    """Build a Flower ``ServerApp`` running FedAvg over the codec-ordered adapter tensors.

    This used to discard all five arguments and return a bare ``ServerApp()`` with no strategy and no
    registered handler, so the docstring's claim was false, ``federated_average`` was wired to nothing,
    and the CLI's ``--output`` was never written. The per-round aggregation + save is
    :func:`~mobiletransformers.federated.flower_sim.aggregate_round`, which is pure and unit-tested;
    only the Grid messaging below needs Flower.
    """
    from flwr.app import ArrayRecord, Message, MessageType, RecordDict  # noqa: PLC0415
    from flwr.serverapp import Grid, ServerApp  # noqa: PLC0415

    from mobiletransformers.federated.adapter_record import codec_tensor_specs  # noqa: PLC0415
    from mobiletransformers.federated.flower_sim import ClientUpdate, aggregate_round  # noqa: PLC0415

    specs = codec_tensor_specs(handoff)  # validate the codec resolves before the run
    app = ServerApp()

    @app.main()
    def main(grid: Grid, ctx: Any) -> None:  # noqa: ANN001
        global_arrays: list[np.ndarray] | None = None

        for round_index in range(1, rounds + 1):
            node_ids = list(grid.get_node_ids())
            if not node_ids:
                raise HandoffError(f"round {round_index}: no client nodes available")

            content = RecordDict({"arrays": ArrayRecord(global_arrays)} if global_arrays is not None else {})
            replies = grid.send_and_receive(
                [Message(content=content, message_type=MessageType.TRAIN, dst_node_id=n) for n in node_ids]
            )

            updates: list[ClientUpdate | None] = []
            for reply in replies:
                # A dropped/failed client contributes None; federated_average skips it and the round
                # still completes over the survivors.
                if reply.has_error():
                    updates.append(None)
                    continue
                metrics = reply.content["metrics"]
                updates.append(
                    ClientUpdate(
                        arrays=list(reply.content["arrays"].to_numpy_ndarrays()),
                        num_examples=int(metrics["numExamples"]),
                    )
                )

            global_arrays, saved = aggregate_round(
                handoff,
                updates,
                base_model_id=base_model_id,
                peft_method=peft_method,
                round_index=round_index,
                output_dir=output_dir,
                specs=specs,
            )
            log_path = saved

        _ = log_path  # the last round's artifact path

    return app


__all__ = ["run_local_training_step", "build_client_app", "build_server_app"]
