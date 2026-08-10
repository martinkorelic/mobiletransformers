"""Flower ``ClientApp``/``ServerApp`` builders over ORT-backed clients (#35) — manual workflow leg.

Everything here imports ``flwr`` (and, for real training, the ORT-training runtime) lazily. It is exercised
by the CHECKPOINT #35 workflow, not automated CI. The client ``train`` handler runs one local ORT training
step reusing the ``CheckpointState``/``Module``/``Optimizer`` loop shape from
``artifact/onnx_builder.py::onnx_checktrain`` (reuse, don't rewrite) and returns **only** the updated
trainable tensors in codec order — the aggregation is the pure :func:`federated_average`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mobiletransformers.exceptions import HandoffError

if TYPE_CHECKING:
    import numpy as np

    from mobiletransformers.artifacts.handoff_map import HandoffMap


#: Deterministic per-client corpus. Federated averaging over clients that all see the SAME batch is
#: an average of identical updates — arithmetically a no-op, and it cannot show that aggregation does
#: anything a single client does not. Each shard is a different topic so the clients genuinely
#: disagree, and the HELD-OUT sentences below are what the aggregated adapter is scored on.
CLIENT_SHARDS: tuple[tuple[str, ...], ...] = (
    (
        "The kettle boiled and the tea was poured.",
        "She sliced the bread and buttered it warm.",
        "Dinner simmered slowly on the back burner.",
    ),
    (
        "The train pulled into the station on time.",
        "He bought a ticket and found his seat.",
        "The platform emptied as the doors closed.",
    ),
    (
        "Rain fell steadily against the window.",
        "The clouds broke and the sun came through.",
        "A cold wind moved across the open field.",
    ),
    (
        "The library was quiet all afternoon.",
        "She returned the book she had borrowed.",
        "Shelves of paperbacks lined the far wall.",
    ),
)

#: Fixed evaluation batch, scored identically by every round. Held out from every shard so a falling
#: number means the aggregate generalised, not that it memorised one client's rows.
EVAL_SENTENCES: tuple[str, ...] = (
    "The morning light came in through the window.",
    "He closed the book and set it on the table.",
)


def _load_tokenizer(tokenizer_dir: str | Path) -> Any:  # pragma: no cover - env-gated
    """Load the tokenizer the PACKAGE ships, never from the Hub.

    This used to be `AutoTokenizer.from_pretrained(model_id, token=require_hf_token())`, which fails
    with "HF_TOKEN is not set" inside a Ray actor (the actors do not inherit the driver's settings)
    and, more importantly, makes a *federated client* reach out to a remote service to do local work.
    A federated client that phones home for a tokenizer it already has on disk is the wrong shape.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tokenizer.pad_token_id = 0
    return tokenizer


def _batch(
    tokenizer: Any, sentences: Sequence[str], *, np: Any
) -> tuple[Any, Any, Any, Any]:  # pragma: no cover - env-gated
    """Tokenize to the (input_ids, attention_mask, position_ids, labels) tuple the graph expects."""
    encoded = tokenizer(list(sentences), return_tensors="np", padding=True)
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    labels = np.copy(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # ignore the final position in the loss
    return input_ids, attention_mask, position_ids, labels


def evaluate_adapter(
    train_dir: str | Path,
    tokenizer_dir: str | Path,
    arrays: list[np.ndarray],
    handoff: HandoffMap,
) -> float:  # pragma: no cover - env-gated (ORT-training runtime)
    """Loss of ``arrays`` (a global adapter) on the fixed held-out batch.

    This is the metric the #35 self-check asks about. It is deliberately measured on the SERVER over
    the AGGREGATED tensors: a falling client-side training loss only says each client fitted its own
    shard, which is true even when aggregation is broken.
    """
    import numpy as np  # noqa: PLC0415
    from onnxruntime.training.api import CheckpointState, Module  # noqa: PLC0415

    train_dir = Path(train_dir)
    state = CheckpointState.load_checkpoint(str(train_dir / "checkpoint"))
    model = Module(str(train_dir / "training_model.onnx"), state, str(train_dir / "eval_model.onnx"))

    by_name = {name: param for name, param in state.parameters if param.requires_grad}
    if arrays:
        for name, array in zip(_codec_ordered_names(handoff, by_name), arrays, strict=True):
            by_name[name].data = np.ascontiguousarray(array, dtype=by_name[name].data.dtype)

    tokenizer = _load_tokenizer(tokenizer_dir)
    inputs = _batch(tokenizer, EVAL_SENTENCES, np=np)

    model.eval()
    out = model(*inputs)
    return float(np.asarray(out[0] if isinstance(out, (tuple, list)) else out).mean())


def _codec_ordered_names(handoff: HandoffMap, available: dict[str, Any]) -> list[str]:
    """The codec's adapter-factor names, in serialization order, checked against what exists.

    Fails closed naming the first missing tensor: a checkpoint that cannot supply a declared factor is
    a package/codec mismatch, and continuing would exchange a short, silently misaligned vector.
    """
    from mobiletransformers.federated.adapter_record import codec_tensor_specs  # noqa: PLC0415

    names = [spec.name for spec in codec_tensor_specs(handoff)]
    missing = [n for n in names if n not in available]
    if missing:
        raise HandoffError(
            f"{len(missing)} codec-declared adapter factors are absent from the checkpoint "
            f"(e.g. {missing[0]}); the package and the handoff map disagree"
        )
    return names


def run_local_training_step(
    train_dir: str | Path,
    tokenizer_dir: str | Path,
    incoming: list[np.ndarray],
    handoff: HandoffMap,
    *,
    max_steps: int = 2,
    shard_index: int = 0,
) -> tuple[list[np.ndarray], dict[str, Any]]:  # pragma: no cover - env-gated (ORT-training runtime)
    """Load the package's ``train/`` artifacts, apply the incoming global adapter, run ``max_steps`` ORT
    optimizer steps on THIS client's shard, and return the updated trainable tensors (codec order) +
    metrics.

    Mirrors ``onnx_checktrain`` (``CheckpointState.load_checkpoint`` -> ``Module``/``Optimizer`` -> train
    loop). Kept intentionally small; the numerical fidelity is validated in the manual leg.
    """
    import numpy as np  # noqa: PLC0415
    from onnxruntime.training.api import CheckpointState, Module, Optimizer  # noqa: PLC0415

    # The TRAIN STAGE dir, resolved by the caller from the manifest — NOT a package root with
    # "train" appended. `<package>/train` is the on-device CACHE layout; a hub package puts the same
    # stage at `variants/<variantId>/train`, which the manifest declares in `paths.train`. Appending
    # blind produced ORT's opaque `Invalid fd was supplied: -1`, which names no file at all.
    train_dir = Path(train_dir)
    state = CheckpointState.load_checkpoint(str(train_dir / "checkpoint"))
    model = Module(str(train_dir / "training_model.onnx"), state, str(train_dir / "eval_model.onnx"))
    optimizer = Optimizer(str(train_dir / "optimizer_model.onnx"), model)

    # Apply the incoming GLOBAL adapter before training. Skipping this made every round start from the
    # client's own checkpoint, so aggregation had no effect whatsoever on the next round's clients.
    #
    # Matched BY NAME against the codec order, never by the checkpoint's own iteration order. The two
    # need not agree — codec order is (entries sorted by canonical weight name) x adapter role, while
    # `state.parameters` yields whatever order ORT stored — and a positional mismatch would quietly
    # write layer 7's `lora_A` over layer 3's. Shapes differ per layer, so most such swaps would raise;
    # "most" is not a guarantee, and the ones that did not raise would be silent corruption.
    by_name = {name: param for name, param in state.parameters if param.requires_grad}
    ordered = _codec_ordered_names(handoff, by_name)
    if incoming:
        if len(incoming) != len(ordered):
            raise HandoffError(
                f"incoming global adapter has {len(incoming)} tensors, "
                f"but the codec declares {len(ordered)} adapter factors"
            )
        for name, array in zip(ordered, incoming, strict=True):
            param = by_name[name]
            if tuple(param.data.shape) != tuple(array.shape):
                raise HandoffError(
                    f"incoming tensor for {name!r} has shape {tuple(array.shape)}, "
                    f"expected {tuple(param.data.shape)}"
                )
            param.data = np.ascontiguousarray(array, dtype=param.data.dtype)

    # THIS client's shard. Every client used to train the same two hardcoded sentences, which makes
    # FedAvg an average of identical updates — so "aggregation improves the metric" could not be
    # shown even in principle.
    tokenizer = _load_tokenizer(tokenizer_dir)
    shard = CLIENT_SHARDS[shard_index % len(CLIENT_SHARDS)]
    input_ids, attention_mask, position_ids, labels = _batch(tokenizer, shard, np=np)

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

    # Returned in CODEC order, so the server's `federated_average` lines tensors up with the specs it
    # aggregates against. Previously this returned raw `state.parameters` order and happened to work
    # only because nothing checked.
    updated = [by_name[name].data for name in ordered]
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
        incoming = list(msg.content["arrays"].to_numpy_ndarrays()) if "arrays" in msg.content else []
        # The package dir and shard index arrive in the MESSAGE, not in `ctx.node_config`.
        # `ctx.node_config["package_dir"]` was unreachable: `flwr.simulation.run_simulation` takes
        # only (server_app, client_app, num_supernodes, backend_config) — it has no node_config
        # parameter at all, so nothing could ever populate that key and every client raised KeyError
        # on its first round.
        config = msg.content["config"]
        train_dir = str(config["trainDir"])
        tokenizer_dir = str(config["tokenizerDir"])
        shard_index = int(config["shardIndex"])
        updated, metrics = run_local_training_step(
            train_dir,
            tokenizer_dir,
            incoming,
            handoff,
            max_steps=local_max_steps,
            shard_index=shard_index,
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
    train_dir: str | Path,
    tokenizer_dir: str | Path,
    node_wait_seconds: float = 120.0,
) -> Any:  # pragma: no cover - manual workflow leg (needs flwr)
    """Build a Flower ``ServerApp`` running FedAvg over the codec-ordered adapter tensors.

    This used to discard all five arguments and return a bare ``ServerApp()`` with no strategy and no
    registered handler, so the docstring's claim was false, ``federated_average`` was wired to nothing,
    and the CLI's ``--output`` was never written. The per-round aggregation + save is
    :func:`~mobiletransformers.federated.flower_sim.aggregate_round`, which is pure and unit-tested;
    only the Grid messaging below needs Flower.
    """
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415
    from logging import INFO  # noqa: PLC0415

    from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict  # noqa: PLC0415
    from flwr.common.logger import log  # noqa: PLC0415
    from flwr.serverapp import Grid, ServerApp  # noqa: PLC0415

    from mobiletransformers.federated.adapter_record import codec_tensor_specs  # noqa: PLC0415
    from mobiletransformers.federated.flower_sim import ClientUpdate, aggregate_round  # noqa: PLC0415

    specs = codec_tensor_specs(handoff)  # validate the codec resolves before the run
    app = ServerApp()

    def await_nodes(grid: Grid, round_index: int) -> list[int]:
        """Supernodes register asynchronously; round 1 always arrives before they are up.

        This used to read `grid.get_node_ids()` once and fail with "no client nodes available" on
        every run — the first thing a real simulation does is wait.
        """
        deadline = time.monotonic() + node_wait_seconds
        while True:
            node_ids = list(grid.get_node_ids())
            if node_ids:
                return node_ids
            if time.monotonic() >= deadline:
                raise HandoffError(
                    f"round {round_index}: no client nodes registered within {node_wait_seconds:.0f}s"
                )
            time.sleep(0.5)

    @app.main()
    def main(grid: Grid, ctx: Any) -> None:  # noqa: ANN001
        global_arrays: list[np.ndarray] | None = None
        eval_losses: list[float] = []

        for round_index in range(1, rounds + 1):
            node_ids = await_nodes(grid, round_index)

            messages = []
            for shard_index, node_id in enumerate(node_ids):
                # Each client gets its OWN shard index, so the clients genuinely disagree and FedAvg
                # has something to average. `packageDir` travels here because `run_simulation` has no
                # node_config parameter to put it in.
                content = RecordDict(
                    {
                        "config": ConfigRecord(
                            {
                                "trainDir": str(train_dir),
                                "tokenizerDir": str(tokenizer_dir),
                                "shardIndex": shard_index,
                            },
                        ),
                    },
                )
                if global_arrays is not None:
                    content["arrays"] = ArrayRecord(global_arrays)
                messages.append(Message(content=content, message_type=MessageType.TRAIN, dst_node_id=node_id))
            replies = grid.send_and_receive(messages)

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

            # THE metric the #35 self-check asks about, measured on the AGGREGATED tensors over a
            # held-out batch. Client-side training loss is not this: it falls whenever each client
            # fits its own shard, which happens even if aggregation is broken.
            eval_loss = evaluate_adapter(train_dir, tokenizer_dir, global_arrays, handoff)
            eval_losses.append(eval_loss)
            log(INFO, "round %d: global adapter eval loss %.6f -> %s", round_index, eval_loss, saved)

        # Relative, self-calibrating: the last round must beat the first. No absolute threshold, which
        # would encode one model and one fixture and silently measure the wrong thing when either
        # changes.
        if len(eval_losses) >= 2 and eval_losses[-1] >= eval_losses[0]:
            raise HandoffError(
                "aggregation did not improve the held-out metric: "
                f"round 1 loss {eval_losses[0]:.6f} -> round {len(eval_losses)} loss "
                f"{eval_losses[-1]:.6f}. Rounds are aggregating, but the aggregate is not learning."
            )
        (Path(output_dir) / "eval_losses.json").write_text(
            json.dumps({"evalLossPerRound": eval_losses}, indent=2),
        )

    return app


__all__ = [
    "run_local_training_step",
    "evaluate_adapter",
    "build_client_app",
    "build_server_app",
    "CLIENT_SHARDS",
    "EVAL_SENTENCES",
]
