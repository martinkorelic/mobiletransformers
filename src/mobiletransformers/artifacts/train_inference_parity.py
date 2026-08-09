"""Export-time proof that a package's train and inference halves agree numerically.

## Why this exists

A package ships two graphs built by two different toolchains from one model: ``inference/model.onnx``
(optimum export, external-initializer split) and ``train/training_model.onnx`` (torch → ONNX →
dynamic quantization → ``generate_artifacts``). Everything that compared them compared *names*. Nothing
compared *numbers*, in either direction, at any point in the pipeline or the device suite.

That is not hypothetical. A variant directory named ``cpu-int4`` was found shipping a **pure fp32**
inference graph (273 initializers, all float32) beside a **uint8 weight-quantized** training graph. Both
halves work; nobody had established what the gap between them is, so when a device test read a
higher-than-expected training loss there was no reference to judge it against, and the number was
misread as missing weights (see ``parameter_budget.py``).

This module supplies that reference: the same tokens through both graphs, one loss each, one delta.

## What "agreement" means here

Not equality. The two graphs are *deliberately* different — quantization is the whole point of the
training half — so the check is a **bound on the disagreement**, not a bit-parity assertion. Measured
weight-quantization error on the embedding table alone is ~3.2% RMS, which moves a cross-entropy by
tenths of a nat. A structural defect — wrong weights, a dropped layer range, a mis-parameterised graph
— moves it by whole nats and pushes the loss toward or past the uniform-prediction floor
``ln(vocab_size)``. :data:`MAX_LOSS_DELTA_NATS` sits between those two regimes.

## Runtime requirements

Both legs need ``onnxruntime``; the training leg additionally needs ``onnxruntime.training``. Those live
in mutually-exclusive dependency profiles from the export profile, so this check runs when the training
stage runs (``ort-training-local``) and reports honestly that it did not run otherwise. It never
silently passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mobiletransformers.exceptions import ExportError
from mobiletransformers.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

#: Largest train-vs-inference cross-entropy gap, in nats, that is attributed to quantization.
#:
#: Calibrated, not guessed. On SmolLM2-135M the two halves differ by ~0.39 nats (13.861 fp32 vs 14.254
#: measured on the uint8 training graph) — that is weight-only uint8 quantization with a ~3.2% RMS
#: error on the embedding table. A graph that had genuinely lost its pretrained weights would sit at or
#: above ``ln(vocab_size)`` (10.80 for this tokenizer), i.e. nats away. 1.5 admits the former with room
#: for a wider-quantized variant and still excludes the latter by a wide margin.
MAX_LOSS_DELTA_NATS = 1.5

#: Deterministic token ids for the probe batch. Arbitrary but FIXED: the check compares two graphs
#: against each other on identical input, so the only requirement is that both see the same tokens and
#: that the batch exercises a real sequence length. Ids stay well inside every supported vocabulary.
PROBE_INPUT_IDS: tuple[tuple[int, ...], ...] = (
    (1, 338, 263, 1243, 310, 278, 1904, 29889),
    (1, 450, 4996, 17354, 1701, 29916, 432, 17204),
)


@dataclass
class ParityResult:
    """Outcome of the train-vs-inference comparison."""

    inference_loss: float
    training_loss: float

    @property
    def delta(self) -> float:
        return abs(self.training_loss - self.inference_loss)

    def describe(self) -> str:
        return (
            f"inference {self.inference_loss:.4f} vs training {self.training_loss:.4f} nats "
            f"(delta {self.delta:.4f})"
        )


def causal_cross_entropy(logits: np.ndarray, input_ids: np.ndarray) -> float:
    """Mean next-token cross-entropy in nats, HF-style.

    Applies the causal shift the way the exported graphs do — ``logits[:, :-1]`` against
    ``input_ids[:, 1:]`` — so a caller cannot accidentally double-shift, which is the exact defect that
    made the old host-side ``onnx_checktrain`` number incomparable to the device's.

    Pure numpy so it is unit-testable in the core profile, with no onnxruntime present.
    """
    import numpy as np

    if logits.ndim != 3:
        raise ValueError(f"expected logits [batch, seq, vocab], got shape {logits.shape}")

    shifted_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    # Log-softmax in a numerically stable form; float64 so the subtraction below cannot lose precision
    # at the magnitudes a broken graph produces (fp values in the 1e8 range have been observed).
    x = shifted_logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    log_probs = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

    batch_idx, pos_idx = np.indices(targets.shape)
    token_log_probs = log_probs[batch_idx, pos_idx, targets]
    return float(-token_log_probs.mean())


def _probe_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(input_ids, attention_mask, position_ids) for the fixed probe."""
    import numpy as np

    input_ids = np.asarray(PROBE_INPUT_IDS, dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :].repeat(input_ids.shape[0], 0)
    return input_ids, attention_mask, position_ids


def inference_graph_loss(inference_model_path: str | Path) -> float:
    """Run the packaged inference graph on the probe batch and return its next-token loss.

    External initializers resolve relative to the model file, which is exactly the flat ``inference/``
    layout the package defines, so no weight wiring is needed here.
    """
    import numpy as np
    import onnxruntime as ort

    inference_model_path = Path(inference_model_path)
    if not inference_model_path.is_file():
        raise ExportError(f"inference model not found for parity check: {inference_model_path}")

    session = ort.InferenceSession(str(inference_model_path), providers=["CPUExecutionProvider"])
    input_ids, attention_mask, position_ids = _probe_batch()

    supplied: dict[str, np.ndarray] = {}
    available = {i.name for i in session.get_inputs()}
    for name, value in (
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
        ("position_ids", position_ids),
    ):
        if name in available:
            supplied[name] = value

    # A KV-cache-enabled graph declares empty past_key_values inputs; feed zero-length tensors so the
    # first (prefill) step is what runs — the same thing the device does on a fresh conversation.
    for model_input in session.get_inputs():
        if model_input.name in supplied:
            continue
        if not model_input.name.startswith("past_key_values"):
            raise ExportError(
                f"inference graph declares an input the parity probe cannot supply: {model_input.name}"
            )
        shape = [d if isinstance(d, int) else 0 for d in model_input.shape]
        shape[0] = input_ids.shape[0]
        supplied[model_input.name] = np.zeros(shape, dtype=np.float32)

    logits = session.run(["logits"], supplied)[0]
    return causal_cross_entropy(np.asarray(logits), input_ids)


def training_graph_loss(train_dir: str | Path) -> float:
    """Run one forward pass of the training graph on the probe batch and return its loss.

    No optimizer step is taken: the question is what the graph's loss IS at step 0, not whether it
    falls. Requires ``onnxruntime.training`` (the ``ort-training-local`` profile).
    """
    from onnxruntime.training.api import CheckpointState, Module

    train_dir = Path(train_dir)
    for required in ("training_model.onnx", "eval_model.onnx", "checkpoint"):
        if not (train_dir / required).exists():
            raise ExportError(f"training artifact missing for parity check: {train_dir / required}")

    state = CheckpointState.load_checkpoint(str(train_dir / "checkpoint"))
    module = Module(
        str(train_dir / "training_model.onnx"),
        state,
        str(train_dir / "eval_model.onnx"),
    )
    input_ids, attention_mask, position_ids = _probe_batch()

    # Labels UNSHIFTED — the graph applies the causal shift itself. See the note in
    # `artifacts/builder.py::onnx_checktrain`, where pre-shifting inflated every printed loss.
    module.train()
    outputs = module(input_ids, attention_mask, position_ids, input_ids.copy())
    return float(outputs[0])


def verify_train_inference_parity(
    inference_model_path: str | Path,
    train_dir: str | Path,
    *,
    max_delta_nats: float = MAX_LOSS_DELTA_NATS,
) -> ParityResult | None:
    """Assert both halves of the package agree on the same tokens, within quantization error.

    Returns ``None`` (and warns) when the runtime for either leg is unavailable — the check is skipped
    loudly, never passed silently.

    Raises:
        ExportError: naming both losses and the delta, when the two halves disagree by more than
            ``max_delta_nats``.
    """
    try:
        import onnxruntime  # noqa: F401
        from onnxruntime.training.api import CheckpointState  # noqa: F401
    except ImportError as exc:
        logger.warning(
            "train/inference parity check SKIPPED (%s). The two halves of this package have not been "
            "compared numerically; run the training stage under the ort-training-local profile to "
            "verify them.",
            exc,
        )
        return None

    inference_loss = inference_graph_loss(inference_model_path)
    training_loss = training_graph_loss(train_dir)
    result = ParityResult(inference_loss=inference_loss, training_loss=training_loss)

    if result.delta > max_delta_nats:
        raise ExportError(
            f"train and inference halves of this package disagree: {result.describe()}, over the "
            f"{max_delta_nats} nat bound.\n"
            "Quantization alone does not move a cross-entropy this far. Either the training graph is "
            "not carrying the same weights as the inference graph, or the two were exported from "
            "different models/revisions. Check the quantization settings the training stage used "
            "against what the inference stage shipped."
        )

    logger.info("train/inference parity: %s, within the %.1f nat bound", result.describe(), max_delta_nats)
    return result
