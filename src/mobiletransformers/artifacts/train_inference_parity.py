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
#: Calibrated, not guessed. Weight-only uint8 quantization has a ~3.2% RMS error on the embedding table,
#: which moves a cross-entropy by tenths of a nat. A graph that had genuinely lost its pretrained
#: weights would sit at or above ``ln(vocab_size)``, i.e. nats away. 1.5 admits the former and excludes
#: the latter by a wide margin.
#:
#: **Re-measured 2026-08-15**, after :func:`probe_token_ids` started tokenizing with the package's own
#: tokenizer. The figures this constant used to cite (SmolLM2 "13.861 fp32 vs 14.254") came from the
#: borrowed-id probe and are not comparable to anything now:
#:
#: ===========================  ==========  =========  =======
#: package                      inference   training   delta
#: ===========================  ==========  =========  =======
#: SmolLM2-135M-Instruct            3.1803     3.3264   0.1461
#: google/gemma-3-270m              3.2870     3.3057   0.0188
#: google/functiongemma-270m-it     5.1658     6.0495   0.8837
#: ===========================  ==========  =========  =======
#:
#: FunctionGemma's higher absolute loss and wider gap are the probe being off-distribution for it, not
#: a defect: it is fine-tuned hard on function-call JSON, and re-probed on such JSON the same package
#: gives 4.4513 / 4.1526 for a delta of **0.2987**. Quantization error is largest exactly where the
#: model is least confident, so a specialised model measured on generic English is the worst case this
#: bound has to tolerate — which is the reason to keep 1.5 rather than tighten it toward the 0.02-0.15
#: the general-purpose models show.
MAX_LOSS_DELTA_NATS = 1.5

#: FALLBACK token ids, used only when the package ships no tokenizer to derive better ones from.
#:
#: These are **Llama-family ids** — they decode to English under SmolLM2's tokenizer and to nothing in
#: particular under anyone else's. That was fine while the claim was only "both graphs see the same
#: tokens", but it makes the absolute losses meaningless off that family and, worse, it makes the
#: DELTA fragile: on a sequence the model finds absurd it is confidently wrong, and quantization error
#: on a confident-but-wrong distribution is far larger than on text the model expects.
#:
#: Measured 2026-08-15 on `google/functiongemma-270m-it` (vocab 262144, ln(vocab) = 12.48):
#: inference 25.70 / training 27.14 nats — both roughly TWICE uniform-random — for a delta of 1.4417
#: against a 1.5 bound. The package was fine; the probe was gibberish to it. A good package came within
#: 0.06 nats of failing its own integrity gate.
#:
#: :func:`_probe_batch` therefore prefers the package's OWN tokenizer. Keep these as the last resort.
FALLBACK_PROBE_INPUT_IDS: tuple[tuple[int, ...], ...] = (
    (1, 338, 263, 1243, 310, 278, 1904, 29889),
    (1, 450, 4996, 17354, 1701, 29916, 432, 17204),
)

#: Deprecated alias for :data:`FALLBACK_PROBE_INPUT_IDS`, kept so external references keep resolving.
PROBE_INPUT_IDS = FALLBACK_PROBE_INPUT_IDS

#: The probe text, tokenized with the package's own tokenizer when one is available.
#:
#: Ordinary English, deliberately: the point of the check is to compare two graphs on input the model
#: was trained to model, so that the gap between them is quantization error and nothing else.
PROBE_TEXTS: tuple[str, ...] = (
    "This is a test of the model.",
    "The quick brown fox jumps over the lazy dog.",
)

#: How many tokens of each probe text to keep. Fixed so the batch is rectangular without padding, and
#: short enough that every tokenizer produces at least this many for the texts above.
PROBE_SEQUENCE_LENGTH = 8


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


def find_package_tokenizer(reference: str | Path) -> Path | None:
    """Locate ``tokenizer.json`` for the package containing ``reference`` (an inference model or dir).

    Looks beside the graph first (the flat ``inference/`` layout puts one there), then at the package's
    ``shared/tokenizer/``. Returns ``None`` rather than raising: a missing tokenizer degrades the probe,
    it does not invalidate the comparison.
    """
    reference = Path(reference)
    start = reference.parent if reference.is_file() else reference
    candidates = [
        start / "tokenizer.json",
        start.parent / "tokenizer" / "tokenizer.json",
        # <pkg>/variants/<variant>/inference -> <pkg>/shared/tokenizer
        start.parent.parent.parent / "shared" / "tokenizer" / "tokenizer.json",
    ]
    return next((c for c in candidates if c.is_file()), None)


def probe_token_ids(tokenizer_json: str | Path | None) -> tuple[tuple[int, ...], ...]:
    """Tokenize :data:`PROBE_TEXTS` with the package's tokenizer, or fall back to fixed ids.

    Using the package's OWN tokenizer is what makes the resulting loss interpretable: an absolute
    cross-entropy is only comparable to ``ln(vocab_size)`` if the ids actually mean the text they were
    meant to mean. With borrowed ids a Gemma-vocabulary model scored ~25 nats against a 12.5-nat
    uniform bound, and the train/inference delta inflated to within 0.06 of the failure threshold.
    """
    if tokenizer_json is not None:
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tokenizer_json))
            rows = []
            for text in PROBE_TEXTS:
                ids = tok.encode(text).ids
                if len(ids) < PROBE_SEQUENCE_LENGTH:
                    raise ValueError(f"{text!r} tokenized to only {len(ids)} ids")
                rows.append(tuple(ids[:PROBE_SEQUENCE_LENGTH]))
            return tuple(rows)
        except Exception as exc:  # noqa: BLE001 — any tokenizer problem degrades, never fails, the probe
            logger.warning(
                "could not tokenize the parity probe with %s (%s); falling back to fixed ids, so the "
                "absolute losses below are NOT comparable to ln(vocab_size)",
                tokenizer_json,
                exc,
            )
    return FALLBACK_PROBE_INPUT_IDS


def _probe_batch(
    tokenizer_json: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(input_ids, attention_mask, position_ids) for the probe, in the package's own vocabulary."""
    import numpy as np

    input_ids = np.asarray(probe_token_ids(tokenizer_json), dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :].repeat(input_ids.shape[0], 0)
    return input_ids, attention_mask, position_ids


def inference_graph_loss(inference_model_path: str | Path, tokenizer_json: str | Path | None = None) -> float:
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
    if tokenizer_json is None:
        tokenizer_json = find_package_tokenizer(inference_model_path)
    input_ids, attention_mask, position_ids = _probe_batch(tokenizer_json)

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


def training_graph_loss(train_dir: str | Path, tokenizer_json: str | Path | None = None) -> float:
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
    input_ids, attention_mask, position_ids = _probe_batch(tokenizer_json)

    # Feed BY NAME, in the order the graph declares — not by a hardcoded decoder tuple.
    #
    # This used to be `module(input_ids, attention_mask, position_ids, labels)`, which silently
    # assumed every trainable graph takes those four inputs in that order. Gemma-3 does not:
    # `Gemma3TextOnnxConfig` declares no `position_ids`, so its training graph has three user inputs
    # and the fourth positional argument fell off the end as
    # "Train input name index out of range. Expected in range [0-3). Actual: 3" — an ORT-internal
    # message that names neither the graph nor the offending input.
    #
    # Reading the names off the graph makes the probe follow whatever the exporter actually produced,
    # so a new architecture with a different input set needs no edit here. Anything unrecognised fails
    # closed rather than being fed a zero tensor that would quietly change the measured loss.
    available = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        # Labels UNSHIFTED — the graph applies the causal shift itself. See the note in
        # `artifacts/builder.py::onnx_checktrain`, where pre-shifting inflated every printed loss.
        "labels": input_ids.copy(),
    }
    # ORT's OWN list of user inputs, not a re-derivation from the graph. In a `generate_artifacts`
    # training graph the trainable weights are graph *inputs* too (that is the format), so filtering
    # `graph.input` against the initializers yields every LoRA factor and layernorm as well — the
    # parameters ORT feeds itself from the checkpoint. `Module.input_names()` is the same list
    # `train_step` indexes into, which is precisely the one that must match.
    module.train()
    user_inputs = list(module.input_names())

    unknown = [name for name in user_inputs if name not in available]
    if unknown:
        raise ExportError(
            f"the training graph declares inputs the parity probe cannot supply: {unknown} "
            f"(graph inputs: {user_inputs}). Add them to the probe batch, or the measured loss would "
            "be taken against tensors this check never set."
        )

    outputs = module(*(available[name] for name in user_inputs))
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

    # Resolve the tokenizer ONCE and hand the same one to both legs. Letting each find its own would
    # reintroduce the very failure mode this check exists to catch: two graphs scored on different
    # tokens produce a delta that means nothing, and it would look exactly like a real disagreement.
    tokenizer_json = find_package_tokenizer(inference_model_path)
    if tokenizer_json is None:
        logger.warning(
            "no tokenizer.json found for %s — the parity probe falls back to fixed Llama-family ids. "
            "The DELTA below is still a like-for-like comparison, but the absolute losses are not "
            "meaningful for this model's vocabulary.",
            inference_model_path,
        )

    inference_loss = inference_graph_loss(inference_model_path, tokenizer_json)
    training_loss = training_graph_loss(train_dir, tokenizer_json)
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
