"""Export-time proof that the training graph carries the model's parameters.

## Why this exists

Every assertion the export and the device suite made about the training stage was byte-level or
structural: the merge rewrote 60/60 `.bin` files, the handoff names resolved (`checkpoint_names.py`),
the reload succeeded. All of them were correct, and none of them looked at a number. A training package
that carried a fraction of the model would have passed every one.

That gap produced a real cost even without a real defect. A session read
``checkpoint 176.3 MB / 4 bytes ≈ 44M parameters`` against SmolLM2-135M's ~135M and concluded that "two
thirds of the model is in neither artifact" — a v1 blocker that went into `HANDOFF.md`, the `CHANGELOG`
Known issues and a deliberately-failing device test. The division was wrong: **~90% of the checkpoint
tensors are uint8, not fp32.** The graph carries all 135,436,915 parameters. Counting bytes without
counting dtypes is exactly the mistake this module exists to make impossible — which is why
:func:`summarize_training_parameters` reports **per dtype** and never divides by a single element size.

## What it checks

Two things, both cheap and both on the host:

* **Budget** — the training graph's parameter total against the parameter count of the HF model it was
  exported from, recorded by ``optimum_hf_export`` into ``training_config.json`` at the moment the
  torch model was in memory. This is exact and architecture-agnostic; no re-derivation from
  ``AutoConfig`` shapes, which would have to be maintained per architecture and would be wrong for the
  next model family added to the registry.
* **Split** — the fp32/quantized breakdown, logged and returned. A package whose trainable adapters got
  swept into quantization has a characteristic signature here (no float parameters outside the
  layernorms), and that defect has happened before — see the ``exclude_weights`` note in
  ``export/training_export.py``.

## How the parameters are found

ORT's ``TrainingBlock`` moves every name in ``requires_grad ∪ frozen_params`` out of
``graph.initializer`` and into graph **inputs**, storing the values in the checkpoint. So the training
graph's parameters are its inputs, minus the data inputs. The two are told apart by shape, not by name:
a data input carries at least one symbolic dimension (``batch_size``/``sequence_length``); a parameter
is fully concrete. Name-matching would be a second wire-format assumption to keep in sync, and this
module exists because assumptions went unchecked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mobiletransformers.exceptions import ExportError
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)

#: How far below the reference count the training graph may sit before the export fails.
#:
#: Not a guess about noise. The graph and the reference count legitimately differ in both directions:
#: PEFT **adds** adapter parameters, while tied embeddings are one tensor in the graph and one entry in
#: torch's de-duplicated ``parameters()``. Neither moves the total by a percent. The failure this gates
#: is a *structural* one — a stage that dropped a projection, a layer range, or the embedding table —
#: and the smallest such loss on a real architecture is several percent. 5% separates the two without
#: encoding any one model.
DEFAULT_TOLERANCE = 0.05

#: ONNX ``TensorProto`` element types treated as quantized parameter storage.
_QUANTIZED_ELEM_TYPES = {2, 3, 21, 22}  # UINT8, INT8, UINT4, INT4


@dataclass
class ParameterSummary:
    """The training graph's parameter accounting, per dtype."""

    total: int = 0
    #: elements keyed by ONNX ``TensorProto`` elem_type, so a byte total is never inferred from one size.
    by_elem_type: dict[int, int] = field(default_factory=dict)
    tensor_count: int = 0
    data_input_names: list[str] = field(default_factory=list)

    @property
    def quantized(self) -> int:
        """Elements held in quantized storage."""
        return sum(n for t, n in self.by_elem_type.items() if t in _QUANTIZED_ELEM_TYPES)

    @property
    def float_elements(self) -> int:
        """Elements held in float storage (the adapters, layernorms and anything left unquantized)."""
        return self.total - self.quantized

    def describe(self) -> str:
        parts = ", ".join(
            f"elem_type={t}: {n:,}" for t, n in sorted(self.by_elem_type.items(), key=lambda kv: -kv[1])
        )
        return f"{self.total:,} parameters across {self.tensor_count} tensors ({parts})"


def summarize_training_parameters(training_model_path: str | Path) -> ParameterSummary:
    """Count the parameters an ORT training graph carries, split by element type.

    Parameters live in ``graph.input`` (ORT's ``TrainingBlock`` moves them there); data inputs are told
    apart by carrying a symbolic dimension.

    Raises:
        ExportError: if the graph cannot be read, or declares no parameter inputs at all.
    """
    import onnx

    training_model_path = Path(training_model_path)
    if not training_model_path.is_file():
        raise ExportError(f"training model not found for parameter-budget check: {training_model_path}")

    # The values live in the checkpoint, not the graph, so the parameter *shapes* are all we need —
    # loading external data here would read hundreds of MB to count elements we can read from dims.
    model = onnx.load(str(training_model_path), load_external_data=False)

    summary = ParameterSummary()
    for graph_input in model.graph.input:
        tensor_type = graph_input.type.tensor_type
        dims = []
        symbolic = False
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(dim.dim_value)
            else:
                symbolic = True
                break
        if symbolic:
            summary.data_input_names.append(graph_input.name)
            continue

        elements = 1
        for d in dims:
            elements *= d
        summary.total += elements
        summary.tensor_count += 1
        summary.by_elem_type[tensor_type.elem_type] = (
            summary.by_elem_type.get(tensor_type.elem_type, 0) + elements
        )

    if summary.tensor_count == 0:
        raise ExportError(
            f"{training_model_path.name} declares no fully-shaped parameter inputs. Either the graph is "
            "not an ORT training graph, or generate_artifacts moved nothing into the checkpoint."
        )
    return summary


#: ONNX ``TensorProto`` elem_type -> the name used when declaring a graph's observed precision.
_ELEM_TYPE_NAMES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    10: "float16",
    16: "bfloat16",
    21: "uint4",
    22: "int4",
}

#: Integer/bool elem_types that hold shape, index and mask constants rather than weights. Counting them
#: would report a pure-fp32 graph as "mixed" purely because it has `Reshape` targets.
_NON_WEIGHT_ELEM_TYPES = {6, 7, 9}  # INT32, INT64, BOOL


def describe_graph_precision(model_path: str | Path) -> str:
    """What precision a graph's weights are ACTUALLY stored in, measured from its initializers.

    Exists because a variant directory named ``cpu-int4`` was found shipping a pure fp32 inference
    graph: the variant id names the *requested* quantization, which the inference export does not
    apply. Declaring the measured answer beside the graph means no reader has to infer precision from
    a directory name again.

    Returns a single dtype name when the weights are homogeneous, else ``mixed(a/b)`` ordered by
    element count, or ``unknown`` for a graph with no initializers (weights held externally as graph
    inputs — the training-graph shape, which :func:`summarize_training_parameters` handles instead).
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)

    elements: dict[int, int] = {}
    for init in model.graph.initializer:
        if init.data_type in _NON_WEIGHT_ELEM_TYPES:
            continue
        count = 1
        for dim in init.dims:
            count *= dim
        elements[init.data_type] = elements.get(init.data_type, 0) + count

    if not elements:
        return "unknown"
    ordered = sorted(elements.items(), key=lambda kv: -kv[1])
    names = [_ELEM_TYPE_NAMES.get(t, f"elem_type={t}") for t, _ in ordered]
    return names[0] if len(names) == 1 else f"mixed({'/'.join(names)})"


def verify_checkpoint_parameter_budget(
    training_model_path: str | Path,
    expected_total: int | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ParameterSummary:
    """Assert the training graph carries (about) as many parameters as the source model has.

    Args:
        training_model_path: the exported ``training_model.onnx``.
        expected_total: the source HF model's parameter count, as recorded by ``optimum_hf_export``
            into ``training_config.json``. ``None`` skips the comparison — the summary is still
            computed and logged, and the caller is expected to say the check was not run.
        tolerance: allowed shortfall as a fraction of ``expected_total``. See
            :data:`DEFAULT_TOLERANCE` for why a tolerance is correct here rather than equality.

    Returns:
        The :class:`ParameterSummary`, so callers can record the split in the package report.

    Raises:
        ExportError: naming both counts. Fails closed: a training package that cannot carry the
            pretrained weights must not ship looking like one that can.
    """
    summary = summarize_training_parameters(training_model_path)

    if expected_total is None:
        logger.warning(
            "parameter-budget check: no reference count recorded by the export; counted %s but "
            "verified nothing. The training stage is UNVERIFIED against the source model.",
            summary.describe(),
        )
        return summary

    floor = int(expected_total * (1.0 - tolerance))
    if summary.total < floor:
        shortfall = 1.0 - (summary.total / expected_total)
        raise ExportError(
            f"training graph carries {summary.total:,} parameters but the source model has "
            f"{expected_total:,} — {shortfall:.1%} short (floor {floor:,} at {tolerance:.0%} "
            f"tolerance).\n  {summary.describe()}\n"
            "A training package this far below its own model cannot start from the pretrained "
            "weights. Check which parameters generate_artifacts moved into the checkpoint vs left as "
            "graph initializers, and whether the quantizer swept tensors it should have excluded "
            "(export/training_export.py, `exclude_weights`)."
        )

    logger.info(
        "parameter-budget check: %s (reference %s, %+.1f%%)",
        summary.describe(),
        f"{expected_total:,}",
        100.0 * (summary.total / expected_total - 1.0),
    )
    if summary.float_elements == 0:
        raise ExportError(
            f"training graph has {summary.total:,} parameters but NONE in float storage. The PEFT "
            "adapters were swept into quantization, so there is nothing trainable to compute a "
            "gradient for (see `exclude_weights` in export/training_export.py)."
        )
    return summary
