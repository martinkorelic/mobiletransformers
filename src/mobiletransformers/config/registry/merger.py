"""Merger registry — resolves the device-side merger variant + descriptive ONNX filename from data.

Replaces the peft_method-keyed if/elif merger dispatch (``artifact/onnx_builder.py``) and the four
near-duplicate ``create_*_merger_model{,_2}`` factories (``artifact/merger.py``). The
resolved ``MergerVariant`` + filename go into the ``weight_handoff_map.json`` so the C++ side selects
the merger session from data, not string literals.

Scope note: ``resolve_merger`` / ``MergerSpec`` (the registry contract) are owned and implemented here.
The single ``build_merger_model`` ONNX-graph builder that collapses the four legacy factories — and the
C++ ``get_merger_type``/``run_merger_model`` rewrite — are *wired* by #9
which owns the on-disk merge filename contract and the golden-equivalence test.
Until then ``build_merger_model`` fails closed rather than silently emitting a wrong graph.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mobiletransformers._typing import PathLike
from mobiletransformers.config.constants import MergerVariant, PEFTMethod
from mobiletransformers.config.registry.peft import get_peft_spec
from mobiletransformers.exceptions import MergeError, UnsupportedModelError
from mobiletransformers.utils.logging import get_logger

if TYPE_CHECKING:
    import onnx

logger = get_logger(__name__)


@dataclass(frozen=True)
class MergerSpec:
    peft_method: PEFTMethod
    quant_in: bool
    quant_out: bool
    variant: MergerVariant  # resolved device-side tag ("lora"/"lora_q"/"mars_q")
    output_filename: str  # descriptive, e.g. "merger_mars_q_qin_qout.onnx" — NO "_2" suffixes


def resolve_merger(peft_method: PEFTMethod, quant_in: bool, quant_out: bool) -> MergerSpec:
    """Resolve the merger variant + descriptive filename for a (method, quant_in, quant_out) tuple.

    The ``_q`` variants correspond to quantized inputs (mirrors the legacy ``*_qmerger`` naming).
    Fails closed for methods that produce no merger (``all`` / ``nolora``).
    """
    spec = get_peft_spec(peft_method)
    variant = spec.merger_variant_q if quant_in else spec.merger_variant_fp
    if variant is None:
        raise UnsupportedModelError(f"PEFT method {peft_method.value} has no merger variant")
    in_tag = "qin" if quant_in else "fpin"
    out_tag = "qout" if quant_out else "fpout"
    output_filename = f"merger_{variant.value}_{in_tag}_{out_tag}.onnx"
    return MergerSpec(peft_method, quant_in, quant_out, variant, output_filename)


def build_merger_model(spec: MergerSpec, output_path: PathLike) -> None:
    """Build the merger ONNX graph for ``spec`` at ``output_path``.

    The single parameterized builder that collapses the four legacy factories
    (``create_lora_merger_model{,_2}`` / ``create_mars_merger_model{,_2}`` in ``artifact/merger.py``).
    Two axes fully determine the graph, both carried on the :class:`MergerSpec`:

    * **family** — ``variant`` root: ``lora``/``lora_q`` → LoRA math; ``mars_q`` → MARS math.
    * **quantization** — ``quant_in`` / ``quant_out``, honored *independently* (the ``_2``-factory
      superset). Do NOT re-derive quantization from ``variant``: ``resolve_merger`` keys the ``_q``
      variant tag on ``quant_in`` only, while the graph's input *and* output quantization are the two
      spec flags. A float-input variant (``lora``) with ``quant_out=True`` is a valid graph.

    The emitted node topology, IO names, dtypes, opset (11), producer metadata and ``metadata_props``
    reproduce the legacy ``*_2`` factories so the graph is byte-comparable (golden-equivalence test).
    """
    import onnx

    if spec.variant in (MergerVariant.LORA, MergerVariant.LORA_Q):
        model = _build_lora_merger(spec.quant_in, spec.quant_out)
    elif spec.variant is MergerVariant.MARS_Q:
        model = _build_mars_merger(spec.quant_in, spec.quant_out)
    else:  # pragma: no cover - MergerVariant is a closed enum; guards a future member.
        raise MergeError(f"no merger graph builder for variant {spec.variant.value!r}")

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    logger.info(
        "built merger model variant=%s quant_in=%s quant_out=%s -> %s",
        spec.variant.value,
        spec.quant_in,
        spec.quant_out,
        output_path,
    )


def _append_quant_metadata(model: onnx.ModelProto, quant_in: bool, quant_out: bool) -> None:
    """Append the four ``metadata_props`` the legacy factories emit (byte-comparable)."""
    entries = (
        ("quantized_inputs", str(quant_in).lower()),
        ("quantized_outputs", str(quant_out).lower()),
        ("input_type", "quantized" if quant_in else "float"),
        ("output_type", "quantized" if quant_out else "float"),
    )
    for key, value in entries:
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value


def _build_lora_merger(quant_in: bool, quant_out: bool) -> onnx.ModelProto:
    """LoRA merger graph: ``merged = base + alpha * (adapter_B @ adapter_A)``.

    Reproduces ``artifact/merger.py::create_lora_merger_model_2``.
    """
    from onnx import TensorProto, helper

    inputs = []
    outputs = []
    nodes = []

    if quant_in:
        inputs.append(
            helper.make_tensor_value_info(
                "weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]
            )
        )
        inputs.append(helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, []))
    else:
        inputs.append(
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, ["out_features", "in_features"])
        )

    inputs.append(helper.make_tensor_value_info("adapter_A", TensorProto.FLOAT, ["rank", "in_features"]))
    inputs.append(helper.make_tensor_value_info("adapter_B", TensorProto.FLOAT, ["out_features", "rank"]))
    inputs.append(helper.make_tensor_value_info("alpha", TensorProto.FLOAT, []))

    if quant_out:
        outputs.append(
            helper.make_tensor_value_info(
                "merged_weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]
            )
        )
        outputs.append(helper.make_tensor_value_info("merged_scale", TensorProto.FLOAT, []))
        outputs.append(helper.make_tensor_value_info("merged_zero_point", TensorProto.UINT8, []))
    else:
        outputs.append(
            helper.make_tensor_value_info("merged_weight", TensorProto.FLOAT, ["out_features", "in_features"])
        )

    if quant_in:
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                inputs=["weight_quantized", "x_scale", "x_zero_point"],
                outputs=["base_weight_fp32"],
                name="dequantize_base_weights",
            )
        )
        base_input = "base_weight_fp32"
    else:
        base_input = "weight"

    nodes.append(
        helper.make_node(
            "MatMul", inputs=["adapter_B", "adapter_A"], outputs=["lora_delta"], name="compute_lora_delta"
        )
    )
    nodes.append(
        helper.make_node(
            "Mul", inputs=["lora_delta", "alpha"], outputs=["scaled_lora_delta"], name="scale_lora_delta"
        )
    )
    nodes.append(
        helper.make_node(
            "Add",
            inputs=[base_input, "scaled_lora_delta"],
            outputs=["merged_weight_fp32"],
            name="add_lora_delta",
        )
    )

    if quant_out:
        nodes.append(
            helper.make_node(
                "DynamicQuantizeLinear",
                inputs=["merged_weight_fp32"],
                outputs=["merged_weight_quantized", "merged_scale", "merged_zero_point"],
                name="quantize_merged",
            )
        )
    else:
        nodes.append(
            helper.make_node(
                "Identity", inputs=["merged_weight_fp32"], outputs=["merged_weight"], name="identity_output"
            )
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="LoRAMergerModel",
        inputs=inputs,
        outputs=outputs,
        doc_string=(
            f"LoRA merger model for merging adapters "
            f"(inputs: {'quantized' if quant_in else 'float'}, "
            f"outputs: {'quantized' if quant_out else 'float'})."
        ),
    )
    model = helper.make_model(
        graph,
        producer_name="LoRAMerger_v1.0",
        producer_version="1.0.0",
        doc_string=(
            f"LoRA (Low-Rank Adaptation) weight merger for PEFT models. "
            f"Input quantization: {quant_in}, Output quantization: {quant_out}."
        ),
        model_version=1,
        opset_imports=[helper.make_opsetid("", 11)],
    )
    _append_quant_metadata(model, quant_in, quant_out)
    return model


def _build_mars_merger(quant_in: bool, quant_out: bool) -> onnx.ModelProto:
    """MARS merger graph: ``merged = base + alpha * (adapter_B @ intermediate_chunk @ shared_A)``.

    Reproduces ``artifact/merger.py::create_mars_merger_model_2`` (incl. the ``adapter_index``/``rank``
    slice-chunk selection and the ``com.martinkorelic.mars`` model domain).
    """
    from onnx import TensorProto, helper

    if quant_in:
        inputs = [
            helper.make_tensor_value_info(
                "weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]
            ),
            helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, []),
            helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, []),
        ]
    else:
        inputs = [
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, ["out_features", "in_features"]),
        ]

    if quant_out:
        outputs = [
            helper.make_tensor_value_info(
                "merged_weight_quantized", TensorProto.UINT8, ["out_features", "in_features"]
            ),
            helper.make_tensor_value_info("merged_zero_point", TensorProto.UINT8, []),
            helper.make_tensor_value_info("merged_scale", TensorProto.FLOAT, []),
        ]
    else:
        outputs = [
            helper.make_tensor_value_info(
                "merged_weight", TensorProto.FLOAT, ["out_features", "in_features"]
            ),
        ]

    inputs.extend(
        [
            helper.make_tensor_value_info("shared_A", TensorProto.FLOAT, ["shared_rank", "in_features"]),
            helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, ["n_times_rank", "shared_rank"]),
            helper.make_tensor_value_info("adapter_B", TensorProto.FLOAT, ["out_features", "rank"]),
            helper.make_tensor_value_info("adapter_index", TensorProto.INT64, []),
            helper.make_tensor_value_info("rank", TensorProto.INT64, []),
            helper.make_tensor_value_info("alpha", TensorProto.FLOAT, []),
        ]
    )

    nodes = []
    if quant_in:
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                inputs=["weight_quantized", "x_scale", "x_zero_point"],
                outputs=["base_weight_fp32"],
                name="dequantize_base_weights",
            )
        )
    else:
        nodes.append(
            helper.make_node(
                "Identity", inputs=["weight"], outputs=["base_weight_fp32"], name="pass_through_base_weight"
            )
        )

    nodes.append(
        helper.make_node("Mul", ["adapter_index", "rank"], ["slice_start"], name="compute_slice_start")
    )
    nodes.append(helper.make_node("Add", ["slice_start", "rank"], ["slice_end"], name="compute_slice_end"))
    nodes.append(
        helper.make_node(
            "Unsqueeze", ["slice_start"], ["slice_start_1d"], axes=[0], name="unsqueeze_slice_start"
        )
    )
    nodes.append(
        helper.make_node("Unsqueeze", ["slice_end"], ["slice_end_1d"], axes=[0], name="unsqueeze_slice_end")
    )
    nodes.append(
        helper.make_node(
            "Constant", [], ["axes_0"], value=helper.make_tensor("axes_0_tensor", TensorProto.INT64, [1], [0])
        )
    )
    nodes.append(
        helper.make_node(
            "Slice",
            ["intermediate", "slice_start_1d", "slice_end_1d", "axes_0"],
            ["chunked_intermediate"],
            name="slice_intermediate",
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            ["adapter_B", "chunked_intermediate"],
            ["adapter_times_chunk"],
            name="adapter_chunk_matmul",
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul", ["adapter_times_chunk", "shared_A"], ["lora_delta_prealpha"], name="final_matmul"
        )
    )
    nodes.append(
        helper.make_node("Mul", ["lora_delta_prealpha", "alpha"], ["lora_delta"], name="scale_alpha")
    )
    nodes.append(
        helper.make_node("Add", ["base_weight_fp32", "lora_delta"], ["merged_weight_fp32"], name="add_delta")
    )

    if quant_out:
        nodes.append(
            helper.make_node(
                "DynamicQuantizeLinear",
                inputs=["merged_weight_fp32"],
                outputs=["merged_weight_quantized", "merged_scale", "merged_zero_point"],
                name="quantize_merged",
            )
        )
    else:
        nodes.append(
            helper.make_node(
                "Identity", inputs=["merged_weight_fp32"], outputs=["merged_weight"], name="identity_output"
            )
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="MARS Merger",
        inputs=inputs,
        outputs=outputs,
        doc_string=(
            f"MARS merger model for merging adapters "
            f"(inputs: {'quantized' if quant_in else 'float'}, "
            f"outputs: {'quantized' if quant_out else 'float'})."
        ),
    )
    model = helper.make_model(
        graph,
        producer_name="MARS_Merger_v1.0",
        producer_version="1.0.0",
        doc_string=(
            f"MARS (Multi-Adapter Rank Sharing) weight merger for PEFT models. "
            f"Input quantization: {quant_in}, Output quantization: {quant_out}."
        ),
        model_version=1,
        domain="com.martinkorelic.mars",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    _append_quant_metadata(model, quant_in, quant_out)
    return model


def emit_merger_models(
    output_dir: str,
    peft_method: PEFTMethod,
    quant_out: bool,
    quant_ins: Iterable[bool] = (True, False),
    extra_methods: Iterable[PEFTMethod] = (),
) -> dict[str, str]:
    """Emit the merger ONNX graph(s) a package needs, via the registry (no hand-picked factories).

    Returns ``{MergerVariant.value: output_filename}`` for the handoff map's ``mergerModels``. Filenames
    are descriptive (``merger_<variant>_<qin|fpin>_<qout|fpout>.onnx``). ``extra_methods`` lets a MARS
    package also carry the LoRA mergers used for its non-MARS layers (device mixes per-layer).
    """
    os.makedirs(output_dir, exist_ok=True)
    emitted: dict[str, str] = {}
    for method in (peft_method, *extra_methods):
        for quant_in in quant_ins:
            spec = resolve_merger(method, quant_in=quant_in, quant_out=quant_out)
            build_merger_model(spec, os.path.join(output_dir, spec.output_filename))
            emitted[spec.variant.value] = spec.output_filename
    return emitted


# `emit_merger_models` was defined here and imported by artifacts/builder.py, but omitted from
# __all__ — so the module's declared public surface disagreed with its actual one. Surfaced by
# the S9 symbol golden, which reads __all__ when a module declares one.
__all__ = ["MergerSpec", "resolve_merger", "build_merger_model", "emit_merger_models"]
