"""Merger registry — resolves the device-side merger variant + descriptive ONNX filename from data.

Replaces the peft_method-keyed if/elif merger dispatch (``artifact/onnx_builder.py``) and the four
near-duplicate ``create_*_merger_model{,_2}`` factories (``artifact/merger.py``). The
resolved ``MergerVariant`` + filename go into the ``weight_handoff_map.json`` so the C++ side selects
the merger session from data, not string literals.

Scope note: ``resolve_merger`` / ``MergerSpec`` (the registry contract) are owned and implemented here.
The single ``build_merger_model`` ONNX-graph builder that collapses the four legacy factories — and the
C++ ``get_merger_type``/``run_merger_model`` rewrite — are *wired* by #9
(``01_code_plans/01``), which owns the on-disk merge filename contract and the golden-equivalence test.
Until then ``build_merger_model`` fails closed rather than silently emitting a wrong graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from mobiletransformers._typing import PathLike
from mobiletransformers.config.constants import MergerVariant, PEFTMethod
from mobiletransformers.config.registry.peft import get_peft_spec
from mobiletransformers.exceptions import MergeError, UnsupportedModelError


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

    The single parameterized builder that replaces ``create_lora_merger_model{,_2}`` /
    ``create_mars_merger_model{,_2}`` is wired by #9 (it owns the on-disk merge contract + the
    golden-equivalence test against the legacy factories). Until then this fails closed.
    """
    raise MergeError(
        "build_merger_model is not yet wired; the four-factory ONNX collapse lands with "
        "01_code_plans/01 (#9). resolve_merger + MergerSpec are available now."
    )


__all__ = ["MergerSpec", "resolve_merger", "build_merger_model"]
