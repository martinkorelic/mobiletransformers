# DECOMPOSE(#5) — DONE (#9): the four near-duplicate ONNX-graph factories that used to live here
# (create_lora_merger_model{,_2} / create_mars_merger_model{,_2}) are collapsed into the single
# parameterized build_merger_model(MergerSpec) in mobiletransformers.config.registry.merger. On-disk
# merger filenames are now descriptive (MergerSpec.output_filename) and recorded in the handoff map's
# `mergerModels`, replacing the legacy lora_merger_model.onnx / lora_qmerger_model.onnx / mars_*.onnx
# names and the string-keyed factory dispatch.
#
# This module remains as the offline merger-emit driver: it iterates resolve_merger() and calls
# build_merger_model() so both the offline (here) and on-device (weight_merger.cpp) paths agree on
# graph + filename. The numerical merge itself runs on device.
"""Offline merger-model emit driver (#9). See module comment above."""

from __future__ import annotations

import os
from collections.abc import Iterable

from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.config.registry.merger import (
    MergerSpec,
    build_merger_model,
    resolve_merger,
)

__all__ = ["emit_merger_models", "build_merger_model", "resolve_merger", "MergerSpec"]


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


if __name__ == "__main__":
    # Emit the standard merger set (descriptive filenames) into the CWD for inspection.
    emit_merger_models(".", PEFTMethod.LORA, quant_out=True)
    emit_merger_models(".", PEFTMethod.MARS, quant_out=True, extra_methods=(PEFTMethod.LORA,))
    print("emitted merger models (descriptive filenames) to CWD")
