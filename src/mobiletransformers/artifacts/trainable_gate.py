"""Requested-vs-realized trainable-tensor gate.

Split out of ``artifacts/builder.py`` so it is importable — and therefore testable — in the core env.
``builder.py`` imports ``onnxruntime.training`` at module scope, so nothing in it can be unit-tested
outside the ``ort-training-local`` profile; this decision is pure, and the same extract-the-decision
move as ``cpp/training_inputs.h``.

## The seam this closes

``training_config.json`` declares which parameters require a gradient. The exported graph declares
which initializers exist, and in what dtype. Both halves were correct on their own and **nothing
compared them** — the recurring failure shape in this project.

What slipped through: the dynamic quantizer replaced MARS's shared adapter with
``_quantized``/``_scale``/``_zero_point`` companions. Those are integer tensors and cannot take a
gradient, so ``gen_artifacts`` correctly routed them to ``frozen_params`` — silently demoting tensors
the export had just declared trainable. Training still ran and the loss still fell, because the
per-module factors were still trainable; MARS was simply not training the shared matrices that make it
MARS. Measured before the fix: **12 realized against 24 requested** on a BERT encoder, **4 against 8**
on a Llama decoder. LoRA was unaffected, which is why it went unnoticed.
"""

from __future__ import annotations

from collections.abc import Sequence

from mobiletransformers.exceptions import ExportError

#: Quantizer-produced companions of a weight: the packed payload plus its dequantization parameters.
#: Mirrors the quantized-triple vocabulary `HandoffMap.validate` guards on the emit side.
QUANT_COMPANION_SUFFIXES = ("_quantized", "_scale", "_zero_point")


def is_quant_companion(name: str) -> bool:
    """True for a quantizer-produced companion of a weight (packed payload / scale / zero-point).

    ``requires_grad`` is matched by **substring**, so a trainable ``…lora_B.lora.weight`` also matches
    ``…lora_B.lora.weight_quantized`` and its ``_scale``/``_zero_point``. Those are not differentiable —
    ORT rejects the whole artifact generation with *"Cannot compute the partial derivative for
    '…weight_quantized' as it's unreachable from the output node(s)"*, so a quantized PEFT export could
    not produce training artifacts at all.
    """
    return name.endswith(QUANT_COMPANION_SUFFIXES)


def assert_every_requested_tensor_is_trainable(
    requested: Sequence[str], realized: Sequence[str], frozen: Sequence[str]
) -> None:
    """Fail closed when a tensor the export declared trainable did not survive as trainable.

    Compares **sets**, not counts: a count can coincide while the wrong tensors are frozen. The error
    names the lost tensors and says *why* they were lost, because this failure mode is otherwise
    completely invisible — every structural assertion passes and the loss still falls.

    :param requested: parameter names from ``training_config.json``'s ``requires_grad``.
    :param realized: graph initializers that will actually be handed to ``generate_artifacts``.
    :param frozen: graph initializers that will be frozen (used only to explain a loss).
    """
    lost: list[tuple[str, bool]] = []
    for name in requested:
        if any(name in realized_name for realized_name in realized):
            continue
        quantized_away = any(name in f and is_quant_companion(f) for f in frozen)
        lost.append((name, quantized_away))

    if not lost:
        return

    quantized = [name for name, was_quantized in lost if was_quantized]
    missing = [name for name, was_quantized in lost if not was_quantized]
    detail: list[str] = []
    if quantized:
        detail.append(
            f"{len(quantized)} were QUANTIZED and so cannot take a gradient (e.g. {quantized[0]}) — "
            "exclude them from quantization: a tensor declared trainable must never be quantized"
        )
    if missing:
        detail.append(f"{len(missing)} are absent from the graph entirely (e.g. {missing[0]})")

    raise ExportError(
        f"{len(lost)} of {len(requested)} tensors declared trainable in training_config.json are not "
        "trainable in the graph: " + "; ".join(detail)
    )


__all__ = [
    "QUANT_COMPANION_SUFFIXES",
    "is_quant_companion",
    "assert_every_requested_tensor_is_trainable",
]
