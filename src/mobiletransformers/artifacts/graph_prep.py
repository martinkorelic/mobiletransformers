"""ONNX graph fixes a model needs before ORT can build a gradient graph from it.

Pure ``onnx`` — no ``onnxruntime`` — so it imports (and is testable) in the core profile, unlike
``artifacts/builder.py``, which pulls the training runtime at module scope.

Everything here exists because a graph that is perfectly valid for **inference** can still be
un-differentiable: ORT's gradient builders read specific forward-node outputs, and exporters omit the
optional ones nothing else consumes.
"""

from __future__ import annotations

import os

import onnx

from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)

#: Suffix for the graph rewritten to be gradient-buildable. Written beside the source model so its
#: RELATIVE external-data references keep resolving.
TRAIN_READY_SUFFIX = ".gradready.onnx"


def ensure_layernorm_grad_outputs(onnx_model_path: str | os.PathLike[str]) -> str:
    """Give ``LayerNormalization`` its optional ``Mean``/``InvStdDev`` outputs (#33).

    ORT's ``LayerNormalizationGrad`` reads the forward node's **second and third outputs** — the saved
    mean and inverse standard deviation — rather than recomputing them. ``torch.onnx`` exports the node
    with only ``Y``, so building a gradient graph through it trips an assertion deep inside ORT:

        GradientBuilderBase::O(size_t, bool) const i < node_->OutputDefs().size() was false

    which names neither the op nor the node. Finding it took bisecting the trainable set until the
    failure boundary landed between the pooler (OK) and the encoder layers (fail).

    Decoders never hit it: Llama-family RMSNorm exports as ``SimplifiedLayerNormalization``, which
    already carries 2 outputs — verified on the shipped decoder package, 61 of them, each with a
    matching ``SimplifiedLayerNormalizationGrad``. BERT-family encoders use real LayerNorm, so encoder
    support was the first thing to need this.

    Adding the outputs is safe: the ONNX spec declares them optional, ORT's kernel fills them when
    present, and nothing downstream consumes them except the gradient builder.

    Returns:
        The path to hand to ``generate_artifacts`` — the original when nothing needed rewriting, else
        a sibling file (written next to the source so relative external-data references resolve).
    """
    onnx_model_path = os.fspath(onnx_model_path)
    model = onnx.load(onnx_model_path, load_external_data=False)

    patched = 0
    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or len(node.output) >= 3:
            continue
        stem = node.name or node.output[0]
        while len(node.output) < 3:
            # Order is fixed by the spec: Y, Mean, InvStdDev.
            suffix = "_saved_mean" if len(node.output) == 1 else "_saved_inv_std"
            node.output.append(f"{stem}{suffix}")
        patched += 1

    if not patched:
        return onnx_model_path

    out_path = onnx_model_path + TRAIN_READY_SUFFIX
    onnx.save(model, out_path)
    logger.info(
        "added Mean/InvStdDev outputs to %d LayerNormalization node(s) so ORT can build their "
        "gradients; training artifacts will be generated from %s",
        patched,
        os.path.basename(out_path),
    )
    return out_path
