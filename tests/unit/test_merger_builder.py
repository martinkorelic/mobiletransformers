"""Golden-equivalence: ``build_merger_model`` reproduces the legacy ``*_2`` factory graphs (#9).

The single parameterized ``build_merger_model`` (``config/registry/merger.py``) collapses the four
``artifact/merger.py`` factories. This test pins its output — byte-for-byte, modulo doc_strings — to
committed goldens generated from the legacy ``create_*_merger_model_2`` factories
(``tests/fixtures/gen_merger_golden.py``), for the full family × quant_in × quant_out cross-product.

Structural comparison uses only ``onnx`` (a core dep), so it runs in the default env. A numerical
sanity check (validating the merge math end-to-end) is gated behind ``onnxruntime`` (export profile).
"""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest

from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.config.registry.merger import build_merger_model, resolve_merger
from tests.fixtures.gen_merger_golden import golden_name

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "merger_golden"

# (PEFTMethod, family-tag-for-golden). LoRA-family resolves to LORA/LORA_Q by quant_in; MARS -> MARS_Q.
_METHODS = [(PEFTMethod.LORA, "lora"), (PEFTMethod.MARS, "mars")]
CASES = [
    (method, family, quant_in, quant_out)
    for method, family in _METHODS
    for quant_in in (False, True)
    for quant_out in (False, True)
]


def _strip_doc_strings(model: onnx.ModelProto) -> onnx.ModelProto:
    """Clear volatile doc_string fields so the comparison is structural, not cosmetic."""
    model.doc_string = ""
    graph = model.graph
    graph.doc_string = ""
    for node in graph.node:
        node.doc_string = ""
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        value_info.doc_string = ""
    return model


def _canonical_bytes(model: onnx.ModelProto) -> bytes:
    return _strip_doc_strings(model).SerializeToString(deterministic=True)


@pytest.mark.parametrize(
    "method,family,quant_in,quant_out",
    CASES,
    ids=[f"{f}-{'qin' if qi else 'fpin'}-{'qout' if qo else 'fpout'}" for _, f, qi, qo in CASES],
)
def test_build_merger_model_matches_legacy_golden(method, family, quant_in, quant_out, tmp_path):
    spec = resolve_merger(method, quant_in=quant_in, quant_out=quant_out)
    out = tmp_path / "merger.onnx"
    build_merger_model(spec, out)

    built = onnx.load(str(out))
    onnx.checker.check_model(built)  # emitted graph is valid

    golden = onnx.load(str(GOLDEN_DIR / golden_name(family, quant_in, quant_out)))
    assert _canonical_bytes(built) == _canonical_bytes(golden), (
        f"build_merger_model output diverged from the legacy {family} "
        f"(quant_in={quant_in}, quant_out={quant_out}) golden graph"
    )


def test_lora_and_mars_use_distinct_graphs():
    """Guard the family dispatch: LoRA and MARS specs must not produce the same graph."""
    lora = resolve_merger(PEFTMethod.LORA, quant_in=True, quant_out=True)
    mars = resolve_merger(PEFTMethod.MARS, quant_in=True, quant_out=True)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        lp, mp = Path(d) / "l.onnx", Path(d) / "m.onnx"
        build_merger_model(lora, lp)
        build_merger_model(mars, mp)
        assert _canonical_bytes(onnx.load(str(lp))) != _canonical_bytes(onnx.load(str(mp)))


# ---------------------------------------------------------------------------
# Numerical sanity (export profile only — needs a runtime).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["lora", "mars"])
def test_merge_math_matches_numpy_reference(family, tmp_path):
    """Run the fp-in/fp-out graph through ORT and check merged = base + alpha*delta (numpy ref)."""
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    rng = np.random.default_rng(0)
    out_features, in_features, rank = 5, 4, 2
    method = PEFTMethod.LORA if family == "lora" else PEFTMethod.MARS
    spec = resolve_merger(method, quant_in=False, quant_out=False)
    model_path = tmp_path / "merger.onnx"
    build_merger_model(spec, model_path)

    base = rng.standard_normal((out_features, in_features)).astype(np.float32)
    alpha_val = 0.5
    alpha = np.array(alpha_val, dtype=np.float32)  # scalar graph inputs must be 0-d arrays for ORT
    adapter_B = rng.standard_normal((out_features, rank)).astype(np.float32)

    if family == "lora":
        adapter_A = rng.standard_normal((rank, in_features)).astype(np.float32)
        feeds = {"weight": base, "adapter_A": adapter_A, "adapter_B": adapter_B, "alpha": alpha}
        expected = base + alpha_val * (adapter_B @ adapter_A)
    else:
        shared_rank = 3
        n_adapters = 2
        adapter_index = 1
        shared_A = rng.standard_normal((shared_rank, in_features)).astype(np.float32)
        intermediate = rng.standard_normal((n_adapters * rank, shared_rank)).astype(np.float32)
        chunk = intermediate[adapter_index * rank : (adapter_index + 1) * rank, :]
        feeds = {
            "weight": base,
            "shared_A": shared_A,
            "intermediate": intermediate,
            "adapter_B": adapter_B,
            "adapter_index": np.array(adapter_index, dtype=np.int64),
            "rank": np.array(rank, dtype=np.int64),
            "alpha": alpha,
        }
        expected = base + alpha_val * (adapter_B @ chunk @ shared_A)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (merged,) = sess.run(["merged_weight"], feeds)
    np.testing.assert_allclose(merged, expected, rtol=1e-4, atol=1e-4)
