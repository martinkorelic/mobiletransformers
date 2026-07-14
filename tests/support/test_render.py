"""#31 compatibility-matrix renderer + committed-doc drift guard (core env; no torch/optimum)."""

from __future__ import annotations

from pathlib import Path

from mobiletransformers.config.constants import MergerVariant, PEFTMethod, QuantizationType
from mobiletransformers.support.render import render_matrix_markdown
from tests.fixtures.gen_compat_matrix_doc import build_sample_matrix


def _docs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "COMPATIBILITY_MATRIX.md"


def test_render_has_headers_axes_and_rows():
    md = render_matrix_markdown(build_sample_matrix())
    assert md.startswith("# Compatibility Matrix")
    # Axes enumerated from the enums (no drift): every enum value appears in the legend.
    for member in (*PEFTMethod, *QuantizationType, *MergerVariant):
        assert member.value in md
    # Every sample model + its blocker/evidence renders.
    assert "HuggingFaceTB/SmolLM2-135M" in md
    assert "MARS/PEFT target modules not verified for this architecture" in md
    assert "no android probe recorded for android_inference_ready" in md


def test_committed_doc_matches_render():
    # F6: the doc is rendered, not hand-edited. Re-render the sample and assert the committed copy
    # matches, so an edit to the renderer/enums that changes the doc fails CI until regenerated
    # (`python tests/fixtures/gen_compat_matrix_doc.py`).
    expected = render_matrix_markdown(build_sample_matrix())
    committed = _docs_path().read_text(encoding="utf-8")
    assert committed == expected, "COMPATIBILITY_MATRIX.md is stale; regenerate with gen_compat_matrix_doc.py"
