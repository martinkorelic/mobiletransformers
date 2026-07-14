"""Generate docs/COMPATIBILITY_MATRIX.md from a representative offline SupportMatrix (#31, F6).

The committed doc is *rendered* (never hand-edited). Live rows come from
`mobiletransformers support-matrix --md ...` under the export profile; this generator produces a
deterministic, network-free sample so the page + its drift test are reproducible in the core env.

Run: `python tests/fixtures/gen_compat_matrix_doc.py` (core env; no torch/optimum needed).
"""

from __future__ import annotations

from pathlib import Path

from mobiletransformers.support.matrix import evaluate_statuses
from mobiletransformers.support.models import CandidateEntry, SupportMatrix
from mobiletransformers.support.render import render_matrix_markdown

#: Fixed stamp/toolchain so the rendered doc is byte-deterministic across runs.
_GENERATED_AT = "2026-07-14T00:00:00Z"
_TOOLCHAIN = {"optimumOnnxVersion": "0.1.0", "transformersVersion": "4.46.2"}


def build_sample_matrix() -> SupportMatrix:
    """A representative, network-free matrix: two causal-LM rows (exportable, no device probe yet) and
    one encoder row whose MARS target modules aren't verified (train-artifacts blocked)."""
    rows = [
        CandidateEntry(
            model_id="HuggingFaceTB/SmolLM2-135M",
            model_type="llama",
            architectures=("LlamaForCausalLM",),
            selected_task="text-generation-with-past",
            supported_tasks=("text-generation", "text-generation-with-past", "feature-extraction"),
            mars_target_modules_known=True,
        ),
        CandidateEntry(
            model_id="Qwen/Qwen2-0.5B",
            model_type="qwen2",
            architectures=("Qwen2ForCausalLM",),
            selected_task="text-generation-with-past",
            supported_tasks=("text-generation", "text-generation-with-past"),
            mars_target_modules_known=True,
        ),
        CandidateEntry(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_type="bert",
            architectures=("BertModel",),
            selected_task="feature-extraction",
            supported_tasks=("feature-extraction",),
            mars_target_modules_known=False,
        ),
    ]
    for entry in rows:
        entry.optimum_onnx_version = _TOOLCHAIN["optimumOnnxVersion"]
        entry.transformers_version = _TOOLCHAIN["transformersVersion"]
        evaluate_statuses(entry, probe=None)  # no device probe -> android/rag honestly false
    return SupportMatrix(models=rows, generated_at=_GENERATED_AT, toolchain=dict(_TOOLCHAIN))


def _docs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "COMPATIBILITY_MATRIX.md"


def main() -> None:
    doc = _docs_path()
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render_matrix_markdown(build_sample_matrix()), encoding="utf-8")
    print(f"wrote {doc}")


if __name__ == "__main__":
    main()
