"""Render a :class:`SupportMatrix` into ``docs/COMPATIBILITY_MATRIX.md`` (#31, F6).

The matrix (``model_support_matrix.json``) is the generated source of truth; the docs page is
**rendered from it**, never hand-maintained. Axis legends are enumerated from the #6 enums/registries so
they cannot drift from the code. A row's "evidence" is its recorded blockers (or ✅ when fully ready).
"""

from __future__ import annotations

from mobiletransformers.config.constants import MergerVariant, PEFTMethod, QuantizationType
from mobiletransformers.support.models import SupportMatrix
from mobiletransformers.support.statuses import STATUS_ORDER

#: Canonical engines: native is the guaranteed path, genai is opt-in.
_ENGINES = ("native", "genai")

_STATUS_HEADERS = {
    "optimum_exportable": "Optimum export",
    "mobile_package_exportable": "Package",
    "train_artifacts_exportable": "Train artifacts",
    "android_inference_ready": "Android inference",
    "android_training_ready": "Android training",
    "rag_ready": "RAG",
}


def _tick(value: bool) -> str:
    return "✅" if value else "❌"


def _legend() -> list[str]:
    lines = [
        "## Axes (enumerated from the registries/enums — not hand-maintained)",
        "",
        f"- **PEFT method:** {', '.join(m.value for m in PEFTMethod)}",
        f"- **Quantization:** {', '.join(q.value for q in QuantizationType)}",
        f"- **Merger variant:** {', '.join(v.value for v in MergerVariant)}",
        f"- **Engine:** {', '.join(_ENGINES)} (native is the guaranteed path; genai is opt-in)",
        "- **Status pipeline (each implies all earlier ones):** "
        + " → ".join(_STATUS_HEADERS[s] for s in STATUS_ORDER),
        "",
    ]
    return lines


def render_matrix_markdown(matrix: SupportMatrix) -> str:
    """Render `matrix` into the COMPATIBILITY_MATRIX.md markdown body (deterministic)."""
    out: list[str] = []
    out.append("# Compatibility Matrix")
    out.append("")
    out.append(
        "> **Generated** — rendered from `model_support_matrix.json`. Do not hand-edit. "
        "Regenerate with `mobiletransformers support-matrix --md docs/COMPATIBILITY_MATRIX.md` "
        "under the `export` profile (live detection needs transformers + optimum)."
    )
    out.append("")
    stamp = matrix.generated_at or "(unstamped)"
    tool = matrix.toolchain or {}
    tool_str = ", ".join(f"{k}={v}" for k, v in sorted(tool.items())) or "n/a"
    out.append(f"- Generated at: `{stamp}`")
    out.append(f"- Toolchain: {tool_str}")
    out.append("")
    out.extend(_legend())

    headers = ["Model", "Type", "Task", *(_STATUS_HEADERS[s] for s in STATUS_ORDER), "Evidence / blockers"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for entry in matrix.models:
        row = entry.to_row()
        statuses = row["statuses"]
        cells = [
            f"`{row['modelId']}`",
            row["modelType"] or "?",
            row["selectedTask"] or "—",
            *(_tick(bool(statuses.get(s, False))) for s in STATUS_ORDER),
            "; ".join(row["blockers"]) if row["blockers"] else "fully ready",
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out) + "\n"


__all__ = ["render_matrix_markdown"]
