"""Generate golden merger ONNX fixtures from the LEGACY ``artifact/merger.py`` ``*_2`` factories.

These goldens pin ``build_merger_model`` (``config/registry/merger.py``, #9) to the exact graphs the
legacy ``create_lora_merger_model_2`` / ``create_mars_merger_model_2`` emitted, for the full
family × quant_in × quant_out cross-product. Committing them lets the equivalence test
(``tests/unit/test_merger_builder.py``) run in the core env (onnx only) without importing the legacy
module — which #9 deletes the factories from anyway.

The legacy factory *bodies* use only onnx/numpy; the module's top-level ``import onnxruntime`` is
unused by them, so we stub it to load the module in the core env.

Regenerate with: ``python tests/fixtures/gen_merger_golden.py`` (core env, onnx only).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "merger_golden"
LEGACY_MERGER = REPO_ROOT / "artifact" / "merger.py"

#: (family, quant_in, quant_out) — matches the build_merger_model dispatch axes.
CASES = [
    ("lora", False, False),
    ("lora", False, True),
    ("lora", True, False),
    ("lora", True, True),
    ("mars", False, False),
    ("mars", False, True),
    ("mars", True, False),
    ("mars", True, True),
]


def golden_name(family: str, quant_in: bool, quant_out: bool) -> str:
    return f"{family}_{'qin' if quant_in else 'fpin'}_{'qout' if quant_out else 'fpout'}.onnx"


def _load_legacy_merger() -> types.ModuleType:
    """Load ``artifact/merger.py`` standalone, stubbing the unused onnxruntime import."""
    sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))
    spec = importlib.util.spec_from_file_location("_legacy_merger", LEGACY_MERGER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    legacy = _load_legacy_merger()
    for family, quant_in, quant_out in CASES:
        out = GOLDEN_DIR / golden_name(family, quant_in, quant_out)
        if family == "lora":
            legacy.create_lora_merger_model_2(
                str(out), quantized_inputs=quant_in, quantized_outputs=quant_out
            )
        else:
            legacy.create_mars_merger_model_2(
                str(out), quantized_inputs=quant_in, quantized_outputs=quant_out
            )
    print(f"wrote {len(CASES)} goldens to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
