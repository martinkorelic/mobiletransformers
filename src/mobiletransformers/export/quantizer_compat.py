"""One import that works across ONNX Runtime's rename of the weight-only MatMul quantizer.

## What this unblocks

`inference/builder.py` — the 3,441-line inference-graph builder, the largest file in the repo and the
last unmigrated one — was recorded for months as "unimportable under every declared profile", which
blocked Migration S6 *and* the rewrite of its 14-branch architecture ladder onto the registry.

The real blocker turned out to be **one line**:

```python
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer, QuantFormat
```

Everything else it needs from `onnxruntime.quantization` (`QuantFormat`, `QuantType`,
`quantize_dynamic`, `quantize_static`, `ONNXQuantizer`, `QuantizationMode`) resolves fine on a current
ORT. ONNX Runtime generalised the 4-bit quantizer to N-bit and **renamed both the module and the
class**, deleting the old names outright rather than leaving a deprecation alias
(`matmul_4bits_quantizer.py` is a 404 on `microsoft/onnxruntime@main`):

| old (<= the pinned era) | new (ORT 1.27 verified) |
| --- | --- |
| `onnxruntime.quantization.matmul_4bits_quantizer` | `onnxruntime.quantization.matmul_nbits_quantizer` |
| `MatMul4BitsQuantizer` | `MatMulNBitsQuantizer` |

The constructor is call-compatible for our use: every keyword the builder passes (`model`,
`block_size`, `is_symmetric`, `accuracy_level`, `nodes_to_exclude`, `quant_format`,
`op_types_to_quantize`) exists on the new class, which additionally takes `bits: int = 4` — so the
default already means 4-bit and the old behaviour is preserved without passing it.

## Why a resolver rather than just editing the import

Both spellings are live in the wild: the repo pins two ORT lines (1.24.3 and 1.27.0 under different
resolution markers) and the source-built training wheel provides its own. Hard-coding either name
re-breaks the other. This resolves at call time and fails with a message naming both spellings and the
installed version, instead of a bare `ModuleNotFoundError` that reads like a missing dependency.
"""

from __future__ import annotations

from typing import Any

#: Newest first — the name a current ORT actually ships.
_CANDIDATES = (
    ("onnxruntime.quantization.matmul_nbits_quantizer", "MatMulNBitsQuantizer"),
    ("onnxruntime.quantization.matmul_4bits_quantizer", "MatMul4BitsQuantizer"),
)


def load_weight_only_matmul_quantizer() -> Any:
    """Return ORT's weight-only MatMul quantizer class, whatever this ORT calls it.

    Raises:
        ImportError: naming both spellings and the installed ONNX Runtime version, so the failure is
            actionable instead of looking like onnxruntime is missing entirely.
    """
    import importlib

    tried: list[str] = []
    for module_path, attr in _CANDIDATES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            tried.append(f"{module_path} (module not found)")
            continue
        candidate = getattr(module, attr, None)
        if candidate is not None:
            return candidate
        tried.append(f"{module_path}.{attr} (module present, class absent)")

    try:
        import onnxruntime

        version = getattr(onnxruntime, "__version__", "unknown")
    except ImportError:
        raise ImportError(
            "onnxruntime is not installed. The inference-graph builder needs it for int4 "
            "weight-only quantization — install the `export` extra."
        ) from None

    raise ImportError(
        "could not locate ONNX Runtime's weight-only MatMul quantizer in onnxruntime "
        f"{version}. Tried:\n  " + "\n  ".join(tried) + "\n"
        "ORT renamed MatMul4BitsQuantizer -> MatMulNBitsQuantizer (and the module "
        "matmul_4bits_quantizer -> matmul_nbits_quantizer) when the quantizer was generalised to "
        "N-bit. If a newer ORT renamed it again, add the new spelling to _CANDIDATES in "
        "mobiletransformers/export/quantizer_compat.py — newest first."
    )


def load_quant_format() -> Any:
    """Return ``QuantFormat``. Re-exported by both quantizer modules and by the package root."""
    from onnxruntime.quantization import QuantFormat

    return QuantFormat
