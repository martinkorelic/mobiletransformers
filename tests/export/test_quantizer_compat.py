"""The ORT quantizer-rename resolver that unblocked Migration S6.

`inference/builder.py` (3,441 lines, the largest file in the repo and the last unmigrated one) was
recorded as "unimportable under every declared profile". That was true, and the cause was **one import**:
ORT generalised its 4-bit weight-only MatMul quantizer to N-bit and renamed both the module and the
class, deleting the old names rather than aliasing them.

These tests run in the core env: the resolver is exercised against stub modules, so no onnxruntime is
needed to prove the resolution order and the failure message.
"""

from __future__ import annotations

import sys
import types

import pytest

from mobiletransformers.export.quantizer_compat import (
    _CANDIDATES,
    load_weight_only_matmul_quantizer,
)

NEW_MODULE, NEW_CLASS = "onnxruntime.quantization.matmul_nbits_quantizer", "MatMulNBitsQuantizer"
OLD_MODULE, OLD_CLASS = "onnxruntime.quantization.matmul_4bits_quantizer", "MatMul4BitsQuantizer"


@pytest.fixture
def fake_ort(monkeypatch):
    """Install a stub `onnxruntime.quantization` package; yields a function to add quantizer modules."""
    for name in ("onnxruntime", "onnxruntime.quantization", NEW_MODULE, OLD_MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)

    root = types.ModuleType("onnxruntime")
    root.__version__ = "9.9.9-stub"
    root.__path__ = []
    quant = types.ModuleType("onnxruntime.quantization")
    quant.__path__ = []
    monkeypatch.setitem(sys.modules, "onnxruntime", root)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quant)

    def add(module_path: str, attr: str | None):
        module = types.ModuleType(module_path)
        if attr is not None:
            setattr(module, attr, type(attr, (), {}))
        monkeypatch.setitem(sys.modules, module_path, module)
        return module

    return add


def test_prefers_the_current_ort_spelling(fake_ort):
    """When both exist, the N-bit name wins — it is the one a current ORT actually ships."""
    fake_ort(NEW_MODULE, NEW_CLASS)
    fake_ort(OLD_MODULE, OLD_CLASS)

    assert load_weight_only_matmul_quantizer().__name__ == NEW_CLASS


def test_falls_back_to_the_legacy_spelling(fake_ort):
    """The repo pins two ORT lines; hard-coding either name re-breaks the other."""
    fake_ort(OLD_MODULE, OLD_CLASS)

    assert load_weight_only_matmul_quantizer().__name__ == OLD_CLASS


def test_module_present_but_class_renamed_again_is_not_fatal(fake_ort):
    """A future ORT could keep the module and rename the class; fall through rather than crash."""
    fake_ort(NEW_MODULE, None)  # module exists, class absent
    fake_ort(OLD_MODULE, OLD_CLASS)

    assert load_weight_only_matmul_quantizer().__name__ == OLD_CLASS


def test_failure_names_both_spellings_and_the_version(fake_ort):
    """A bare ModuleNotFoundError reads like onnxruntime is missing. It is not — it is renamed."""
    fake_ort(NEW_MODULE, None)

    with pytest.raises(ImportError) as excinfo:
        load_weight_only_matmul_quantizer()

    message = str(excinfo.value)
    assert "9.9.9-stub" in message, "must name the installed ORT version"
    assert "MatMulNBitsQuantizer" in message and "MatMul4BitsQuantizer" in message
    assert "quantizer_compat.py" in message, "must say where to add a new spelling"


def test_missing_onnxruntime_says_so_plainly(monkeypatch):
    for name in ("onnxruntime", "onnxruntime.quantization", NEW_MODULE, OLD_MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)

    with pytest.raises(ImportError, match="onnxruntime is not installed"):
        load_weight_only_matmul_quantizer()


def test_candidate_order_is_newest_first():
    """Ordering is the contract: a stale name must never shadow the current one."""
    assert [m for m, _ in _CANDIDATES] == [NEW_MODULE, OLD_MODULE]
    assert [c for _, c in _CANDIDATES] == [NEW_CLASS, OLD_CLASS]
