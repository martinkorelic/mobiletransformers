"""Unit tests for the export discovery + frontend registries (plan #7).

Pure tests (task selection, frontend registry) run in any profile. Discovery tests need optimum
(metadata-only lookups, no network) and skip when it is absent."""

from __future__ import annotations

import importlib.util

import pytest

from mobiletransformers.config.constants import ExportFrontend
from mobiletransformers.exceptions import ExportError, UnsupportedModelError
from mobiletransformers.export.registry import (
    EXPORT_FRONTEND_REGISTRY,
    choose_task,
    is_supported,
    resolve_frontend,
    supported_onnx_tasks,
)

_HAS_OPTIMUM = importlib.util.find_spec("optimum") is not None
requires_optimum = pytest.mark.skipif(not _HAS_OPTIMUM, reason="needs export profile (optimum)")


# --- task selection (pure) ------------------------------------------------------------------------
def test_choose_task_prefers_with_past() -> None:
    supported = ["feature-extraction", "text-generation", "text-generation-with-past"]
    assert choose_task(supported) == "text-generation-with-past"


def test_choose_task_falls_back_to_text_generation() -> None:
    assert choose_task(["feature-extraction", "text-generation"]) == "text-generation"


def test_choose_task_feature_extraction_only() -> None:
    assert choose_task(["feature-extraction"]) == "feature-extraction"


def test_choose_task_override_wins_even_outside_auto_order() -> None:
    assert choose_task(["text-generation-with-past"], override="feature-extraction") == "feature-extraction"


def test_choose_task_none_supported_fails_closed() -> None:
    with pytest.raises(UnsupportedModelError):
        choose_task([])


# --- frontend registry (pure, table lookup — not if/elif) -----------------------------------------
def test_resolve_frontend_by_enum_and_wire_value() -> None:
    assert resolve_frontend(ExportFrontend.OPTIMUM_ONNX).frontend is ExportFrontend.OPTIMUM_ONNX
    assert resolve_frontend("optimum-onnx").frontend is ExportFrontend.OPTIMUM_ONNX
    assert resolve_frontend("torch.onnx").frontend is ExportFrontend.TORCH_ONNX


def test_resolve_frontend_unknown_fails_closed() -> None:
    with pytest.raises(ExportError):
        resolve_frontend("tensorrt")


def test_frontend_registry_capabilities() -> None:
    assert set(EXPORT_FRONTEND_REGISTRY) == {ExportFrontend.OPTIMUM_ONNX, ExportFrontend.TORCH_ONNX}
    assert "inference" in EXPORT_FRONTEND_REGISTRY[ExportFrontend.OPTIMUM_ONNX].capabilities
    assert "training" in EXPORT_FRONTEND_REGISTRY[ExportFrontend.TORCH_ONNX].capabilities


# --- discovery (needs optimum, no network) --------------------------------------------------------
@requires_optimum
@pytest.mark.parametrize("model_type", ["llama", "phi3", "qwen2"])
def test_discovery_supported_model_types(model_type: str) -> None:
    tasks = supported_onnx_tasks(model_type)
    assert tasks, f"{model_type} should have ONNX tasks"
    assert "text-generation-with-past" in tasks
    assert is_supported(model_type)


@requires_optimum
def test_discovery_unknown_model_type_is_empty_not_raising() -> None:
    assert supported_onnx_tasks("totally-unknown-xyz") == ()
    assert not is_supported("totally-unknown-xyz")
