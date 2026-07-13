"""Export discovery + front-door registries — the single inference-export dispatcher.

Two data-driven registries, no ``if/elif`` in business logic:

* **Task discovery** wraps Optimum's ``TasksManager`` (``discover_tasks`` / ``choose_task`` /
  ``is_supported``). It answers *which ONNX task* a model supports; it does **not** pick the
  per-architecture ``OnnxConfig`` class — that mapping is the architecture registry
  (``config.registry.architecture``). TasksManager keys on ``AutoConfig.model_type`` (``"llama"``),
  not ``architectures[0]`` (``"LlamaForCausalLM"``).
* **Export-frontend registry** (F3) selects the export *engine* as data: ``optimum-onnx`` (default,
  durable inference exporter) and ``torch.onnx`` (the manual graph path used by the training-graph
  fallback after optimum 2.1 removed ``OnnxConfigWithLoss`` — see ``spikes/optimum_migration``).
  Adding a frontend is a registry row + an :class:`ExportFrontend` enum member, never a branch.

Optimum imports are lazy (inside functions) so this module imports cleanly in the core env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mobiletransformers.config.constants import ExportFrontend
from mobiletransformers.config.registry.architecture import import_from_path
from mobiletransformers.exceptions import ExportError, UnsupportedModelError

#: Optimum exporter backend key.
ONNX_EXPORTER = "onnx"

#: Auto task-selection order. ``*-with-past`` is preferred because the inference engine needs the
#: KV-cache (past/present) graph; ``feature-extraction`` covers encoders. An explicit override wins.
TASK_PREFERENCE: tuple[str, ...] = (
    "text-generation-with-past",
    "text-generation",
    "feature-extraction",
    "sentence-similarity",
)


# --------------------------------------------------------------------------------------------------
# Task discovery (TasksManager wrapper)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskDiscovery:
    """Result of ONNX task discovery for one model. Never raises on unknown types — fail-open here,
    fail-closed at export time."""

    model_type: str | None
    supported_tasks: tuple[str, ...]
    optimum_exportable: bool
    blocker: str | None = None


def _ensure_onnx_registered() -> None:
    """Import the model-config module so its ``@register_tasks_manager_onnx`` decorators populate
    ``TasksManager``'s ONNX map. Importing ``TasksManager`` alone leaves the map empty (optimum 2.1)."""
    import optimum.exporters.onnx.model_configs  # noqa: F401  (import for its registration side effect)


def supported_onnx_tasks(model_type: str, library_name: str = "transformers") -> tuple[str, ...]:
    """Sorted ONNX-supported task names for ``model_type`` (empty tuple if the type is unknown)."""
    _ensure_onnx_registered()
    from optimum.exporters.tasks import TasksManager

    try:
        tasks = TasksManager.get_supported_tasks_for_model_type(
            model_type, ONNX_EXPORTER, library_name=library_name
        )
    except KeyError:
        return ()
    return tuple(sorted(tasks.keys()))


def is_supported(model_type: str, library_name: str = "transformers") -> bool:
    """True iff Optimum can export any ONNX task for ``model_type``."""
    return bool(supported_onnx_tasks(model_type, library_name=library_name))


def discover_tasks(
    model_id: str, *, token: str | None = None, trust_remote_code: bool = False
) -> TaskDiscovery:
    """Discover the ONNX tasks Optimum supports for ``model_id``.

    Reads ``AutoConfig.model_type`` (the TasksManager key), then looks up its supported ONNX tasks.
    An unknown/unsupported model type yields an empty set with ``optimum_exportable=False`` and a
    blocker note — it does **not** raise (fail-open discovery; export itself fails closed).
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001  (report any load failure as a discovery blocker)
        return TaskDiscovery(None, (), False, f"could not load config for {model_id!r}: {exc}")

    model_type = getattr(config, "model_type", None)
    if not model_type:
        return TaskDiscovery(None, (), False, f"{model_id!r} config has no model_type")

    tasks = supported_onnx_tasks(model_type)
    if not tasks:
        return TaskDiscovery(
            model_type, (), False, f"model_type {model_type!r} has no ONNX exporter in Optimum"
        )
    return TaskDiscovery(model_type, tasks, True, None)


def choose_task(supported_tasks: tuple[str, ...] | list[str], override: str | None = None) -> str:
    """Pick the export task.

    An explicit ``override`` is honored verbatim (it forces a task, even outside the auto order, and
    is recorded by the caller). Otherwise the first :data:`TASK_PREFERENCE` entry present in
    ``supported_tasks`` wins; if none match, fail closed.
    """
    if override is not None:
        return override
    supported = set(supported_tasks)
    for task in TASK_PREFERENCE:
        if task in supported:
            return task
    raise UnsupportedModelError(
        f"no supported task in preference order {TASK_PREFERENCE} for {sorted(supported)}"
    )


# --------------------------------------------------------------------------------------------------
# Export-frontend registry (F3)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExportFrontendSpec:
    """One export engine, declared as data. Callables are lazy dotted paths (resolved only when an
    export runs) so the registry imports cleanly in the core env."""

    frontend: ExportFrontend
    export_callable: str  # dotted path to the export function
    availability_probe: str  # dotted path to a ``() -> bool`` availability check
    capabilities: frozenset[str] = field(default_factory=frozenset)  # e.g. {"inference"}/{"training"}

    def load_export(self) -> Any:
        return import_from_path(self.export_callable)

    def available(self) -> bool:
        try:
            return bool(import_from_path(self.availability_probe)())
        except Exception:  # noqa: BLE001  (a missing/broken probe means "unavailable", never a crash)
            return False


EXPORT_FRONTEND_REGISTRY: dict[ExportFrontend, ExportFrontendSpec] = {
    ExportFrontend.OPTIMUM_ONNX: ExportFrontendSpec(
        ExportFrontend.OPTIMUM_ONNX,
        "mobiletransformers.export.inference_export.optimum_onnx_export",
        "mobiletransformers.export.inference_export.optimum_available",
        frozenset({"inference"}),
    ),
    ExportFrontend.TORCH_ONNX: ExportFrontendSpec(
        ExportFrontend.TORCH_ONNX,
        "mobiletransformers.export.torch_frontend.torch_onnx_training_export",
        "mobiletransformers.export.torch_frontend.torch_available",
        frozenset({"training"}),
    ),
}


def resolve_frontend(frontend: ExportFrontend | str) -> ExportFrontendSpec:
    """Look up a frontend spec by enum or wire value. Fail closed on any unknown key."""
    try:
        key = frontend if isinstance(frontend, ExportFrontend) else ExportFrontend(frontend)
    except ValueError as exc:
        raise ExportError(f"unknown export frontend: {frontend!r}") from exc
    spec = EXPORT_FRONTEND_REGISTRY.get(key)
    if spec is None:
        raise ExportError(f"no export frontend registered for {key!r}")
    return spec


__all__ = [
    "ONNX_EXPORTER",
    "TASK_PREFERENCE",
    "TaskDiscovery",
    "supported_onnx_tasks",
    "is_supported",
    "discover_tasks",
    "choose_task",
    "ExportFrontendSpec",
    "EXPORT_FRONTEND_REGISTRY",
    "resolve_frontend",
]
