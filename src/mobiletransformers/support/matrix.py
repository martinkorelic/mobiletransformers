"""Support-matrix generator (#20) — detect candidates, evaluate inherited statuses, emit the matrix.

Reporting layer over #7's task discovery. Detection deps (transformers ``AutoConfig``, optimum
``TasksManager``) are injectable so the generator is testable with no network; the real loaders are
lazy-imported. The three ``android_*``/``rag`` statuses are read from a probe-results file the device/CI
instrumentation writes — absent probes leave those statuses honestly ``false`` with a blocker. The
generator never runs a device itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mobiletransformers.config.registry.architecture import resolve_architecture
from mobiletransformers.exceptions import UnsupportedModelError
from mobiletransformers.support.models import CandidateEntry, SupportMatrix
from mobiletransformers.support.statuses import SupportStatus, apply_inheritance, first_blocked

#: Task auto-select priority (kept in sync with export/registry.choose_task).
_TASK_PRIORITY = ("text-generation-with-past", "text-generation", "feature-extraction", "sentence-similarity")

ConfigLoader = Callable[[str, bool], Any]
TasksLookup = Callable[[str], list[str]]


def _default_config_loader(model_id: str, trust_remote_code: bool) -> Any:
    from transformers import AutoConfig  # lazy: transformers is an export-profile dep

    return AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def _default_tasks_lookup(model_type: str) -> list[str]:
    from optimum.exporters.tasks import TasksManager  # lazy: optimum is an export-profile dep

    return list(TasksManager.get_supported_tasks_for_model_type(model_type, "onnx"))


def _default_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    def _v(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {"optimumOnnxVersion": _v("optimum-onnx"), "transformersVersion": _v("transformers")}


def _select_task(supported: list[str], requested: str | None) -> str | None:
    if requested:
        return requested if requested in supported else None
    for task in _TASK_PRIORITY:
        if task in supported:
            return task
    return None


def _mars_target_modules_known(architectures: tuple[str, ...]) -> bool:
    if not architectures:
        return False
    try:
        spec = resolve_architecture(SimpleNamespace(architectures=list(architectures)))
    except UnsupportedModelError:
        return False
    return bool(spec.target_modules)


def detect_candidate(
    model_id: str,
    *,
    requested_task: str | None = None,
    trust_remote_code: bool = False,
    opset: int = 20,
    config_loader: ConfigLoader | None = None,
    tasks_lookup: TasksLookup | None = None,
    versions: dict[str, str | None] | None = None,
) -> CandidateEntry:
    """Detect one candidate model's export capabilities (no status inheritance applied yet)."""
    config_loader = config_loader or _default_config_loader
    tasks_lookup = tasks_lookup or _default_tasks_lookup
    versions = versions if versions is not None else _default_versions()

    config = config_loader(model_id, trust_remote_code)
    model_type = getattr(config, "model_type", None)
    architectures = tuple(getattr(config, "architectures", ()) or ())
    supported = tasks_lookup(model_type) if model_type else []
    selected = _select_task(supported, requested_task)
    return CandidateEntry(
        model_id=model_id,
        model_type=model_type,
        architectures=architectures,
        optimum_onnx_version=versions.get("optimumOnnxVersion"),
        transformers_version=versions.get("transformersVersion"),
        opset=opset,
        supported_tasks=tuple(supported),
        selected_task=selected,
        trust_remote_code=trust_remote_code,
        mars_target_modules_known=_mars_target_modules_known(architectures),
    )


def evaluate_statuses(entry: CandidateEntry, probe: dict[str, bool] | None) -> CandidateEntry:
    """Fill ``entry.statuses`` (with inheritance) + ``entry.blockers`` from detection + a probe row."""
    raw = {
        SupportStatus.OPTIMUM_EXPORTABLE.value: entry.selected_task is not None,
        # Package export is a dry-run normalization; proxied by exportability of a usable task here.
        SupportStatus.MOBILE_PACKAGE_EXPORTABLE.value: entry.selected_task is not None,
        SupportStatus.TRAIN_ARTIFACTS_EXPORTABLE.value: entry.mars_target_modules_known,
        SupportStatus.ANDROID_INFERENCE_READY.value: bool(probe and probe.get("inferenceOk")),
        SupportStatus.ANDROID_TRAINING_READY.value: bool(
            probe and probe.get("trainStepOk") and probe.get("mergeOk")
        ),
        SupportStatus.RAG_READY.value: bool(probe and probe.get("ragOk")),
    }
    entry.statuses = apply_inheritance(raw)
    blockers: list[str] = []
    blocked = first_blocked(entry.statuses)
    if blocked == SupportStatus.OPTIMUM_EXPORTABLE.value:
        blockers.append("no supported ONNX task for this model type")
    elif blocked == SupportStatus.TRAIN_ARTIFACTS_EXPORTABLE.value:
        blockers.append("MARS/PEFT target modules not verified for this architecture")
    elif blocked in {
        SupportStatus.ANDROID_INFERENCE_READY.value,
        SupportStatus.ANDROID_TRAINING_READY.value,
        SupportStatus.RAG_READY.value,
    }:
        blockers.append(f"no android probe recorded for {blocked}")
    entry.blockers = blockers
    return entry


def ingest_probes(path: str | Path | None) -> dict[str, dict[str, bool]]:
    """Read ``android_probes.json`` (``{modelId: {inferenceOk, trainStepOk, mergeOk, ragOk}}``).

    Missing file -> empty (all ready statuses fall to ``false`` with a blocker, honestly)."""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_matrix(
    candidates: list[dict[str, Any] | str],
    *,
    probes_path: str | Path | None = None,
    generated_at: str | None = None,
    config_loader: ConfigLoader | None = None,
    tasks_lookup: TasksLookup | None = None,
    versions: dict[str, str | None] | None = None,
) -> SupportMatrix:
    """Detect + evaluate every candidate into a :class:`SupportMatrix`.

    ``candidates`` items are either a model-id string or a dict
    ``{modelId, task?, trustRemoteCode?, opset?}``.
    """
    probes = ingest_probes(probes_path)
    versions = versions if versions is not None else _default_versions()
    entries: list[CandidateEntry] = []
    for cand in candidates:
        spec = {"modelId": cand} if isinstance(cand, str) else dict(cand)
        entry = detect_candidate(
            spec["modelId"],
            requested_task=spec.get("task"),
            trust_remote_code=bool(spec.get("trustRemoteCode", False)),
            opset=int(spec.get("opset", 20)),
            config_loader=config_loader,
            tasks_lookup=tasks_lookup,
            versions=versions,
        )
        evaluate_statuses(entry, probes.get(entry.model_id))
        entries.append(entry)
    return SupportMatrix(models=entries, generated_at=generated_at, toolchain=dict(versions))


def write_matrix(matrix: SupportMatrix, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(matrix.to_json(), encoding="utf-8")
    return path


def write_filtered_docs(matrix: SupportMatrix, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(matrix.filtered_docs_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "detect_candidate",
    "evaluate_statuses",
    "ingest_probes",
    "build_matrix",
    "write_matrix",
    "write_filtered_docs",
]
