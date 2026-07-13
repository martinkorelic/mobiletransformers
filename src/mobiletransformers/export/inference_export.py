"""Inference export front door: HF model id -> normalized device-ready ONNX package.

``export_inference`` is the single entry point. It discovers the ONNX task (``registry``), selects the
export frontend (default ``optimum-onnx``'s ``main_export``), runs it, normalizes the output into repo
conventions (``normalize``), and returns an :class:`ExportResult` carrying the exact toolchain versions
(recorded in the package manifest + support matrix downstream).

Optimum/torch imports are lazy so the module imports cleanly in the core env; the actual export runs
only in the ``export`` profile (public ``onnxruntime`` + ``optimum-onnx[onnxruntime]``).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from mobiletransformers.config.constants import ExportFrontend
from mobiletransformers.config.settings import get_settings
from mobiletransformers.exceptions import ExportError, UnsupportedModelError
from mobiletransformers.export.registry import choose_task, discover_tasks, resolve_frontend
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)


def _pkg_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def optimum_available() -> bool:
    """Availability probe for the ``optimum-onnx`` frontend (needs optimum + torch)."""
    return importlib.util.find_spec("optimum") is not None and importlib.util.find_spec("torch") is not None


@dataclass
class ExportResult:
    """Everything a caller/manifest needs about one export."""

    out_dir: Path
    model_id: str
    model_type: str
    task: str
    opset: int
    frontend: str
    optimum_version: str | None
    optimum_onnx_version: str | None
    transformers_version: str | None
    onnx_model: Path
    external_data: Path | None
    io_inputs: tuple[str, ...] = ()
    io_outputs: tuple[str, ...] = ()
    kv_layers: int = 0
    tokenizer_files: tuple[str, ...] = ()
    generation_config: bool = False
    trust_remote_code: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


def optimum_onnx_export(
    model_id: str,
    out_dir: Path,
    task: str,
    opset: int,
    trust_remote_code: bool,
    token: str | None,
) -> dict[str, str]:
    """Run Optimum's durable ``main_export`` (CLI equivalent: ``optimum-cli export onnx``).

    Returns the toolchain-version metadata to fold into the package manifest / support matrix.
    """
    from optimum.exporters.onnx import main_export

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("optimum main_export: model=%s task=%s opset=%s", model_id, task, opset)
    main_export(
        model_name_or_path=model_id,
        output=str(out_dir),
        task=task,
        opset=opset,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    return {
        "exporter": "optimum-onnx.main_export",
        "optimum_version": _pkg_version("optimum") or "",
        "optimum_onnx_version": _pkg_version("optimum-onnx") or "",
        "transformers_version": _pkg_version("transformers") or "",
        "opset": str(opset),
        "task": task,
        "trust_remote_code": str(trust_remote_code),
    }


def export_inference(
    model_id: str,
    out_dir: str | Path,
    *,
    task: str | None = None,
    opset: int = 20,
    trust_remote_code: bool = False,
    frontend: ExportFrontend | str = ExportFrontend.OPTIMUM_ONNX,
    token: str | None = None,
) -> ExportResult:
    """Export ``model_id`` to a normalized inference package under ``out_dir``.

    Discovery -> task selection -> frontend export -> normalization. Fails closed (typed error, no
    partial package left implied as valid) if the model has no ONNX exporter, the frontend is
    unavailable, or the normalized graph is missing required outputs.
    """
    out_dir = Path(out_dir)
    token = token or get_settings().hf_token

    disc = discover_tasks(model_id, token=token, trust_remote_code=trust_remote_code)
    if not disc.optimum_exportable or disc.model_type is None:
        raise UnsupportedModelError(
            f"{model_id!r} is not Optimum-exportable: {disc.blocker or 'no supported ONNX task'}"
        )

    chosen_task = choose_task(disc.supported_tasks, override=task)

    spec = resolve_frontend(frontend)
    if "inference" not in spec.capabilities:
        raise ExportError(
            f"frontend {spec.frontend.value!r} does not support inference export "
            f"(capabilities={sorted(spec.capabilities)})"
        )
    if not spec.available():
        raise ExportError(
            f"export frontend {spec.frontend.value!r} is unavailable in this environment "
            "(sync the `export` profile: `uv sync --extra export`)"
        )

    export_fn = spec.load_export()
    meta = export_fn(model_id, out_dir, chosen_task, opset, trust_remote_code, token)

    # Normalize the raw optimum output into repo conventions (lazy import: needs onnx).
    from mobiletransformers.export.normalize import normalize_package

    norm = normalize_package(out_dir, model_id=model_id, task=chosen_task)

    return ExportResult(
        out_dir=out_dir,
        model_id=model_id,
        model_type=disc.model_type,
        task=chosen_task,
        opset=opset,
        frontend=spec.frontend.value,
        optimum_version=meta.get("optimum_version") or None,
        optimum_onnx_version=meta.get("optimum_onnx_version") or None,
        transformers_version=meta.get("transformers_version") or None,
        onnx_model=norm.onnx_model,
        external_data=norm.external_data,
        io_inputs=norm.io_inputs,
        io_outputs=norm.io_outputs,
        kv_layers=norm.kv_layers,
        tokenizer_files=norm.tokenizer_files,
        generation_config=norm.generation_config,
        trust_remote_code=trust_remote_code,
        metadata=meta,
    )


__all__ = ["ExportResult", "optimum_available", "optimum_onnx_export", "export_inference"]
