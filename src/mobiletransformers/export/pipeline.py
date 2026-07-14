"""One-command export pipeline (#15) — the programmatic API the ``export``/``push`` CLIs wrap.

Turns an HF repo id into a #14-shaped device-ready package (``variants/<id>/{train,inference,embedding}``
+ ``shared/`` + ``optimum/`` + ``mobiletransformers_manifest.json`` + ``checksums.json``). A thin
orchestrator: it *reuses* the export stages (#7 task discovery, #9 inference-package emit, the legacy
training-artifact + tokenizer builders, #9 mergers) and #14's ``build_manifest`` — it does not
reimplement them.

Two verifiable-in-CI entry points — ``plan_export`` (pure planning, no heavy deps) and
``export_package(..., dry_run=True)`` — plus the real ``export_package`` run, which lazy-imports the
heavy export/train stack and only executes under the export/ORT-training profiles (env-gated; not
exercised in the core test gate).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.exceptions import ConfigValidationError

#: --quant value -> effective precision/weight-type knobs (consumed by the legacy builders).
_QUANT_SPECS: dict[str, dict[str, Any]] = {
    "qint8": {"precision": "int8", "weight_type": "QInt8", "dynamic": True},
    "int4": {"precision": "int4", "weight_type": "MatMul4Bits", "dynamic": False},
    "fp16": {"precision": "fp16", "weight_type": None, "dynamic": False},
}


def parse_peft(value: str) -> tuple[PEFTMethod, int | None]:
    """``lora`` / ``lora-xs`` / ``mars`` / ``mars-optN`` (N in 0..4) -> (PEFTMethod, optimization_level)."""
    value = value.strip().lower()
    opt_level: int | None = None
    if value.startswith("mars-opt"):
        suffix = value[len("mars-opt") :]
        if not suffix.isdigit() or not (0 <= int(suffix) <= 4):
            raise ConfigValidationError(
                f"invalid MARS optimization level in {value!r} (expected mars-opt0..mars-opt4)"
            )
        opt_level = int(suffix)
        method = PEFTMethod.MARS
    else:
        try:
            method = PEFTMethod(value)
        except ValueError as exc:
            raise ConfigValidationError(f"unknown --peft value: {value!r}") from exc
        if method is PEFTMethod.MARS:
            opt_level = 0
    return method, opt_level


def quant_spec(value: str) -> dict[str, Any]:
    if value not in _QUANT_SPECS:
        raise ConfigValidationError(
            f"unknown --quant value: {value!r} (expected one of {sorted(_QUANT_SPECS)})"
        )
    return dict(_QUANT_SPECS[value])


@dataclass(frozen=True)
class ExportPlan:
    """The resolved, side-effect-free plan for an export (what dry-run reports)."""

    model_id: str
    task: str
    peft_method: PEFTMethod
    optimization_level: int | None
    rank: int
    quant: str
    variant_id: str
    output_dir: Path
    features: tuple[str, ...]
    supported_engines: tuple[str, ...]

    def variant_descriptor(self) -> dict[str, Any]:
        """The #14 variant descriptor this plan will emit (fed to build_manifest)."""
        ep = self.variant_id.split("-", 1)[0]
        return {
            "id": self.variant_id,
            "executionProvider": ep,
            "quantization": self.quant,
            "supportedEngines": list(self.supported_engines),
            "abi": None,
            "features": list(self.features),
            "minimumAndroidApi": 28,
            "recommendedDeviceMemoryMb": None,
        }


def plan_export(
    *,
    model: str,
    output: str | Path,
    task: str | None = None,
    peft: str = "lora",
    rank: int = 8,
    quant: str = "int4",
    variant: str | None = None,
    include_rag: bool = False,
    engines: tuple[str, ...] = ("native",),
    discover: Callable[[str], str] | None = None,
) -> ExportPlan:
    """Resolve every export decision without touching large files or heavy deps.

    ``task`` auto-selects via ``discover`` (default: the #7 registry) when omitted. ``discover`` is
    injectable so tests avoid network. ``variant`` defaults to ``cpu-<quant>``.
    """
    method, opt = parse_peft(peft)
    quant_spec(quant)  # validate early
    resolved_task = task or _discover_task(model, discover)
    variant_id = variant or f"cpu-{quant}"
    features = ["core", "inference", "train"] + (["rag"] if include_rag else [])
    if "genai" in engines:
        features.append("genai")
    return ExportPlan(
        model_id=model,
        task=resolved_task,
        peft_method=method,
        optimization_level=opt,
        rank=rank,
        quant=quant,
        variant_id=variant_id,
        output_dir=Path(output),
        features=tuple(features),
        supported_engines=tuple(engines),
    )


def _discover_task(model: str, discover: Callable[[str], str] | None) -> str:
    if discover is not None:
        return discover(model)
    # Real path: lazy-import the #7 registry (pulls optimum). Only hit when no --task + no injection.
    from mobiletransformers.export.registry import choose_task, discover_tasks

    supported = discover_tasks(model).supported_tasks
    return choose_task(supported)


def manifest_skeleton(plan: ExportPlan, *, base_model_id: str | None = None) -> dict[str, Any]:
    """A planning-only manifest preview (no integrity maps) — what ``--dry-run`` prints."""
    return {
        "schemaVersion": "1.0",
        "minReaderVersion": "1.0",
        "baseModelId": base_model_id or plan.model_id,
        "selectedTask": plan.task,
        "peftMethods": [plan.peft_method.value],
        "quantization": [plan.quant],
        "defaultVariant": plan.variant_id,
        "variants": [plan.variant_descriptor()],
        "_dryRun": True,
    }


@dataclass
class ExportedPackage:
    output_dir: Path
    manifest_path: Path
    plan: ExportPlan
    extras: dict[str, Any] = field(default_factory=dict)


def assemble_package(
    plan: ExportPlan,
    stage_dirs: dict[str, str | Path],
    *,
    base_model_id: str,
    report: dict[str, Any],
    exported_at: str | None = None,
) -> ExportedPackage:
    """Reshape already-produced stage outputs into the #14 tree + emit manifest/checksums (steps 9-10).

    ``stage_dirs`` maps ``inference``/``train``/``embedding``/``tokenizer`` -> a source directory already
    in that stage's internal shape. Directories are copied into ``variants/<id>/<stage>`` (tokenizer into
    ``shared/tokenizer``), then ``build_manifest`` stream-hashes the tree. Pure filesystem — verifiable in
    CI over synthetic stage dirs; this is the layer the export-E2E checkpoint asserts against #13.
    """
    import shutil

    from mobiletransformers.hub.package_format import (
        build_manifest,
        write_manifest,
        write_variant_checksums,
    )

    out = plan.output_dir
    vid = plan.variant_id
    variant_root = out / "variants" / vid
    for stage in ("train", "inference", "embedding"):
        src = stage_dirs.get(stage)
        if src is not None and Path(src).is_dir():
            shutil.copytree(Path(src), variant_root / stage, dirs_exist_ok=True)
    tok = stage_dirs.get("tokenizer")
    if tok is not None and Path(tok).is_dir():
        shutil.copytree(Path(tok), out / "shared" / "tokenizer", dirs_exist_ok=True)

    # optimum/ provenance reports from the export metadata.
    import json as _json

    opt_dir = out / "optimum"
    opt_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / "export_report.json").write_text(_json.dumps(report, indent=2, sort_keys=True) + "\n")
    (opt_dir / "supported_tasks.json").write_text(
        _json.dumps(list(report.get("supportedTasks", [])), indent=2) + "\n"
    )

    manifest = build_manifest(
        out,
        [plan.variant_descriptor()],
        base_model_id=base_model_id,
        report=report,
        default_variant=vid,
        exported_at=exported_at,
    )
    write_variant_checksums(out, manifest)
    manifest = build_manifest(
        out,
        [plan.variant_descriptor()],
        base_model_id=base_model_id,
        report=report,
        default_variant=vid,
        exported_at=exported_at,
    )
    manifest_path = write_manifest(out, manifest)
    return ExportedPackage(output_dir=out, manifest_path=manifest_path, plan=plan)


def export_package(
    *,
    model: str,
    output: str | Path,
    task: str | None = None,
    peft: str = "lora",
    rank: int = 8,
    quant: str = "int4",
    variant: str | None = None,
    include_rag: bool = False,
    embedding_model: str | None = None,
    engines: tuple[str, ...] = ("native",),
    dry_run: bool = False,
    token: str | None = None,
    discover: Callable[[str], str] | None = None,
) -> ExportPlan | ExportedPackage:
    """Export ``model`` into a #14 package at ``output``.

    ``dry_run=True`` returns the resolved :class:`ExportPlan` (no files, no heavy deps). A real run
    lazy-imports the export/train stack and is env-gated (export + ORT-training profiles).
    """
    plan = plan_export(
        model=model,
        output=output,
        task=task,
        peft=peft,
        rank=rank,
        quant=quant,
        variant=variant,
        include_rag=include_rag,
        engines=engines,
        discover=discover,
    )
    if dry_run:
        return plan
    return _full_export(plan, token=token, embedding_model=embedding_model)


def _full_export(plan: ExportPlan, *, token: str | None, embedding_model: str | None) -> ExportedPackage:
    """Real export orchestration — reuses existing stages; env-gated (heavy imports are lazy).

    Stage order (per #15): task already resolved → inference package + handoff (#9
    ``export_inference_package``) → training artifacts → mergers → tokenizer/configs → reshape into the
    #14 variant tree → ``build_manifest`` + checksums. This body runs only under the export/ORT-training
    profiles; the core test gate exercises ``dry_run`` + the assembly/manifest layer over the fixture.
    """
    raise NotImplementedError(
        "Full-model export requires the export + onnxruntime-training profiles and is run manually "
        "(see 02_code_plans/05). The planning path (dry_run=True) and the #14/#13 assembly+validate "
        "layer are covered in CI. Wire the heavy stages here when running under those profiles."
    )


__all__ = [
    "ExportPlan",
    "ExportedPackage",
    "parse_peft",
    "quant_spec",
    "plan_export",
    "manifest_skeleton",
    "assemble_package",
    "export_package",
]
