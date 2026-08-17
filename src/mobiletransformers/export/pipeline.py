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

from mobiletransformers.config.constants import PEFTMethod, TaskType
from mobiletransformers.config.registry.architecture import import_from_path
from mobiletransformers.config.registry.task import get_task_spec
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
    #: PEFT target modules override. Empty means "use the architecture registry's row for this model",
    #: which is the per-model source of truth and the one place to edit for a new architecture.
    peft_targets: tuple[str, ...] = ()

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
    peft_targets: tuple[str, ...] = (),
    discover: Callable[[str], str] | None = None,
) -> ExportPlan:
    """Resolve every export decision without touching large files or heavy deps.

    ``task`` auto-selects via ``discover`` (default: the #7 registry) when omitted. ``discover`` is
    injectable so tests avoid network. ``variant`` defaults to ``cpu-<quant>``.

    ``peft_targets`` overrides which modules PEFT adapts. Left empty (the normal case) the
    architecture registry's row for the model decides, so support for a new architecture is a data
    row rather than a flag every caller has to remember.
    """
    method, opt = parse_peft(peft)
    quant_spec(quant)  # validate early
    resolved_task = task or _discover_task(model, discover)
    variant_id = variant or f"cpu-{quant}"
    # The task decides whether a training stage is even possible: `feature-extraction` has no head and
    # therefore no loss, so claiming `train` produced a package advertising a stage that could never be
    # built. `_effective_features` still demotes it afterwards based on what actually landed on disk.
    task_spec = get_task_spec(resolved_task)
    features = ["core", *task_spec.stages] + (["rag"] if include_rag else [])
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
        peft_targets=tuple(peft_targets),
    )


def _discover_task(model: str, discover: Callable[[str], str] | None) -> str:
    if discover is not None:
        return discover(model)
    # Real path: lazy-import the registry (pulls optimum). Only hit when no --task + no injection.
    from mobiletransformers.config.settings import get_settings
    from mobiletransformers.exceptions import UnsupportedModelError
    from mobiletransformers.export.registry import choose_task, discover_tasks

    # The token matters here: a GATED base model 401s on its config read, and `discover_tasks` is
    # fail-open — it swallows the exception into `blocker` and returns no tasks.
    discovery = discover_tasks(model, token=get_settings().hf_token)

    # Surface that blocker. Without this the caller sees `choose_task(())` fail with "no supported
    # task in preference order (...) for []", which names neither the model nor the reason — an empty
    # list reads as "this architecture is unsupported" when the actual cause was a missing token or a
    # network error. Measured 2026-08-17 on google/gemma-3-270m-it, which cost a full export run to
    # diagnose because the real message had already been computed and thrown away.
    if not discovery.supported_tasks:
        raise UnsupportedModelError(
            f"cannot determine an export task for {model!r}: "
            f"{discovery.blocker or 'Optimum reports no ONNX task for this model type'}"
        )
    return choose_task(discovery.supported_tasks)


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
    peft_targets: tuple[str, ...] = (),
    dry_run: bool = False,
    stages: set[str] | None = None,
    token: str | None = None,
    discover: Callable[[str], str] | None = None,
) -> ExportPlan | ExportedPackage:
    """Export ``model`` into a #14 package at ``output``.

    ``dry_run=True`` returns the resolved :class:`ExportPlan` (no files, no heavy deps). A real run
    lazy-imports the export stack and builds the selected ``stages`` (default: auto by request +
    importable deps). A train-capable package is produced across two profile-scoped runs (see
    :func:`_full_export`).
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
        peft_targets=peft_targets,
        discover=discover,
    )
    if dry_run:
        return plan
    return _full_export(plan, token=token, embedding_model=embedding_model, stages=stages)


# --- real export: stage-gated orchestrator (#15) ------------------------------------------------

#: Stage name -> the manifest feature it satisfies (F1 feature→subtree contract, #13 validator).
_STAGE_FEATURE = {"inference": "inference", "training": "train", "embedding": "rag"}


@dataclass
class StageOutput:
    """What one stage builder produced: the stage_dirs to hand ``assemble_package`` + report fields.

    ``stage_dirs`` keys are the assemble keys (``inference``/``train``/``embedding``/``tokenizer``);
    ``report`` fields are merged into the manifest provenance (non-null wins).
    """

    stage_dirs: dict[str, Path] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


#: A stage builder: resolved plan + a staging dir -> its outputs. Heavy deps imported lazily inside.
StageBuilder = Callable[..., StageOutput]


def _pkg_version(dist: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _training_available() -> bool:
    """True iff the ORT-training stack is actually **usable** (``ort-training-local`` profile active).

    This performs the real import rather than probing for the module, because *present* and *usable*
    are different things here and the difference is not hypothetical: the **public** ``onnxruntime``
    wheel ships an ``onnxruntime/training/`` directory too, so ``find_spec`` returns a spec under the
    export profile — but importing it dies with

        ImportError: cannot import name 'PropagateCastOpsStrategy' from 'onnxruntime.capi._pybind_state'

    because the public build has no training pybind state. `_select_stages` then selected the training
    stage under a profile that cannot build one, and the export crashed inside ``artifacts/builder.py``
    with a traceback naming a symbol rather than the profile.

    Importing costs a second and happens once per export, against a failure mode that costs a full
    export cycle. The symbols are the ones ``artifacts/builder.py`` imports at module scope, so this
    answers the question that is actually being asked: *will that import succeed?*
    """
    try:
        from onnxruntime.training import artifacts, onnxblock  # noqa: F401
    except Exception:  # noqa: BLE001 - any failure means the stage cannot be built
        return False
    return True


def _select_stages(plan: ExportPlan, stages: set[str] | None) -> set[str]:
    """Which stages to attempt. Explicit ``stages`` wins; else auto-detect by request + importable deps."""
    if stages is not None:
        unknown = stages - set(_STAGE_FEATURE)
        if unknown:
            raise ConfigValidationError(f"unknown export stage(s): {sorted(unknown)}")
        return set(stages)
    selected = {"inference"}  # the always-required floor
    if "rag" in plan.features:
        selected.add("embedding")
    if _training_available():
        selected.add("training")
    return selected


def _base_report(plan: ExportPlan) -> dict[str, Any]:
    return {
        "mobiletransformersVersion": _pkg_version("mobiletransformers") or "0.0.0",
        "architectures": [],
        "supportedTasks": [plan.task],
        "selectedTask": plan.task,
        "trustRemoteCode": False,
        "peftMethods": [plan.peft_method.value],
        "quantization": [plan.quant],
        "androidRuntime": {"minimumAndroidApi": 28, "recommendedDeviceMemoryMb": None, "requiredAbis": []},
        "license": {"framework": None, "baseModelWeights": None, "noticeFile": None},
    }


def _effective_features(plan: ExportPlan, stage_dirs: dict[str, str | Path]) -> tuple[str, ...]:
    """Features the manifest may honestly claim: ``core`` + the stages actually present (+ ``genai`` iff
    requested AND a ``genai_config.json`` was emitted). Unions THIS run's ``stage_dirs`` with what is
    already on disk in the assembled variant tree, so a training-only re-assembly (a separate profile run)
    does not drop the ``inference``/``genai`` features produced by the earlier run. Never claim a subtree
    that isn't present — the #13 validator checks feature→path presence for train/inference/rag."""
    variant_dir = plan.output_dir / "variants" / plan.variant_id

    def present(stage: str, marker: str) -> bool:
        return stage in stage_dirs or (variant_dir / stage / marker).exists()

    feats = ["core"]
    if present("inference", "model.onnx"):
        feats.append("inference")
    if present("train", "training_config.json"):
        feats.append("train")
    if present("embedding", "rag_config.json"):
        feats.append("rag")

    inf = stage_dirs.get("inference")
    genai_config = (
        (Path(inf) / "genai_config.json").is_file()
        if inf is not None
        else (variant_dir / "inference" / "genai_config.json").is_file()
    )
    if "genai" in plan.supported_engines and genai_config:
        feats.append("genai")
    return tuple(feats)


#: Manifest provenance whose ONLY producer is the inference stage: (report key, optimum_config key).
#:
#: A train-capable package is necessarily built by two profile-scoped runs into one output dir (the
#: onnxruntime profiles cannot co-install), and the second run rebuilds the manifest from
#: ``_base_report``, which knows none of these. So `--stages training` used to publish a manifest with
#: ``transformersVersion: null`` and ``architectures: []`` — the fields that would have attributed a
#: pushed package to a transformers line, which is exactly what was needed to diagnose the 4.57 export
#: regression.
_INFERENCE_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("transformersVersion", "transformersVersion"),
    ("optimumOnnxVersion", "optimumOnnxVersion"),
    ("architectures", "modelType"),
    ("trustRemoteCode", "trustRemoteCode"),
)


def _carry_forward_inference_provenance(
    plan: ExportPlan, report: dict[str, Any], stage_dirs: dict[str, str | Path]
) -> None:
    """Recover inference-stage provenance from disk when this run did not build that stage.

    Read back from ``inference/optimum_config.json`` — which the inference stage wrote next to the graph
    it describes — rather than re-derived from the current environment: the training profile pins a
    *different* transformers than the export profile, so re-deriving would stamp the manifest with a
    version that never touched the graph. That is worse than the null it replaces.

    Only unset-shaped values are filled, so a run that DID export inference always wins.
    """
    if "inference" in stage_dirs:
        return
    config_path = plan.output_dir / "variants" / plan.variant_id / "inference" / "optimum_config.json"
    if not config_path.is_file():
        return

    import json as _json

    from mobiletransformers.utils.logging import get_logger

    try:
        recorded = _json.loads(config_path.read_text())
    except (OSError, ValueError):  # a corrupt side-car must not fail an otherwise-good export
        return

    for report_key, config_key in _INFERENCE_PROVENANCE:
        value = recorded.get(config_key)
        if value in (None, "", [], {}):
            continue
        if report_key == "architectures":
            value = [value]
        current = report.get(report_key)
        if current in (None, "", [], {}) or (report_key == "trustRemoteCode" and current is False):
            report[report_key] = value

    # The task is NOT carried over — this run built its own stage for `plan.task`, and silently
    # relabelling the package as the recorded task would hide a genuine disagreement. Say so instead:
    # the training and packaging halves resolving different rows for one model is a defect this
    # project has already paid for once.
    recorded_task = recorded.get("task")
    if recorded_task and recorded_task != report.get("selectedTask"):
        get_logger(__name__).warning(
            "export: this run's task %r differs from the task the shipped inference graph was "
            "exported for (%r, from %s). The package now describes two different tasks.",
            report.get("selectedTask"),
            recorded_task,
            config_path,
        )


def _default_builders() -> dict[str, StageBuilder]:
    return {
        "inference": _build_inference_stage,
        "training": _build_training_stage,
        "embedding": _build_embedding_stage,
    }


def _build_inference_stage(
    plan: ExportPlan, dest: Path, *, token: str | None, embedding_model: str | None
) -> StageOutput:
    """Real inference stage (``export`` profile): one dir both engines read from a single source of truth.

    Produces ``inference/`` with the normalized Native-ready ``model.onnx`` (+ ``model.onnx_data``),
    ``generation_config.json``, tokenizer files, an empty ``weight_handoff_map.json`` (all-frozen base —
    the training stage overwrites it with trainable entries), and a best-effort ``genai_config.json`` so
    the GenAI engine loads the SAME ``model.onnx``. Tokenizer is also copied to the ``tokenizer`` stage
    for ``shared/tokenizer`` + the on-device ``mobiletransformers_tokenizer_config.json``.
    """
    from mobiletransformers.artifacts.handoff_map import HandoffMap
    from mobiletransformers.export.inference_export import export_inference
    from mobiletransformers.utils.logging import get_logger

    log = get_logger(__name__)
    # What this stage writes beyond the graph — cache metadata, a GenAI decoder block — is a property
    # of the TASK, not something every package gets.
    task_spec = get_task_spec(plan.task)
    dest = Path(dest)
    inf = dest / "inference"
    tok = dest / "tokenizer"
    inf.mkdir(parents=True, exist_ok=True)
    tok.mkdir(parents=True, exist_ok=True)

    result = export_inference(plan.model_id, inf, task=plan.task, token=token)

    # All-frozen base: a valid empty handoff map (the #13 validator + build_manifest require the file to
    # exist and resolve; the training stage replaces it with the trainable-tensor entries).
    HandoffMap(entries=[]).save(inf / "weight_handoff_map.json")

    # Tokenizer stage (copied — the GenAI dir also needs tokenizer.json beside model.onnx/genai_config).
    _populate_tokenizer_stage(plan, result, inf, tok, token=token, log=log)

    # The Native engine reads its KV-cache geometry from the graph's own metadata, so this is required,
    # not best-effort — see _stamp_runtime_metadata.
    # Only for tasks that HAVE a cache. The Native engine fails closed when this metadata is missing,
    # but an encoder has no cache to size, and stamping a decoder's geometry onto one is worse than
    # omitting it.
    if task_spec.stamps_kv_metadata:
        _stamp_runtime_metadata(plan, inf / "model.onnx", result, token=token)

    # Best-effort GenAI config so both engines read one dir; dropped from features if it can't be emitted.
    if task_spec.emits_genai_config and "genai" in plan.supported_engines:
        try:
            _emit_genai_config(plan, inf, result, token=token)
        except Exception as exc:  # noqa: BLE001 - never fail the whole export on the GenAI side-car
            log.warning("genai_config.json not emitted (package will be Native-only): %s", exc)

    report = {
        "architectures": [result.model_type] if result.model_type else [],
        "supportedTasks": [result.task],
        "selectedTask": result.task,
        "optimumOnnxVersion": result.optimum_onnx_version,
        "transformersVersion": result.transformers_version,
        "trustRemoteCode": result.trust_remote_code,
    }

    # #15 DoD: record HOW this graph was produced, next to the graph. Without it a shipped package
    # cannot be traced back to the exporter/task that made it.
    #
    # `quantization` is the variant's REQUESTED setting, which drives the training stage and the
    # variant id (`cpu-<quant>`). The inference export does not quantize, so a variant named
    # `cpu-int4` ships an fp32 inference graph — a real asymmetry that nothing declared, leaving the
    # directory name as the only (wrong) signal. `inferenceGraphPrecision` is measured from the graph
    # that actually shipped, so the two can never silently diverge again.
    from mobiletransformers.artifacts.parameter_budget import describe_graph_precision

    optimum_config: dict[str, Any] = {
        "modelId": plan.model_id,
        "task": result.task,
        "modelType": result.model_type,
        "optimumOnnxVersion": result.optimum_onnx_version,
        "transformersVersion": result.transformers_version,
        "trustRemoteCode": result.trust_remote_code,
        "quantization": plan.quant,
        "inferenceGraphPrecision": describe_graph_precision(inf / "model.onnx"),
        "supportedEngines": list(plan.supported_engines),
    }

    # A classification head predicts an INDEX, and an index is not an answer. Without the label names
    # the device can run the graph and report `LABEL_3` — a number in a costume — so #33's encoder
    # support stopped one step short of being usable: a model could be fine-tuned on device and then
    # never asked anything meaningful. The names live in the source model's own config and cost a few
    # bytes to carry.
    id2label = _read_id2label(plan, token=token, log=log)
    if id2label:
        optimum_config["id2label"] = id2label

    _write_json(inf / "optimum_config.json", optimum_config)
    return StageOutput(
        stage_dirs={"inference": inf, "tokenizer": tok},
        report={k: v for k, v in report.items() if v is not None},
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Deterministic JSON write (sorted keys) so package checksums are stable."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _populate_tokenizer_stage(
    plan: ExportPlan, result: Any, inf: Path, tok: Path, *, token: str | None, log: Any
) -> None:
    """Copy the exported tokenizer files into the tokenizer stage, emit ``chat_template.jinja``, and
    the on-device ``mobiletransformers_tokenizer_config.json`` (Android Native tokenizer)."""
    import shutil

    for name in getattr(result, "tokenizer_files", ()):  # relative names under the export dir
        src = inf / name
        if src.is_file():
            shutil.copy2(src, tok / Path(name).name)

    # #15 DoD: the chat template as a standalone file. It is reshaped to shared/chat_template.jinja and
    # flattened into tokenizer/ by the installers, which is where the device conversation state reads it.
    _emit_chat_template(plan, tok, token=token, log=log)

    # Migration Map S1: this now lives in the package, so it resolves from an installed wheel too.
    from mobiletransformers.export.tokenizer_export import export_tokenizer_config

    try:
        # export_tokenizer_config appends its own `tokenizer/` under output_dir, so it takes the stage's
        # PARENT. Passing `tok` nested it one level deeper (`tokenizer/tokenizer/…`), which duplicated
        # every tokenizer file into the package and — because this is the only file carrying
        # `model.vocab_size` — left `ORTTokenizerNative.vocabSize` at 0 on device, aborting generation in
        # `greedySampling`'s `vocab_size > 0` assert.
        export_tokenizer_config(plan.model_id, output_dir=str(tok.parent), hf_token=token)
    except Exception as exc:
        from mobiletransformers.exceptions import ExportError

        raise ExportError(
            f"mobiletransformers_tokenizer_config.json could not be emitted for {plan.model_id!r}: {exc}"
        ) from exc

    # Required, not best-effort: the Native tokenizer reads vocab_size / special-token ids only from
    # this file. Its absence is not a degraded package — generation aborts on device in
    # `greedySampling`'s `vocab_size > 0` assert, long after the export reported success.
    emitted = tok / "mobiletransformers_tokenizer_config.json"
    if not emitted.is_file():
        from mobiletransformers.exceptions import ExportError

        raise ExportError(
            f"expected {emitted} after tokenizer export; the Native engine cannot load a package without it."
        )


def _emit_chat_template(plan: ExportPlan, tok: Path, *, token: str | None, log: Any) -> None:
    """Write the tokenizer's Jinja chat template to ``chat_template.jinja`` when the model has one."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(plan.model_id, token=token)
        template = getattr(tokenizer, "chat_template", None)
    except Exception as exc:  # noqa: BLE001 - a base model without a chat template is normal
        log.warning("chat_template.jinja not emitted: %s", exc)
        return
    if not template:
        log.info("%s declares no chat_template; skipping chat_template.jinja", plan.model_id)
        return
    (tok / "chat_template.jinja").write_text(template, encoding="utf-8")


#: ONNX custom-metadata keys the Android Native engine reads to size its KV cache
#: (`session_cache.h::loadModelMetadata`). Names are the contract; do not rename one side only.
RUNTIME_METADATA_KEYS = ("head_dim", "num_kv_heads", "num_layers")


def _read_id2label(plan: ExportPlan, *, token: str | None, log: Any) -> dict[str, str]:
    """The classification head's label names, keyed by stringified class index.

    A classification graph predicts an **index**, and an index is not an answer. Without the names the
    device can run the model and report ``LABEL_3``, so #33's encoder support stopped one step short of
    usable: a classifier could be fine-tuned on device and then never asked anything meaningful.

    Keys are strings because that is how HF writes them and how JSON carries them; the Kotlin reader
    (``packages/PackageTask.kt``) parses them back to ints.

    Best-effort by design. A model whose config omits ``id2label``, or a config that cannot be reached,
    must not fail an otherwise-good export — the names are a convenience for one task, and every other
    task ignores them entirely.
    """
    try:
        spec = get_task_spec(plan.task)
    except Exception:  # noqa: BLE001 - an unknown task simply has no labels to record
        return {}
    if spec.task is not TaskType.SEQUENCE_CLASSIFICATION:
        # Not a classification objective. A decoder's `id2label` is either absent or a leftover from
        # some other head, and recording it would tell the device this package classifies.
        return {}

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(plan.model_id, token=token, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 - a missing side-car must not fail the export
        log.warning("id2label not recorded (config unreadable): %s", exc)
        return {}

    mapping = getattr(cfg, "id2label", None)
    if not isinstance(mapping, dict) or not mapping:
        return {}
    return {str(index): str(name) for index, name in mapping.items()}


def _model_dims(plan: ExportPlan, result: Any, *, token: str | None) -> dict[str, int]:
    """Decoder geometry from the HF config, with the same fallbacks both consumers need.

    Single source for `genai_config.json` and the graph metadata, so the two can never disagree about
    how many layers or KV heads the exported model has.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        plan.model_id, token=token, trust_remote_code=getattr(result, "trust_remote_code", False)
    )

    def _attr(*names: str, default: Any = None) -> Any:
        for n in names:
            if getattr(cfg, n, None) is not None:
                return getattr(cfg, n)
        return default

    hidden = _attr("hidden_size", default=0)
    heads = _attr("num_attention_heads", default=0)
    return {
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_kv_heads": _attr("num_key_value_heads", "num_attention_heads", default=heads),
        "head_dim": _attr("head_dim", default=(hidden // heads if heads else 0)),
        "context_length": _attr("max_position_embeddings", "n_positions", default=2048),
        "num_layers": _attr("num_hidden_layers", "n_layer", default=result.kv_layers),
        "vocab_size": _attr("vocab_size", default=0),
        "bos_token_id": _attr("bos_token_id", default=1),
        "eos_token_id": _attr("eos_token_id", default=2),
        "pad_token_id": _attr("pad_token_id", "eos_token_id", default=0),
        "model_type": _attr("model_type", default=result.model_type or ""),
    }


def _stamp_runtime_metadata(plan: ExportPlan, model_path: Path, result: Any, *, token: str | None) -> None:
    """Write the KV-cache geometry into the graph's `metadata_props`.

    The Android Native engine sizes its KV cache purely from these three keys
    (`session_cache.h::loadModelMetadata` -> `initializeKVCache`). The legacy `inference/builder.py`
    graphs carried them; an Optimum-exported graph does not. Without them `num_layers` stays 0, no past
    key/value tensors are created, and `generateWithKVCache` then hands ORT 3 input values while
    declaring the graph's full input count — an out-of-bounds read that **segfaults on device**. Nothing
    host-side could see it: the manifest validates, both engines load, and only a real generate crashes.

    Cheap: the model is loaded with `load_external_data=False`, so only the graph proto is rewritten and
    the `model.onnx_data` blob beside it is untouched.
    """
    import onnx

    dims = _model_dims(plan, result, token=token)
    model = onnx.load(str(model_path), load_external_data=False)
    existing = {p.key for p in model.metadata_props}
    for key in RUNTIME_METADATA_KEYS:
        if key in existing:
            continue
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = str(dims[key])
    onnx.save(model, str(model_path))


def _emit_genai_config(plan: ExportPlan, inference_dir: Path, result: Any, *, token: str | None) -> None:
    """Emit ``genai_config.json`` into ``inference_dir`` describing the exported ``model.onnx`` so the
    GenAI engine loads the SAME graph the Native engine does.

    ``genai_config`` is model-intrinsic — dims / head & layer counts / canonical KV-IO names / special
    tokens — and the canonical IO scheme is fixed (the same names ``normalize.py`` verifies and the
    legacy ``make_genai_config`` hard-wires). We build it directly from the HF ``AutoConfig`` (+ the
    normalized ``ExportResult``), so it needs only ``transformers`` (the ``export`` profile) — not the
    vendored GenAI builder, which imports an ``onnxruntime`` symbol absent from that profile.
    """
    import json

    dims = _model_dims(plan, result, token=token)
    hidden = dims["hidden_size"]
    heads = dims["num_attention_heads"]
    kv_heads = dims["num_kv_heads"]
    head_size = dims["head_dim"]
    context = dims["context_length"]
    layers = dims["num_layers"]

    genai = {
        "model": {
            "bos_token_id": dims["bos_token_id"],
            "eos_token_id": dims["eos_token_id"],
            "pad_token_id": dims["pad_token_id"],
            "context_length": context,
            "type": dims["model_type"],
            "vocab_size": dims["vocab_size"],
            "decoder": {
                "session_options": {"provider_options": []},
                "filename": "model.onnx",
                "head_size": head_size,
                "hidden_size": hidden,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "num_hidden_layers": layers,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
            },
        },
        "search": {
            "max_length": context,
            "min_length": 0,
            "do_sample": False,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
        },
    }
    (inference_dir / "genai_config.json").write_text(json.dumps(genai, indent=2) + "\n", encoding="utf-8")


def _build_training_stage(
    plan: ExportPlan, dest: Path, *, token: str | None, embedding_model: str | None
) -> StageOutput:
    """Training stage seam (``ort-training-local`` profile). Wires ``gen_artifacts`` (→ training/eval/
    optimizer models + checkpoint + the extended ``training_config.json`` carrying ``peft_mapping``) and
    then ``export_inference_package`` for the per-tensor ``.bin`` + ``frozen_base.onnx.data`` +
    trainable ``weight_handoff_map.json`` + merger graphs into the inference dir.

    Staged: the body lands in a follow-on ``ort-training-local`` run. Selected-but-unavailable fails
    closed naming the profile.
    """
    import os

    from mobiletransformers.exceptions import ExportError
    from mobiletransformers.utils.logging import get_logger

    log = get_logger(__name__)
    if not _training_available():
        raise ExportError(
            "training stage requires the ort-training-local profile "
            "(`uv sync --python 3.12 --group ort-training-local`)"
        )

    # The inference stage (a prior `export`-profile run) must already have assembled the package; the
    # training stage reads that inference model.onnx and writes the trainable split back into it.
    inference_dir = plan.output_dir / "variants" / plan.variant_id / "inference"
    inference_onnx = inference_dir / "model.onnx"
    if not inference_onnx.is_file():
        raise ExportError(
            f"training stage requires the inference package first — {inference_onnx} not found. "
            "Run the inference export (export profile) into the same --output before --stages training."
        )

    if token:
        os.environ.setdefault("HF_TOKEN", token)

    dest = Path(dest)
    train_export = dest / "train_export"  # optimum_hf_export scratch (quant_model.onnx + training_config)
    train_stage = dest / "train"  # the assembled train/ stage dir
    train_export.mkdir(parents=True, exist_ok=True)
    train_stage.mkdir(parents=True, exist_ok=True)

    # Migration Map S2: in the package now, and core-importable (onnx only) — so it is a normal import.
    # Migration Map S4: training_export is in the package (its torch/optimum imports stay lazy inside).
    # Migration Map S5: the last legacy arrow is gone — the export path is fully in-package, so it
    # resolves from an installed wheel. Its torch/onnxruntime-training imports remain lazy inside.
    from transformers import AutoConfig

    from mobiletransformers.artifacts.builder import gen_artifacts
    from mobiletransformers.export.inference_package import export_inference_package
    from mobiletransformers.export.training_export import optimum_hf_export

    quantized = plan.quant != "fp16"
    # `get_task_spec` already strips the `-with-past` suffix (the suffix selects graph shape, not task
    # identity), so this is the registry lookup the old `plan.task.startswith("text-generation")`
    # string test was standing in for.
    task_spec = get_task_spec(plan.task)
    task_type = task_spec.task.value

    # 1) Producer of peft_mapping/requires_grad + the training graph (quant_model.onnx + training_config).
    log.info("training stage: optimum_hf_export(%s) -> %s", plan.model_id, train_export)
    optimum_hf_export(
        model_id=plan.model_id,
        model_output=str(train_export),
        training_mode=True,
        train_method=plan.peft_method.value,
        lora_rank=plan.rank,
        lora_alpha=plan.rank,
        quantize=quantized,
        task_type=task_type,
        # Empty -> the architecture registry's row decides (the per-model source of truth).
        lora_target=list(plan.peft_targets) or None,
    )

    # 2) Training artifacts (training/eval/optimizer models + checkpoint/ + extended training_config.json).
    extended_config = gen_artifacts(
        train_dir=str(train_export),
        artifact_dir=str(train_stage),
        model_name="quant_model.onnx" if quantized else "model.onnx",
        training_config={},
    )

    # 3) Overwrite the empty handoff map with the real trainable split into the assembled inference dir.
    model_config = AutoConfig.from_pretrained(plan.model_id, token=token, trust_remote_code=False)
    log.info("training stage: export_inference_package -> %s (trainable split + handoff map)", inference_dir)
    export_inference_package(
        model_path=str(inference_onnx),
        output_dir=str(inference_dir),
        training_config=extended_config,
        model_config=model_config,
        peft_method=plan.peft_method,
        quant_in=quantized,
        quant_out=quantized,
    )

    # 3b) Prove the two halves of the merge contract agree BEFORE the package ships.
    #
    # The handoff map records a `trainingBaseLayerName` per entry; the device merger turns each into a
    # checkpoint lookup. Nothing verified those lookups could succeed, so three separate name-shape
    # defects each survived export, push and install, and only surfaced as a merge that wrote nothing.
    # This is the cheap host-side check that would have caught all three (see checkpoint_names.py).
    from mobiletransformers.artifacts.checkpoint_names import verify_handoff_names_resolve

    checkpoint_path = train_stage / "checkpoint"
    if checkpoint_path.exists():
        verify_handoff_names_resolve(inference_dir / "weight_handoff_map.json", checkpoint_path)
    else:
        log.warning(
            "no checkpoint at %s — skipping the handoff-name check; the merge contract is UNVERIFIED",
            checkpoint_path,
        )

    # 3c) Prove the training graph carries the model's parameters — the check that was missing.
    #
    # The name check above is structural: it proves the merge can FIND its weights, not that the
    # weights are there. Nothing counted anything, which is how a byte/dtype arithmetic slip became a
    # recorded "two thirds of the model is missing" v1 blocker that was never true. This counts, per
    # dtype, against the source model's own parameter count. See artifacts/parameter_budget.py.
    from mobiletransformers.artifacts.parameter_budget import verify_checkpoint_parameter_budget

    training_model_path = train_stage / "training_model.onnx"
    parameter_summary = None
    if training_model_path.exists():
        parameter_summary = verify_checkpoint_parameter_budget(
            training_model_path,
            extended_config.get("source_parameter_count"),
        )
    else:
        log.warning(
            "no training model at %s — skipping the parameter-budget check; the training stage is "
            "UNVERIFIED against the source model",
            training_model_path,
        )

    # 3d) Prove the two graphs agree on numbers, not just on names.
    #
    # The budget check proves the parameters are present; this proves they are the SAME ones the
    # inference half ships, by running identical tokens through both and bounding the loss gap. It is
    # what supplies a reference for any later "the training loss looks high" question — the absence of
    # one is how a quantization-sized gap was once read as missing weights.
    parity = None
    if task_spec.parity_check is None:
        # Recorded, not skipped silently. The causal checker shifts logits[:, :-1] against
        # input_ids[:, 1:] and needs rank-3 logits; a per-sequence objective emits [batch, labels], so
        # running it there would raise and read as "the package is broken".
        log.warning(
            "no train/inference parity gate for task %r — this package ships without that check",
            task_spec.task.value,
        )
    elif training_model_path.exists() and inference_onnx.exists():
        verify_parity = import_from_path(task_spec.parity_check)
        parity = verify_parity(inference_onnx, train_stage)

    # #15 DoD: name the trainable tensors in the package itself. The count alone (in the manifest)
    # says how many; this says WHICH, so an adapter push or a federated round can be checked against
    # the package without re-deriving the PEFT mapping.
    peft_mapping = extended_config.get("peft_mapping") or {}
    _write_json(
        train_stage / "trainable_parameters.json",
        {
            "peftMethod": plan.peft_method.value,
            "rank": plan.rank,
            "trainableParameterCount": extended_config.get("trainable_parameter_count"),
            "baseLayers": sorted(peft_mapping),
            "requiresGrad": sorted(extended_config.get("requires_grad") or []),
        },
    )

    report = {
        "peftMethods": [plan.peft_method.value],
        "trainableParameterCount": extended_config.get("trainable_parameter_count"),
        "onnxRuntimeTrainingVersion": _pkg_version("onnxruntime-training"),
    }
    if parameter_summary is not None:
        # Recorded so a package can be audited without re-reading its graph — and so the fp32/quantized
        # split is visible rather than inferred from a file size (the mistake parameter_budget.py exists
        # to prevent).
        report["trainingParameterCount"] = parameter_summary.total
        report["trainingQuantizedParameterCount"] = parameter_summary.quantized
        report["sourceParameterCount"] = extended_config.get("source_parameter_count")
    if parity is not None:
        # The measured gap between the package's two halves. Recorded so the next reader of a
        # surprising training loss has a number to compare against instead of a guess.
        report["trainInferenceLossDeltaNats"] = round(parity.delta, 4)
    return StageOutput(
        stage_dirs={"train": train_stage},
        report={k: v for k, v in report.items() if v is not None},
    )


#: Default RAG encoder for ``--include-rag`` without ``--embedding-model``. 384-dim (in
#: ``DimensionRegistry.SUPPORTED_DIMENSIONS``) and small enough to ship beside the decoder.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Dimensions the on-device vector store can index. Mirrors the Kotlin `DimensionRegistry`
#: (`rag/VectorStoreRegistry.kt`) — a package whose encoder emits anything else cannot be indexed,
#: so the export fails closed rather than shipping an unusable `embedding/` subtree.
SUPPORTED_EMBEDDING_DIMENSIONS = (64, 128, 256, 384, 512, 768, 1024, 1536)

#: On-device embedding graph filename. `ORTRetriever.createEmbeddingModel` resolves
#: `<embedding>/<ORTRagConfig.onnxName>`, appending `.onnx` when absent — so the config records the
#: stem and this is the file it resolves to.
EMBEDDING_MODEL_STEM = "embedding_model"

#: Tokenizer files `ORTTokenizerNative` reads from `embedding/tokenizer/`.
_EMBEDDING_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _pooled_embedding_dimension(pooling_config: dict[str, Any]) -> int:
    """The dimension the pooled graph actually emits.

    sentence-transformers concatenates every enabled pooling mode, so the output width is the word
    embedding dimension times the number of active modes — not the word dimension itself.
    """
    word_dim = pooling_config.get("word_embedding_dimension")
    if not isinstance(word_dim, int) or word_dim <= 0:
        raise ConfigValidationError(
            f"pooling config declares no usable word_embedding_dimension: {pooling_config!r}"
        )
    modes = (
        "pooling_mode_cls_token",
        "pooling_mode_mean_tokens",
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
    )
    active = sum(1 for m in modes if pooling_config.get(m, False))
    return word_dim * active if active else word_dim


def _build_embedding_stage(
    plan: ExportPlan, dest: Path, *, token: str | None, embedding_model: str | None
) -> StageOutput:
    """Real embedding/RAG stage (``export`` profile): the encoder subtree ``ORTRetriever`` loads.

    Exports the sentence-transformer encoder through the same optimum front door the inference stage
    uses (task ``feature-extraction``), grafts the model's declared sentence-transformers pooling onto
    the graph — the device does no pooling, so an unpooled encoder would hand the vector store a
    ``[batch, seq, dim]`` tensor it cannot index — and lays the subtree out exactly as
    ``ORTRetriever.createEmbeddingModel`` reads it::

        embedding/
          embedding_model.onnx      # pooled: [batch, seq] -> [batch, dim]
          rag_config.json           # repoName / onnxName / embeddingDimension + retrieval defaults
          tokenizer/                # tokenizer.json + config + special tokens map

    Fails closed when the pooled dimension is not one the Kotlin ``DimensionRegistry`` can index: a
    package whose vectors cannot enter the store is worse than one with no RAG subtree at all, because
    the failure would only surface on device at first ingest.
    """
    import shutil

    from mobiletransformers.exceptions import ExportError
    from mobiletransformers.export.embedding_export import (
        add_pooling_to_onnx_model,
        load_pooling_config_from_hub,
    )
    from mobiletransformers.export.inference_export import export_inference
    from mobiletransformers.hub.package_format import sanitize_repo_id
    from mobiletransformers.utils.logging import get_logger

    log = get_logger(__name__)
    emb_id = embedding_model or DEFAULT_EMBEDDING_MODEL
    dest = Path(dest)
    raw = dest / "raw"
    emb = dest / "embedding"
    tok = emb / "tokenizer"
    emb.mkdir(parents=True, exist_ok=True)
    tok.mkdir(parents=True, exist_ok=True)

    log.info("embedding stage: exporting encoder %s (feature-extraction)", emb_id)
    result = export_inference(emb_id, raw, task="feature-extraction", token=token)

    # Pooling is model-declared (modules.json / 1_Pooling/config.json), never guessed.
    pooling_config = load_pooling_config_from_hub(emb_id)
    if not pooling_config:
        raise ExportError(
            f"{emb_id!r} declares no sentence-transformers pooling config; the on-device retriever "
            "cannot pool token embeddings itself. Pass --embedding-model with a sentence-transformers "
            "encoder."
        )
    dimension = _pooled_embedding_dimension(pooling_config)
    if dimension not in SUPPORTED_EMBEDDING_DIMENSIONS:
        raise ExportError(
            f"{emb_id!r} pools to {dimension} dimensions, which the on-device vector store cannot "
            f"index (supported: {list(SUPPORTED_EMBEDDING_DIMENSIONS)})."
        )

    import onnx

    # Self-contained: load external data back in and save one file, so `embedding/` is a single graph
    # the ORT embedding session opens by name (no sidecar blob to keep in step).
    graph = onnx.load(str(result.onnx_model), load_external_data=True)
    add_pooling_to_onnx_model(graph, emb_id, str(emb / f"{EMBEDDING_MODEL_STEM}.onnx"))

    for name in _EMBEDDING_TOKENIZER_FILES:
        src = raw / name
        if src.is_file():
            shutil.copy2(src, tok / name)
    missing = [n for n in ("tokenizer.json",) if not (tok / n).is_file()]
    if missing:
        raise ExportError(
            f"encoder export produced no {missing} for {emb_id!r} (needed by the device tokenizer)"
        )

    # repoName must match the on-device package directory (the sanitized BASE model id) — the retriever
    # resolves `<cacheDir>/<repoName>/embedding/`, not the encoder's own id.
    _write_json(
        emb / "rag_config.json",
        {
            "repoName": sanitize_repo_id(plan.model_id),
            "onnxName": EMBEDDING_MODEL_STEM,
            "embeddingDimension": dimension,
            "embeddingModelId": emb_id,
            "topK": 10,
            "searchType": "semantic",
            "minScore": 0.0,
            "indexingMode": "precompute",
            "maxTextLength": 1024,
            "chunkSize": 512,
            "chunkOverlap": 50,
        },
    )

    report = {
        "embeddingModel": emb_id,
        "embeddingDimension": dimension,
        "embeddingOptimumOnnxVersion": result.optimum_onnx_version,
    }
    return StageOutput(
        stage_dirs={"embedding": emb},
        report={k: v for k, v in report.items() if v is not None},
    )


def _full_export(
    plan: ExportPlan,
    *,
    token: str | None,
    embedding_model: str | None,
    stages: set[str] | None = None,
    builders: dict[str, StageBuilder] | None = None,
) -> ExportedPackage:
    """Real export orchestration — stage-gated, reuses existing stages, heavy imports lazy (#15).

    Builds each selected stage into a staging dir, computes the features actually produced, and delegates
    the #14 reshape + #13 manifest/checksums to :func:`assemble_package`. ``stages`` selects which to
    attempt (default: auto by request + importable deps — inference always, embedding iff RAG requested,
    training iff the ORT-training stack is present). ``builders`` is injectable so the orchestration is
    unit-testable in the core env without the heavy export stack.

    Producing a train-capable package straddles two conflicting uv profiles, so it runs as separate
    profile-scoped invocations into the same ``output``: assemble copies with ``dirs_exist_ok`` and
    rebuilds the manifest from disk, so a later ``stages={"training"}`` run fills in ``train/`` + the
    trainable handoff map without rework.
    """
    import tempfile

    from mobiletransformers.utils.logging import get_logger

    log = get_logger(__name__)
    builders = builders or _default_builders()
    selected = _select_stages(plan, stages)
    report = _base_report(plan)

    with tempfile.TemporaryDirectory(prefix="mtf-export-") as staging_root:
        staging = Path(staging_root)
        stage_dirs: dict[str, str | Path] = {}
        for stage in ("inference", "training", "embedding"):  # deterministic order
            if stage not in selected:
                log.info(
                    "export: skipping %s stage (not selected). Add it with a %s-profile run / --stages.",
                    stage,
                    stage,
                )
                continue
            out = builders[stage](plan, staging / stage, token=token, embedding_model=embedding_model)
            stage_dirs.update(out.stage_dirs)
            report.update({k: v for k, v in out.report.items() if v is not None})

        if "inference" not in stage_dirs and "train" not in stage_dirs:
            from mobiletransformers.exceptions import ExportError

            raise ExportError("no export stage produced any output")

        import dataclasses

        # Same reasoning as _effective_features below: a stage this run did not build still exists on
        # disk, and what it recorded about itself must survive the manifest rebuild.
        _carry_forward_inference_provenance(plan, report, stage_dirs)

        eff_features = _effective_features(plan, stage_dirs)
        # Honesty: don't advertise an engine the package can't serve. genai stays only if a genai_config
        # was actually emitted (i.e. "genai" survived into effective features).
        eff_engines = tuple(e for e in plan.supported_engines if e != "genai" or "genai" in eff_features)
        effective_plan = dataclasses.replace(plan, features=eff_features, supported_engines=eff_engines)
        log.info(
            "export: assembling package with features %s engines %s",
            effective_plan.features,
            effective_plan.supported_engines,
        )
        return assemble_package(
            effective_plan,
            stage_dirs,
            base_model_id=plan.model_id,
            report=report,
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
