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
    """True iff the ORT-training stack is importable (the ``ort-training-local`` profile is active).

    ``find_spec`` on a submodule imports the parent, which raises (not returns None) when ``onnxruntime``
    is absent — so guard it and treat any import failure as "not available".
    """
    import importlib.util

    try:
        return importlib.util.find_spec("onnxruntime.training") is not None
    except (ImportError, ValueError):
        return False


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
    _stamp_runtime_metadata(plan, inf / "model.onnx", result, token=token)

    # Best-effort GenAI config so both engines read one dir; dropped from features if it can't be emitted.
    if "genai" in plan.supported_engines:
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
    _write_json(
        inf / "optimum_config.json",
        {
            "modelId": plan.model_id,
            "task": result.task,
            "modelType": result.model_type,
            "optimumOnnxVersion": result.optimum_onnx_version,
            "transformersVersion": result.transformers_version,
            "trustRemoteCode": result.trust_remote_code,
            "quantization": plan.quant,
            "supportedEngines": list(plan.supported_engines),
        },
    )
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
    # optimum_hf_export picks AutoModelForCausalLM for text-generation; strip the KV-cache suffix.
    task_type = "text-generation" if plan.task.startswith("text-generation") else plan.task

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
