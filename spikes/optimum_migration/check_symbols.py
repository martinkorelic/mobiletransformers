"""Optimum 2.1 migration spike — import-survival matrix (plan #7, step 6).

Run in the **export** profile (``uv sync --extra export``; optimum + optimum-onnx + torch):

    uv run --extra export python spikes/optimum_migration/check_symbols.py

Prints a PASS/FAIL matrix for the symbols the legacy ``trainer/builder.py`` imported. The decisive
question is whether ``OnnxConfigWithLoss`` and ``export`` survived optimum's ``optimum-onnx`` split —
tested **independently** (do not infer their survival from ``main_export`` working).

Recorded result (optimum 2.1.0 / optimum-onnx 0.1.0, 2026-07-13):
    PASS  main_export                    (optimum.exporters.onnx)
    PASS  TasksManager                   (optimum.exporters.tasks)
    PASS  *OnnxConfig model_configs      (optimum.exporters.onnx.model_configs)
    PASS  export                         (optimum.exporters.onnx  -> convert.export)   [survives]
    FAIL  OnnxConfigWithLoss             (optimum.exporters.onnx)                       [REMOVED, no replacement]

Decision: ``export()`` survives, so the training-graph path stays on Optimum's durable ``export()`` with
a **vendored** ``OnnxConfigWithLoss`` (``mobiletransformers.export.onnx_config_with_loss``) — its deps
(``OnnxConfig``, ``OnnxConfigWithPast``, ``DummyLabelsGenerator``, ``DEFAULT_DUMMY_SHAPES``) all survive.
The plan's Fallback A (torch.onnx reconstruction) is therefore NOT needed and stays a reserved,
fail-closed frontend row. Discovery gotcha: ``TasksManager``'s ONNX map is empty until
``optimum.exporters.onnx.model_configs`` is imported (decorator registration) and requires
``library_name="transformers"``.
"""

from __future__ import annotations

import importlib
import importlib.metadata


def _check(label: str, import_fn) -> bool:  # type: ignore[no-untyped-def]
    try:
        import_fn()
        print(f"PASS  {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {label}  ->  {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print("== optimum symbol-survival matrix ==")
    for dist in ("optimum", "optimum-onnx", "transformers", "torch"):
        try:
            print(f"   {dist}: {importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"   {dist}: NOT INSTALLED")

    _check("main_export            (optimum.exporters.onnx)", lambda: __import__("optimum.exporters.onnx", fromlist=["main_export"]).main_export)
    _check("TasksManager           (optimum.exporters.tasks)", lambda: __import__("optimum.exporters.tasks", fromlist=["TasksManager"]).TasksManager)
    _check("*OnnxConfig model_configs", lambda: [getattr(__import__("optimum.exporters.onnx.model_configs", fromlist=[n]), n) for n in ("LlamaOnnxConfig", "GemmaOnnxConfig", "Phi3OnnxConfig", "BertOnnxConfig", "Qwen2OnnxConfig", "OPTOnnxConfig")])
    export_ok = _check("export                 (optimum.exporters.onnx)  [at-risk]", lambda: __import__("optimum.exporters.onnx", fromlist=["export"]).export)
    ocl_ok = _check("OnnxConfigWithLoss     (optimum.exporters.onnx)  [at-risk]", lambda: __import__("optimum.exporters.onnx", fromlist=["OnnxConfigWithLoss"]).OnnxConfigWithLoss)

    print("\n== vendoring viability (deps for a self-owned OnnxConfigWithLoss) ==")
    _check("OnnxConfig + OnnxConfigWithPast", lambda: (__import__("optimum.exporters.onnx", fromlist=["OnnxConfig"]).OnnxConfig, __import__("optimum.exporters.onnx.base", fromlist=["OnnxConfigWithPast"]).OnnxConfigWithPast))
    _check("DummyLabelsGenerator + DEFAULT_DUMMY_SHAPES", lambda: (__import__("optimum.utils", fromlist=["DummyLabelsGenerator"]).DummyLabelsGenerator, __import__("optimum.utils", fromlist=["DEFAULT_DUMMY_SHAPES"]).DEFAULT_DUMMY_SHAPES))

    print("\nDecision:", "vendor OnnxConfigWithLoss on surviving export()" if (export_ok and not ocl_ok) else "re-evaluate — see docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
