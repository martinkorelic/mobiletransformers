"""Integration test for support-matrix merge (plan #7). Pure JSON — runs in any profile."""

from __future__ import annotations

import copy

from mobiletransformers.export.support_matrix import (
    SupportRow,
    update_support_matrix,
    write_matrix,
)


def _row(model_id: str, model_type: str, ok: bool) -> SupportRow:
    return SupportRow(
        model_id=model_id,
        model_type=model_type,
        optimum_exportable=ok,
        mobile_package_exportable=ok,
        supported_tasks=("text-generation-with-past",) if ok else (),
        chosen_task="text-generation-with-past" if ok else None,
        blocker=None if ok else "no ONNX exporter",
        toolchain={"optimum": "2.1.0"} if ok else {},
    )


def test_merge_two_synthetic_results(tmp_path) -> None:
    path = tmp_path / "model_support_matrix.json"
    update_support_matrix(path, _row("org/good", "llama", True))
    matrix = update_support_matrix(path, _row("org/bad", "myst", False))

    good = matrix["models"]["org/good"]
    bad = matrix["models"]["org/bad"]
    assert good["optimum_exportable"] is True and good["mobile_package_exportable"] is True
    assert bad["optimum_exportable"] is False and bad["mobile_package_exportable"] is False
    # deferred statuses seeded None for later plans
    assert good["android_inference_ready"] is None
    assert matrix["schemaVersion"] == "1.0"


def test_merge_is_idempotent(tmp_path) -> None:
    path = tmp_path / "m.json"
    row = _row("org/good", "llama", True)
    first = copy.deepcopy(update_support_matrix(path, row))
    second = update_support_matrix(path, row)
    assert second == first


def test_merge_preserves_later_plan_status(tmp_path) -> None:
    path = tmp_path / "m.json"
    row = _row("org/good", "llama", True)
    matrix = update_support_matrix(path, row)
    # A later plan flips a deferred status it owns.
    matrix["models"]["org/good"]["android_inference_ready"] = True
    write_matrix(matrix, path)
    # Re-merging the #7 row must not clobber the later plan's value.
    remerged = update_support_matrix(path, row)
    assert remerged["models"]["org/good"]["android_inference_ready"] is True
