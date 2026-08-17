"""#20 support matrix: inheritance, detection (mocked), probe merge, JSON shape, filtered docs, CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace

from mobiletransformers.support.matrix import build_matrix, detect_candidate, evaluate_statuses
from mobiletransformers.support.models import CandidateEntry
from mobiletransformers.support.statuses import (
    STATUS_ORDER,
    USER_FACING_STATUSES,
    apply_inheritance,
    first_blocked,
)

# --- inheritance ------------------------------------------------------------


def test_apply_inheritance_zeros_after_first_false():
    got = apply_inheritance(
        {
            "optimum_exportable": True,
            "mobile_package_exportable": True,
            "train_artifacts_exportable": False,
            "android_inference_ready": True,  # must be forced false
            "rag_ready": True,
        }
    )
    assert got["train_artifacts_exportable"] is False
    assert got["android_inference_ready"] is False
    assert got["android_training_ready"] is False
    assert got["rag_ready"] is False
    assert list(got.keys()) == list(STATUS_ORDER)


def test_first_blocked():
    assert first_blocked({k: True for k in STATUS_ORDER}) is None
    assert first_blocked({"optimum_exportable": False}) == "optimum_exportable"


# --- detection (mocked, no network) ----------------------------------------


def _loader(model_type, architectures):
    return lambda mid, trc: SimpleNamespace(model_type=model_type, architectures=architectures)


def test_detect_supported_llama():
    entry = detect_candidate(
        "org/llama",
        config_loader=_loader("llama", ["LlamaForCausalLM"]),
        tasks_lookup=lambda mt: ["text-generation", "text-generation-with-past"],
        versions={"optimumOnnxVersion": "0.1.0", "transformersVersion": "4.46.2"},
    )
    assert entry.selected_task == "text-generation-with-past"
    assert entry.mars_target_modules_known is True  # LlamaForCausalLM is in the registry


def test_detect_unsupported_task_sets_no_selected():
    entry = detect_candidate(
        "org/weird",
        config_loader=_loader("weird", ["WeirdModel"]),
        tasks_lookup=lambda mt: [],
        versions={},
    )
    assert entry.selected_task is None
    assert entry.mars_target_modules_known is False


# --- status evaluation + probe merge ---------------------------------------


def test_evaluate_no_probe_leaves_ready_false_with_blocker():
    entry = CandidateEntry(
        model_id="org/llama",
        architectures=("LlamaForCausalLM",),
        selected_task="text-generation-with-past",
        mars_target_modules_known=True,
    )
    evaluate_statuses(entry, probe=None)
    assert entry.statuses["train_artifacts_exportable"] is True
    assert entry.statuses["android_inference_ready"] is False
    assert any("android probe" in b for b in entry.blockers)


def test_evaluate_with_full_probe_promotes_ready():
    entry = CandidateEntry(
        model_id="org/llama",
        architectures=("LlamaForCausalLM",),
        selected_task="text-generation-with-past",
        mars_target_modules_known=True,
    )
    evaluate_statuses(entry, probe={"inferenceOk": True, "trainStepOk": True, "mergeOk": True, "ragOk": True})
    assert all(entry.statuses[s] for s in STATUS_ORDER)
    assert entry.blockers == []


def test_unexportable_forces_everything_false():
    entry = CandidateEntry(model_id="org/x", selected_task=None)
    evaluate_statuses(entry, probe={"inferenceOk": True})
    assert not any(entry.statuses.values())
    assert first_blocked(entry.statuses) == "optimum_exportable"


# --- build_matrix + JSON shape + filtered docs -----------------------------


def test_build_matrix_json_shape_and_filtered(tmp_path):
    probes = tmp_path / "android_probes.json"
    probes.write_text(
        json.dumps({"org/llama": {"inferenceOk": True, "trainStepOk": True, "mergeOk": True, "ragOk": False}})
    )
    matrix = build_matrix(
        [{"modelId": "org/llama"}, {"modelId": "org/x"}],
        probes_path=probes,
        generated_at="2026-07-14T00:00:00Z",
        config_loader=lambda mid, trc: SimpleNamespace(
            model_type="llama" if mid == "org/llama" else "weird",
            architectures=["LlamaForCausalLM"] if mid == "org/llama" else ["WeirdModel"],
        ),
        tasks_lookup=lambda mt: ["text-generation-with-past"] if mt == "llama" else [],
        versions={"optimumOnnxVersion": "0.1.0", "transformersVersion": "4.46.2"},
    )
    d = matrix.to_dict()
    assert d["statusOrder"] == list(STATUS_ORDER)
    assert {m["modelId"] for m in d["models"]} == {"org/llama", "org/x"}
    llama = next(m for m in d["models"] if m["modelId"] == "org/llama")
    assert llama["statuses"]["android_inference_ready"] is True
    assert llama["statuses"]["rag_ready"] is False  # ragOk was false
    # filtered docs: org/x (no user-facing true) dropped; contributor-only statuses stripped.
    filt = matrix.filtered_docs_dict()
    assert {m["modelId"] for m in filt["models"]} == {"org/llama"}
    assert set(filt["models"][0]["statuses"]) == set(USER_FACING_STATUSES)


def test_support_matrix_cli_smoke(tmp_path, monkeypatch):
    # Drive the CLI with an injected build via a candidates file + mocked detection through monkeypatch.
    import mobiletransformers.support.matrix as mtx

    monkeypatch.setattr(
        mtx,
        "_default_config_loader",
        lambda mid, trc: SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"]),
    )
    monkeypatch.setattr(mtx, "_default_tasks_lookup", lambda mt: ["text-generation-with-past"])
    monkeypatch.setattr(
        mtx, "_default_versions", lambda: {"optimumOnnxVersion": "0.1.0", "transformersVersion": "4.46.2"}
    )

    from mobiletransformers.cli.main import main

    cands = tmp_path / "cands.json"
    cands.write_text(json.dumps(["org/llama"]))
    out = tmp_path / "matrix.json"
    code = main(["support-matrix", "--candidates", str(cands), "--out", str(out)])
    assert code == 0
    written = json.loads(out.read_text())
    assert written["models"][0]["modelId"] == "org/llama"
