"""genai_config.json must stay parseable by the bundled onnxruntime-genai.

An unsupported `session_options` key is not ignored by GenAI — it is a hard parse error that rejects the
whole config. The package then still "works" because `ModelRuntimeFactory` falls back to Native, so the
damage is invisible on device: the dual-engine parity test compares Native with Native and passes.

That is exactly what shipped. `config_entries` (documented by onnxruntime.ai, rejected by 0.14.1) was
injected into every package, pointing at a *host* path that would not have existed on a device anyway.

These tests pin both halves so it cannot recur.
"""

from __future__ import annotations

import json

import pytest

from mobiletransformers.export.inference_package import (
    GENAI_CONFIG_FILENAME,
    GENAI_SESSION_OPTION_KEYS,
    _sanitize_genai_session_options,
)


def _write(tmp_path, session_options):
    config = {"model": {"type": "llama", "decoder": {"session_options": session_options}}}
    (tmp_path / GENAI_CONFIG_FILENAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return tmp_path / GENAI_CONFIG_FILENAME


def _session_options(path):
    return json.loads(path.read_text(encoding="utf-8"))["model"]["decoder"]["session_options"]


def test_config_entries_is_stripped(tmp_path):
    """The specific key that made every training-stage package GenAI-unloadable."""
    path = _write(
        tmp_path,
        {
            "provider_options": [],
            "config_entries": [["session.model_external_initializers_file_folder_path", "/host/path"]],
        },
    )

    assert _sanitize_genai_session_options(tmp_path) == ["config_entries"]
    assert _session_options(path) == {"provider_options": []}


def test_supported_keys_are_preserved(tmp_path):
    kept = {"log_id": "mobiletransformers", "intra_op_num_threads": 4, "provider_options": []}
    path = _write(tmp_path, dict(kept))

    assert _sanitize_genai_session_options(tmp_path) == []
    assert _session_options(path) == kept


def test_unknown_keys_are_stripped_not_just_config_entries(tmp_path):
    """The guard is an allow-list, so a *future* bad key is caught too — not just the one that bit us."""
    path = _write(tmp_path, {"log_id": "x", "use_env_allocators": True, "disable_cpu_ep_fallback": True})

    assert _sanitize_genai_session_options(tmp_path) == [
        "disable_cpu_ep_fallback",
        "use_env_allocators",
    ]
    assert _session_options(path) == {"log_id": "x"}


def test_no_host_paths_leak_into_the_package(tmp_path):
    """A package is relocatable: it is exported on a host and pushed to a device.

    The old code embedded `str(output_dir)` in the config. Nothing may write an absolute host path into
    genai_config.json — external data resolves relative to the model file's directory (#23).
    """
    path = _write(tmp_path, {"config_entries": [["k", str(tmp_path)]], "log_id": "x"})
    _sanitize_genai_session_options(tmp_path)

    assert str(tmp_path) not in path.read_text(encoding="utf-8")


def test_missing_config_is_not_an_error(tmp_path):
    """genai_config.json is optional — produced upstream, only augmented here."""
    assert _sanitize_genai_session_options(tmp_path) == []


def test_allowlist_matches_the_bundled_runtime():
    """Pin the probed set. If a genai bump changes it, re-probe rather than editing this by hand.

    Derived by feeding each key to `og.Model()` alone against onnxruntime-genai 0.14.1 and recording
    whether the config parsed — not by reading the docs, which are wrong about `config_entries`.
    """
    assert GENAI_SESSION_OPTION_KEYS == {
        "log_id",
        "log_severity_level",
        "enable_profiling",
        "enable_cpu_mem_arena",
        "enable_mem_pattern",
        "intra_op_num_threads",
        "inter_op_num_threads",
        "graph_optimization_level",
        "custom_ops_library",
        "provider",
        "provider_options",
        "external_data_file",
    }
    assert "config_entries" not in GENAI_SESSION_OPTION_KEYS


@pytest.mark.parametrize("key", ["config_entries", "use_env_allocators", "disable_cpu_ep_fallback"])
def test_known_rejected_keys_stay_out_of_the_allowlist(key):
    """Probed as REJECTED by genai 0.14.1. Adding one back re-breaks GenAI loading silently."""
    assert key not in GENAI_SESSION_OPTION_KEYS
