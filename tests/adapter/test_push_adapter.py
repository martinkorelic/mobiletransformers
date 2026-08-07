"""#22 push-adapter CLI: PEFT (Mode 1) / native (Mode 2) / --peft-only / injected upload."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mobiletransformers.cli import push_adapter as pa
from mobiletransformers.cli.main import build_parser
from mobiletransformers.exceptions import MobileTransformersError
from tests.adapter._helpers import TRAINABLE, make_cache

_LORA = {"adapter_A": "l.lora_A", "adapter_B": "l.lora_B"}
_MARS = {"shared_A": "m.shared_A", "adapter_B": "m.mars_B"}


def _args(cache, repo="org/adp", extra=None):
    argv = ["push-adapter", "--cache-repo", str(cache), "--repo-id", repo, "--base-license", "Apache-2.0"]
    argv += extra or []
    return build_parser().parse_args(argv)


def test_lora_dry_run_builds_peft_adapter(tmp_path):
    cache = make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA)
    args = _args(cache, extra=["--dry-run"])
    assert args.func(args) == 0
    out = cache / ".adapter-push"
    assert (out / "adapter_config.json").is_file()
    assert (out / "README.md").is_file()
    assert "Privacy warning" in (out / "README.md").read_text()


def test_lora_dry_run_reports_unmaterialized_weights(tmp_path, capsys):
    """In the core env (no torch/safetensors) the dry run must SAY the weights are missing."""
    cache = make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA)
    args = _args(cache, extra=["--dry-run"])
    assert args.func(args) == 0
    out = cache / ".adapter-push"
    if not (out / "adapter_model.safetensors").is_file():
        assert "adapter_model.safetensors NOT materialized" in capsys.readouterr().out


def test_peft_upload_without_weights_fails_closed(tmp_path):
    """#22 regression: a Mode-1 push used to publish adapter_config.json with NO weights.

    `PeftModel.from_pretrained` cannot load such a repo, and the failure was invisible because the
    materialization step was simply skipped with a comment. Outside the `train` profile the upload
    must now refuse rather than publish an unusable adapter.
    """
    if _train_profile_available():
        pytest.skip("train profile present; materialization succeeds, see test_convert.py")
    cache = make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA)
    args = _args(cache)  # no --dry-run
    with pytest.raises(MobileTransformersError, match="without adapter_model.safetensors"):
        pa.run(args, uploader=lambda **kw: None)


def _train_profile_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None
    )


def test_mars_dry_run_builds_native_adapter(tmp_path):
    cache = make_cache(tmp_path / "c", peft_method="mars", component_roles=_MARS)
    args = _args(cache, extra=["--dry-run"])
    assert args.func(args) == 0
    out = cache / ".adapter-push"
    assert (out / "mobiletransformers_adapter.json").is_file()
    assert (out / f"{TRAINABLE}.bin").is_file()  # merged tensor copied
    assert (out / "weight_handoff_map.json").is_file()
    assert not (out / "adapter_config.json").exists()


def test_peft_only_errors_on_mars(tmp_path):
    cache = make_cache(tmp_path / "c", peft_method="mars", component_roles=_MARS)
    args = _args(cache, extra=["--dry-run", "--peft-only"])
    assert args.func(args) == 1


def test_upload_via_injected_uploader(tmp_path):
    # Mode 2 (native): copies merged .bin files straight out of the cache, so the upload path is
    # exercisable in the core env. Mode 1 needs the train profile to produce its weights.
    cache = make_cache(tmp_path / "c", peft_method="mars", component_roles=_MARS)
    args = _args(cache)  # no --dry-run
    calls = {}
    rc = pa.run(args, uploader=lambda **kw: calls.update(kw))
    assert rc == 0
    assert calls["repo_id"] == "org/adp"
    assert Path(calls["folder_path"]) == cache / ".adapter-push"
