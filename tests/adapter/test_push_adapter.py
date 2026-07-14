"""#22 push-adapter CLI: PEFT (Mode 1) / native (Mode 2) / --peft-only / injected upload."""

from __future__ import annotations

from pathlib import Path

from mobiletransformers.cli import push_adapter as pa
from mobiletransformers.cli.main import build_parser
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
    cache = make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA)
    args = _args(cache)  # no --dry-run
    calls = {}
    rc = pa.run(args, uploader=lambda **kw: calls.update(kw))
    assert rc == 0
    assert calls["repo_id"] == "org/adp"
    assert Path(calls["folder_path"]) == cache / ".adapter-push"
