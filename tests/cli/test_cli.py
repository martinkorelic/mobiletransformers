"""#15 CLI: export dry-run + push validate/dry-run/upload (injected). No network, no heavy deps."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mobiletransformers.cli import export as export_cli
from mobiletransformers.cli import push as push_cli
from mobiletransformers.cli.main import build_parser, main

PKG = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_package"


def test_export_dry_run_prints_plan(tmp_path, capsys):
    code = main(
        [
            "export",
            "--model",
            "org/tiny",
            "--output",
            str(tmp_path / "out"),
            "--task",
            "text-generation-with-past",
            "--peft",
            "mars-opt1",
            "--quant",
            "int4",
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "cpu-int4" in out
    assert not (tmp_path / "out").exists()


def test_export_registers_push_and_help_lists_it():
    parser = build_parser()
    # push subcommand is registered
    ns = parser.parse_args(["push", "--package", "p", "--repo", "r", "--dry-run"])
    assert ns.command == "push" and ns.func is push_cli.run


def test_push_dry_run_validates_and_writes_card(tmp_path):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg)
    args = build_parser().parse_args(["push", "--package", str(pkg), "--repo", "org/x", "--dry-run"])
    assert args.func(args) == 0
    assert (pkg / "README.md").exists()
    assert "MobileTransformers package" in (pkg / "README.md").read_text()


def test_push_uploads_via_injected_uploader(tmp_path):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg)
    args = build_parser().parse_args(["push", "--package", str(pkg), "--repo", "org/x"])
    calls = {}
    push_cli.run(args, uploader=lambda **kw: calls.update(kw))
    assert calls["repo_id"] == "org/x" and Path(calls["folder_path"]) == pkg


def test_push_aborts_on_invalid_package(tmp_path):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg)
    # Corrupt: point defaultVariant at a nonexistent variant.
    mpath = pkg / "mobiletransformers_manifest.json"
    data = json.loads(mpath.read_text())
    data["defaultVariant"] = "ghost"
    mpath.write_text(json.dumps(data))
    args = build_parser().parse_args(["push", "--package", str(pkg), "--repo", "org/x", "--dry-run"])
    assert args.func(args) == 1  # fail closed before upload


# --- federated (#35) — the subcommand had zero CLI coverage ---------------------------------------
def test_federated_is_registered_in_the_parser():
    """`cli/main.py` registers it, but `docs/PUBLIC_API.md`'s table omitted it — pin the wiring."""
    parser = build_parser()
    args = parser.parse_args(
        ["federated", "simulate", "--package", "pkg", "--output", "out", "--clients", "3"]
    )
    assert args.clients == 3
    assert args.output == "out"
    assert callable(args.func)


def test_federated_simulate_fails_closed_on_a_missing_package(tmp_path, capsys):
    parser = build_parser()
    args = parser.parse_args(
        ["federated", "simulate", "--package", str(tmp_path / "nope"), "--output", str(tmp_path / "o")]
    )
    assert args.func(args) == 1
    assert "federated simulate failed" in capsys.readouterr().out


def test_federated_simulate_rejects_an_unknown_strategy(tmp_path, capsys):
    """v1 supports fedavg only; anything else must fail rather than silently fall back."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "federated",
            "simulate",
            "--package",
            str(tmp_path),
            "--output",
            str(tmp_path / "o"),
            "--strategy",
            "fedprox",
        ]
    )
    assert args.func(args) == 1


# --- export --config overlay + --validate (#15) ---------------------------------------------------
def _export_args(argv):
    return build_parser().parse_args(["export", *argv])


def test_export_config_overlay_supplies_unset_knobs(tmp_path):
    """`--config` was accepted and silently dropped while docs/EXPORT.md documented it as working.

    Exercised through the pure overlay rather than a full dry-run: task discovery imports
    `transformers`, which the core env deliberately does not install.
    """
    cfg = tmp_path / "export.yml"
    cfg.write_text("export:\n  model: org/from-yaml\n  output: out-from-yaml\n  peft: mars\n  rank: 16\n")
    args = _export_args(["--config", str(cfg)])
    export_cli._apply_config_overlay(args)
    assert args.model == "org/from-yaml"
    assert args.output == "out-from-yaml"
    assert args.peft == "mars"
    assert args.rank == 16


def test_export_config_accepts_a_top_level_mapping_too(tmp_path):
    cfg = tmp_path / "export.yml"
    cfg.write_text("model: org/flat\noutput: o\nquant: fp16\n")
    args = _export_args(["--config", str(cfg)])
    export_cli._apply_config_overlay(args)
    assert args.model == "org/flat"
    assert args.quant == "fp16"


def test_export_cli_flags_win_over_the_config(tmp_path):
    """Documented precedence is CLI > YAML > default."""
    cfg = tmp_path / "export.yml"
    cfg.write_text("export:\n  model: org/from-yaml\n  output: o\n  peft: mars\n  rank: 32\n")
    args = _export_args(["--config", str(cfg), "--model", "org/from-cli", "--peft", "lora-xs"])
    export_cli._apply_config_overlay(args)
    assert args.model == "org/from-cli"
    assert args.peft == "lora-xs"
    assert args.rank == 32  # not passed on the CLI, so the YAML value still applies


def test_export_config_rejects_unknown_keys(tmp_path, capsys):
    cfg = tmp_path / "export.yml"
    cfg.write_text("export:\n  model: m\n  output: o\n  nonsense: 1\n")
    args = _export_args(["--config", str(cfg), "--dry-run"])
    assert args.func(args) == 1
    assert "unknown export key" in capsys.readouterr().out


def test_export_requires_model_from_some_source(capsys):
    args = _export_args(["--output", "o", "--dry-run"])
    assert args.func(args) == 1
    assert "--model is required" in capsys.readouterr().out


def test_validate_subcommand_reports_a_missing_package(tmp_path, capsys):
    """The stub used to print 'not yet wired' and return 0 for ANY input, including nonexistent ones."""
    args = build_parser().parse_args(["validate", "--package", str(tmp_path / "absent")])
    assert args.func(args) == 1
    assert "validation failed" in capsys.readouterr().out


def test_validate_subcommand_accepts_the_tiny_fixture_package(capsys):
    pkg = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_package"
    args = build_parser().parse_args(["validate", "--package", str(pkg)])
    assert args.func(args) == 0
    assert "package OK" in capsys.readouterr().out


def test_validate_rejects_a_directory_without_a_manifest(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    args = build_parser().parse_args(["validate", "--package", str(tmp_path / "empty")])
    assert args.func(args) == 1
    assert "no mobiletransformers_manifest.json" in capsys.readouterr().out
