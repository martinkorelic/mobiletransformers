"""#15 CLI: export dry-run + push validate/dry-run/upload (injected). No network, no heavy deps."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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
