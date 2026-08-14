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


# --- package-model: was a stub that printed "not yet wired" and returned 0 --------------------


def test_package_model_fails_closed_on_a_missing_package(tmp_path, capsys):
    """The whole point of the rewrite: a package that does not exist must NOT report success."""
    code = main(["package-model", "--package", str(tmp_path / "nope")])
    assert code == 1
    assert "not a package directory" in capsys.readouterr().out


def test_package_model_fails_closed_without_a_manifest(tmp_path, capsys):
    bare = tmp_path / "bare"
    bare.mkdir()
    code = main(["package-model", "--package", str(bare)])
    assert code == 1
    assert "mobiletransformers_manifest.json" in capsys.readouterr().out


def test_package_model_requires_a_package_argument(capsys):
    assert main(["package-model"]) == 2
    assert "--package" in capsys.readouterr().out


def test_package_model_dry_run_writes_nothing(tmp_path, capsys):
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg)
    before = json.loads((pkg / "mobiletransformers_manifest.json").read_text())
    assert main(["package-model", "--package", str(pkg), "--dry-run"]) == 0
    assert json.loads((pkg / "mobiletransformers_manifest.json").read_text()) == before
    assert "would re-emit" in capsys.readouterr().out


def test_package_model_reemits_the_integrity_block(tmp_path):
    """Re-emitting over an unchanged tree is a no-op; over a changed one it re-hashes."""
    pkg = tmp_path / "pkg"
    shutil.copytree(PKG, pkg)
    manifest_path = pkg / "mobiletransformers_manifest.json"
    original = json.loads(manifest_path.read_text())

    assert main(["package-model", "--package", str(pkg)]) == 0
    unchanged = json.loads(manifest_path.read_text())
    assert unchanged["sha256"] == original["sha256"]
    assert unchanged["baseModelId"] == original["baseModelId"]

    # Change a file the manifest hashes; the re-emit must notice.
    target = next(rel for rel in original["sha256"] if rel.endswith("generation_config.json"))
    (pkg / target).write_text('{"max_length": 999}\n')
    assert main(["package-model", "--package", str(pkg)]) == 0
    rehashed = json.loads(manifest_path.read_text())
    assert rehashed["sha256"][target] != original["sha256"][target]
    assert rehashed["fileSizes"][target] == (pkg / target).stat().st_size


# --- agent-dataset (#37): import a tool-call corpus, or synthesise a per-user one -------------

AGENT_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "agent" / "mobile_actions_sample.jsonl"


def test_agent_dataset_imports_a_corpus(tmp_path, capsys):
    code = main(["agent-dataset", "--source", str(AGENT_FIXTURE), "--output", str(tmp_path)])
    assert code == 0

    rows = [json.loads(line) for line in (tmp_path / "mobile_actions.jsonl").read_text().splitlines()]
    assert rows and all(set(r) == {"prompt", "completion"} for r in rows)

    schema = json.loads((tmp_path / "action_schema.json").read_text())
    assert {a["actionName"] for a in schema} >= {"send_email", "show_map"}

    out = capsys.readouterr().out
    assert "wrote" in out
    # Unmapped actions are announced, not silently emitted with an empty intent.
    assert "no Android intent mapped" in out and "flashlight" in out


def test_agent_dataset_dry_run_writes_nothing(tmp_path, capsys):
    assert (
        main(["agent-dataset", "--source", str(AGENT_FIXTURE), "--output", str(tmp_path), "--dry-run"]) == 0
    )
    assert not list(tmp_path.iterdir())
    assert "[dry-run]" in capsys.readouterr().out


def test_agent_dataset_limit_is_deterministic(tmp_path):
    def build(where):
        main(["agent-dataset", "--source", str(AGENT_FIXTURE), "--output", str(where), "--limit", "2"])
        return (where / "mobile_actions.jsonl").read_text()

    assert build(tmp_path / "a") == build(tmp_path / "b")
    assert len(build(tmp_path / "c").strip().splitlines()) == 2


def test_agent_dataset_generated_requires_an_allowlist(tmp_path, capsys):
    assert main(["agent-dataset", "--source", "generated", "--output", str(tmp_path)]) == 1
    assert "requires --allowlist" in capsys.readouterr().out


def test_agent_dataset_generated_round_trips_through_the_schema(tmp_path):
    """The imported schema drives the synthetic generator — the per-user layer on the same boundary."""
    main(["agent-dataset", "--source", str(AGENT_FIXTURE), "--output", str(tmp_path / "corpus")])
    schema = tmp_path / "corpus" / "action_schema.json"

    code = main(
        [
            "agent-dataset",
            "--source",
            "generated",
            "--allowlist",
            str(schema),
            "--output",
            str(tmp_path / "user"),
            "--per-action",
            "2",
        ]
    )
    assert code == 0
    rows = [
        json.loads(line) for line in (tmp_path / "user" / "mobile_actions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 * len(json.loads(schema.read_text()))


def test_agent_dataset_reports_an_empty_result_instead_of_writing_nothing(tmp_path, capsys):
    assert (
        main(["agent-dataset", "--source", str(AGENT_FIXTURE), "--output", str(tmp_path), "--split", "nope"])
        == 1
    )
    assert "no rows produced" in capsys.readouterr().out
