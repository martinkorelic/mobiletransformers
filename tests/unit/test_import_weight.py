"""`import mobiletransformers` must stay cheap.

The package deliberately defers every heavy dependency: the CLI has to start, `--dry-run` has to work,
and `make check` has to run in a core environment with no torch, no onnxruntime and no optimum. The
legacy roots do the opposite — `trainer/builder.py` imports torch at module level,
`trainer/utils.py` imports `deepeval.benchmarks`, `database/vector_entity.py` star-imports objectbox —
so moving them into the package is exactly the change most likely to break this, silently, by making
the top-level import drag one of them in.

A subprocess is used because pytest has already imported plenty by the time a test runs; only a fresh
interpreter can answer "what does importing this package alone cost?".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Importing the package must not pull any of these in.
HEAVY = (
    "torch",
    "onnxruntime",
    "optimum",
    "transformers",
    "peft",
    "deepeval",
    "objectbox",
    "tensorflow",
    "flwr",
    "safetensors",
)

_PROBE = """
import json, sys
import mobiletransformers  # noqa: F401
print(json.dumps(sorted(m for m in sys.modules if "." not in m)))
"""


def _top_level_modules_after_import() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    assert result.returncode == 0, f"importing mobiletransformers failed:\n{result.stderr}"
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_importing_the_package_pulls_in_no_heavy_dependency() -> None:
    loaded = _top_level_modules_after_import()
    leaked = sorted(set(HEAVY) & loaded)
    assert not leaked, (
        f"`import mobiletransformers` now pulls in {leaked}. Move that import inside the function "
        "that needs it (see the `# noqa: PLC0415` lazy-import convention used throughout the package)."
    )


def test_cli_help_runs_without_the_heavy_profiles() -> None:
    """The CLI must be usable in the core env — `--help` importing torch would be a regression."""
    result = subprocess.run(
        [sys.executable, "-m", "mobiletransformers.cli.main", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "export" in result.stdout


def test_placeholder_subpackages_stay_empty_until_they_are_migrated() -> None:
    """The Migration Map's target subpackages are empty placeholders.

    When one gains content it must also gain a `MODULE_LOCATIONS` entry in `test_symbol_golden.py`;
    this catches code landing there without the move being recorded.
    """
    from tests.fixtures.symbol_tools import public_symbols
    from tests.unit.test_symbol_golden import MIGRATED_PATHS, MODULE_LOCATIONS

    targets = ("peft", "training", "inference", "rag", "evaluation")
    recorded = set(MODULE_LOCATIONS.values()) | MIGRATED_PATHS
    for name in targets:
        package = REPO_ROOT / "src" / "mobiletransformers" / name
        if not package.is_dir():
            continue
        for module in sorted(package.rglob("*.py")):
            rel = str(module.relative_to(REPO_ROOT))
            if module.name == "__init__.py" and not public_symbols(module):
                # A placeholder, or a package `__init__.py` that only carries a docstring. Neither
                # defines a symbol, so there is nothing for the symbol golden to follow — and a
                # migrated subpackage SHOULD get a docstring explaining what it now owns. The check
                # is about code landing here unrecorded, which `public_symbols` measures directly.
                continue
            assert rel in recorded, (
                f"{rel} exists but is recorded in neither MODULE_LOCATIONS nor MIGRATED_PATHS — "
                "record the move so the symbol golden follows the module"
            )
