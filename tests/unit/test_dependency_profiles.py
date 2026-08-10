"""Profile-fork invariants (#2/#3, and #37's unblocking).

The pins in `pyproject.toml` are two different KINDS of thing that look identical:

* **upstream ceilings** — `optimum~=2.1.0`, `transformers<4.58`. Not ours; fighting them fails to
  install. They are recorded in IMPLEMENTATION_ORDER's "Upstream version ceilings" table.
* **paired-stack reproductions** — `torch==2.7.1`, `peft==0.13.2`, `transformers==4.46.2`,
  `numpy<2` in `ort-training-local`. These reproduce the environment the source-built ORT-training
  wheel was compiled against (`third_party/onnxruntime/manifest.json`). They may be forked per
  profile, but never floated inside that group.

Confusing the two costs a cycle in either direction: "the pins are stale, bump them" breaks
`get_peft_model`, and "the pins are untouchable" left #37 unable to load Gemma-3 at all. These tests
pin the distinction so the lock cannot silently collapse the fork back onto one version.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Raw text, not a TOML parse: `tomllib` is 3.11+ and the dev/CI profile floors at 3.10. The rest of
#: the guard tests (`test_version_sites`, `test_gate_ratchet`) read pyproject the same way.
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _requirement(block_pattern: str, package: str) -> str | None:
    """The requirement string for `package` inside the first line matching `block_pattern`."""
    block = re.search(block_pattern, PYPROJECT, re.MULTILINE)
    if block is None:
        return None
    found = re.search(rf'"({re.escape(package)}[^"]*)"', block.group(0))
    return found.group(1) if found else None


def test_export_and_training_transformers_are_forked() -> None:
    """The export profile must be able to resolve a NEWER transformers than the training profile.

    `[tool.uv] conflicts` already declares the group and the extra mutually exclusive, but that alone
    does not force different versions: uv prefers a single version whenever one satisfies both, and
    `transformers==4.46.2` satisfied the old `>=4.45`. The floor is what makes the fork real.

    4.50 is the floor because Gemma-3 (`Gemma3ForCausalLM`, `Gemma3Config`, the `gemma3` row in
    `CONFIG_MAPPING_NAMES`) does not exist before it — #37's gate could not load the model, which is
    upstream of any question about the exported graph.
    """
    export = _requirement(r"^export\s*=\s*\[.*$", "transformers")
    training = _requirement(r'^\s*"transformers==[^"]*",\s*$', "transformers")

    assert export is not None, "no transformers requirement in the export extra"
    assert training is not None, "no exact transformers pin in ort-training-local"
    assert ">=4.50" in export, f"export must floor transformers at 4.50 for Gemma-3, got {export!r}"
    assert training == "transformers==4.46.2", (
        "the ort-training-local pin reproduces the ORT wheel's paired stack and must stay exact; "
        f"got {training!r}"
    )
    # The floors must be incompatible, or uv collapses back to one version and the fork is a no-op.
    assert "4.46.2" not in export


def test_upstream_ceiling_is_preserved() -> None:
    """`<4.58` is optimum-onnx 0.1.0's own declared bound, not a project preference."""
    export = _requirement(r"^export\s*=\s*\[.*$", "transformers")
    assert export is not None and "<4.58" in export


def test_lock_carries_both_transformers_lines() -> None:
    """The resolved lock must actually contain both versions — the proof the fork survived `uv lock`."""
    lock = (REPO_ROOT / "uv.lock").read_text()
    versions = set()
    for block in lock.split("[[package]]"):
        if re.search(r'^name = "transformers"', block, re.MULTILINE):
            versions.update(re.findall(r'^version = "([^"]+)"', block, re.MULTILINE))

    assert "4.46.2" in versions, f"training pin missing from the lock; found {sorted(versions)}"
    newer = [v for v in versions if v != "4.46.2"]
    assert newer, (
        "the lock holds only transformers 4.46.2 — the export fork collapsed, and Gemma-3 is "
        "unavailable again"
    )
