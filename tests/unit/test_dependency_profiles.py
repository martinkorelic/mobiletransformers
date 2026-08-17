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


def test_the_abi_coupled_pins_stay_exact() -> None:
    """The paired-stack pins that are REAL constraints must never float.

    These four are ABI/format couplings to the source-built ORT-training wheel, and manifest.json's
    notes call them out as such:

    * ``torch==2.7.1`` / ``peft==0.13.2`` — floating peft to 0.19 renamed
      ``PEFT_TYPE_TO_MODEL_MAPPING`` and required ``torch.distributed.tensor`` (torch>=2.8), so
      ``get_peft_model`` died with ``AttributeError``.
    * ``numpy<2`` — the C extension is built against the numpy 1.26 ABI.
    * ``onnx<1.19`` — ORT 1.23 supports max ONNX IR 11; onnx>=1.19 emits IR 13.

    ``transformers`` is deliberately NOT in this list. It is the one paired_stack entry with no ABI
    relationship to the wheel — pure Python — and it was raised off ``==4.46.2`` on 2026-08-15 after
    three controls (llama decoder, BERT encoder, gemma3) came back identical or better. Confusing
    "recorded in paired_stack" with "load-bearing" is what kept Gemma-3 training blocked for months
    behind a pin that was never the cause.
    """
    for requirement in ("torch==2.7.1", "peft==0.13.2"):
        assert f'"{requirement}"' in PYPROJECT, (
            f"{requirement} is an ABI coupling to the source-built ORT wheel and must stay exact"
        )
    for requirement in ("numpy<2", "onnx<1.19"):
        assert f"\"{requirement}; python_version == '3.12'\"" in PYPROJECT, (
            f"{requirement} is a runtime constraint of the ORT extension and must stay pinned"
        )


def test_training_can_resolve_a_gemma3_capable_transformers() -> None:
    """The training profile must be able to LOAD Gemma-3, or it cannot export a graph for it.

    ``transformers`` below 4.50 has no ``Gemma3ForCausalLM``, no ``Gemma3Config`` and no ``gemma3`` row
    in ``CONFIG_MAPPING_NAMES``, so ``AutoModelForCausalLM.from_pretrained`` fails before any question
    about the exported graph arises. This is the floor that makes
    ``mobiletransformers/functiongemma-270m-it`` buildable at all.

    Asserted as a FLOOR, not an exact version: the point is the capability, not a particular release.
    """
    training = _requirement(r'^\s*"transformers>=[^"]*",\s*$', "transformers")
    assert training is not None, "no transformers requirement in ort-training-local"
    assert ">=4.50" in training, (
        f"the training profile must floor transformers at 4.50 to load Gemma-3, got {training!r}"
    )
    # A REQUIREMENT line, not a mention: the pin's history is written in the comment right above it,
    # and a substring search for "==4.46.2" matches that prose too.
    assert not re.search(r'^\s*"transformers==', PYPROJECT, re.MULTILINE), (
        "an exact transformers== pin is back in pyproject — 4.46.2 cannot load Gemma-3 at all, and it "
        "was removed with evidence (three controls) rather than by guess"
    )


def test_upstream_ceiling_is_preserved() -> None:
    """`<4.58` is optimum-onnx 0.1.0's own declared bound, not a project preference."""
    export = _requirement(r"^export\s*=\s*\[.*$", "transformers")
    assert export is not None and "<4.58" in export


def test_lock_carries_a_gemma3_capable_transformers() -> None:
    """The resolved lock must actually contain a transformers that can load Gemma-3.

    This test used to assert the OPPOSITE — that the lock carried both 4.46.2 and a newer line, proving
    the export/training fork survived `uv lock`. That fork was retired on 2026-08-15 when the training
    profile adopted the same range, so a single version now satisfies both and the lock holds one. The
    invariant worth guarding is the capability, not the split.
    """
    lock = (REPO_ROOT / "uv.lock").read_text()
    versions = set()
    for block in lock.split("[[package]]"):
        if re.search(r'^name = "transformers"', block, re.MULTILINE):
            versions.update(re.findall(r'^version = "([^"]+)"', block, re.MULTILINE))

    assert versions, "no transformers in the lock at all"

    def _parts(version: str) -> tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", version)[:2])

    assert any(_parts(v) >= (4, 50) for v in versions), (
        f"the lock holds no transformers >= 4.50, so Gemma-3 cannot be loaded; found {sorted(versions)}"
    )
