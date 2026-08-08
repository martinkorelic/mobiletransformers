"""Repository-wide grep guards (#4 secrets, #6 registry dispatch) — the CI ratchets.

Two guards were described in the plans as CI gates but existed in neither the ``Makefile`` nor
``ci.yml``: the #4 secret-read guard, and a #6 dispatch guard covering anything beyond
``src/mobiletransformers``. ``tests/unit/test_registries.py`` already guards ``src/`` itself; this
module extends the same idea to the **legacy roots** and the **C++ tree**, which is where every
violation the audit found actually lives.

Both allow-lists are RATCHETS: an entry may only be removed. A file that drops below its allowed
count fails the test, forcing the allowance down with the fix rather than letting it rot.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- #6: registry dispatch --------------------------------------------------------------------

#: Patterns that mean "branching on a closed-set wire value instead of resolving it via a registry".
DISPATCH_PATTERNS = (
    r"architectures\[0\]\s*==",
    r'peft_method\s*==\s*["\']',
    r'train_method\s*==\s*["\']',
    r'merger_type\s*==\s*["\']',
)

#: Legacy roots still outside the new package. **EMPTY — S9 deleted all seven.** Kept as a named
#: constant (rather than deleted) so the scan below still has a defined subject and so re-introducing a
#: root outside `src/` is a one-line, visible change rather than an invisible omission.
LEGACY_ROOTS: tuple[str, ...] = ()

#: repo-relative path -> the number of dispatch hits currently tolerated. ENTRIES MAY ONLY SHRINK.
#:
#: **EMPTY — #6 is closed.** `inference/builder.py`'s 14-branch architecture ladder is gone; it now
#: resolves through `ARCHITECTURE_REGISTRY`, side effects included (PhiMoE's cuda+int4 forcing, Phi3V's
#: exclude_embeds, ChatGLM's hidden_act=swiglu are `option_overrides` / `extra_option_overrides` /
#: `config_overrides` on the row). Verified class-for-class: all 15 branches resolve to the same class
#: objects the ladder constructed.
#:
#: This sat at 14 for months behind "the module is unimportable under every declared profile". The real
#: cause was one renamed ORT import (see `export/quantizer_compat.py`).
#:
#: An empty allow-list makes the test below an assertion rather than a ratchet: ANY string-literal
#: dispatch in a legacy root now fails.
DISPATCH_ALLOWLIST: dict[str, int] = {}

# --- #4: secrets ------------------------------------------------------------------------------

#: Direct environment reads of credential-shaped names. Settings must come from `config/settings.py`
#: (CLI > env > YAML > default), never an ad-hoc `os.environ[...]` scattered through the code.
SECRET_PATTERNS = (
    r"os\.environ\[[\"'][A-Z_]*(TOKEN|SECRET|PASSWORD|API_KEY|APIKEY)[A-Z_]*[\"']\]",
    r"os\.getenv\(\s*[\"'][A-Z_]*(TOKEN|SECRET|PASSWORD|API_KEY|APIKEY)[A-Z_]*[\"']",
)

#: Hardcoded credential literals (an assignment to a secret-shaped name with a non-empty literal).
SECRET_LITERAL_PATTERN = (
    r'(?i)\b(api_?key|secret|password|access_token|auth_token)\s*=\s*["\'][A-Za-z0-9_\-]{16,}["\']'
)

SCAN_DIRS = ("src", "tests", *LEGACY_ROOTS)


def _is_in_comment(line_text: str, pattern: str) -> bool:
    """True when the match sits inside a `#` or `//` comment (or a docstring-ish prose line).

    A known trap with grep guards (recorded in HANDOFF's gotchas): they hit prose that *mentions* the
    banned pattern, e.g. a comment saying "replaces the `merger_type == \"lora\"` dispatch". Rather
    than contorting the prose, the guard ignores matches that begin after a comment marker.
    """
    match = re.search(pattern, line_text)
    if match is None:
        return False
    prefix = line_text[: match.start()]
    return "#" in prefix or "//" in prefix or prefix.lstrip().startswith(("*", '"""', "'''"))


def _grep(patterns: tuple[str, ...], paths: list[Path], includes: tuple[str, ...]) -> list[str]:
    """Return `path:line:text` hits, or [] when nothing matches. Missing paths are skipped."""
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return []
    cmd = ["grep", "-rnE", *[f"--include={g}" for g in includes], "|".join(patterns), *existing]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):  # 1 == no match
        raise RuntimeError(f"grep failed: {result.stderr}")
    hits = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # `path:lineno:text`
        text = line.split(":", 2)[2] if line.count(":") >= 2 else line
        if any(not _is_in_comment(text, p) for p in patterns if re.search(p, text)):
            hits.append(line)
    return hits


def _relative(hit: str) -> str:
    return str(Path(hit.split(":", 1)[0]).resolve().relative_to(REPO_ROOT))


def test_no_dispatch_literals_in_cpp() -> None:
    """The C++ merger used `merger_type == "lora"` etc.; it now dispatches on a typed MergerVariant.

    The `src/`-only guard could never see this, which is why five sites survived every gate.
    """
    cpp = REPO_ROOT / "android/MobileTransformers/MobileTransformers/src/main/cpp"
    if not cpp.is_dir():
        pytest.skip("Android cpp tree not present")
    hits = [
        h
        for h in _grep(DISPATCH_PATTERNS, [cpp], ("*.cpp", "*.h"))
        # Vendored third-party trees are not ours to police.
        if not re.search(r"/(onnxruntime|onnxruntime-genai|tokenizers|proto|includes)/", h)
    ]
    assert not hits, "string-literal dispatch found in cpp/:\n" + "\n".join(hits)


def test_legacy_dispatch_debt_only_shrinks() -> None:
    """Ratchet over the legacy roots: no NEW dispatch literals, and known ones must decrease."""
    roots = [REPO_ROOT / r for r in LEGACY_ROOTS]
    counts: dict[str, int] = {}
    for hit in _grep(DISPATCH_PATTERNS, roots, ("*.py",)):
        counts[_relative(hit)] = counts.get(_relative(hit), 0) + 1

    unlisted = {p: n for p, n in counts.items() if p not in DISPATCH_ALLOWLIST}
    assert not unlisted, (
        "new string-literal dispatch outside the registries — resolve it via "
        f"config/registry/ instead:\n{unlisted}"
    )

    for path, allowed in DISPATCH_ALLOWLIST.items():
        actual = counts.get(path, 0)
        assert actual <= allowed, f"{path}: dispatch literals grew from {allowed} to {actual}"
        assert actual == allowed, (
            f"{path}: down to {actual} dispatch literals (allowance {allowed}) — "
            "lower DISPATCH_ALLOWLIST so the ratchet holds"
        )


def test_no_direct_secret_environment_reads() -> None:
    """#4: credentials come from `config/settings.py`, never an ad-hoc os.environ read."""
    hits = _grep(SECRET_PATTERNS, [REPO_ROOT / d for d in SCAN_DIRS], ("*.py",))
    # settings.py is the ONE place allowed to read the environment.
    hits = [h for h in hits if not _relative(h).endswith("config/settings.py")]
    assert not hits, (
        "direct credential reads from the environment — route them through "
        "mobiletransformers.config.settings:\n" + "\n".join(hits)
    )


def test_no_hardcoded_credential_literals() -> None:
    """#4: no credential-shaped literal is ever committed."""
    hits = _grep((SECRET_LITERAL_PATTERN,), [REPO_ROOT / d for d in SCAN_DIRS], ("*.py",))
    assert not hits, "hardcoded credential literal:\n" + "\n".join(hits)


def test_allowlist_entries_still_exist() -> None:
    """A stale allow-list entry must fail, so the ratchet cannot silently rot."""
    for path in DISPATCH_ALLOWLIST:
        assert (REPO_ROOT / path).is_file(), (
            f"DISPATCH_ALLOWLIST names {path}, which no longer exists — drop the entry"
        )
