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
#: constant (rather than deleted) so re-introducing a root outside `src/` is a one-line, visible
#: change rather than an invisible omission.
LEGACY_ROOTS: tuple[str, ...] = ()

#: First-party Python that lives OUTSIDE the wheel and is therefore ungated by ruff/mypy: the paper
#: experiments, the shell/Python helpers, and the gate spikes.
#:
#: 2026-08-14: this replaces `LEGACY_ROOTS` as the dispatch guard's subject. With `LEGACY_ROOTS`
#: empty, `test_legacy_dispatch_debt_only_shrinks` and `test_allowlist_entries_still_exist` scanned
#: NOTHING — 2 of the 7 guards asserted nothing at all while reading as if they were enforcing. These
#: directories are the real remaining un-gated first-party Python, and `research/` genuinely carries
#: the debt below.
NON_PACKAGE_PY_ROOTS: tuple[str, ...] = ("research", "scripts", "spikes")

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
#: Within `src/` this is closed and stays closed — `tests/unit/test_registries.py` asserts it there.
#:
#: The counts below are `research/`'s, measured 2026-08-14 when this guard was re-pointed at
#: `NON_PACKAGE_PY_ROOTS`. They are the paper's ablation scripts: `offline_train_eval.py` branches over
#: the PEFT methods it compares, which is what an experiment harness legitimately does — it is out of
#: the wheel and out of ruff/mypy for the same reason. The ratchet's job here is that **no new**
#: dispatch appears, and that these numbers only fall (e.g. if a script is retired).
#:
#: Owner: `research/` is the paper's, not the library's — it is excluded from the wheel, from ruff and
#: from mypy by `pyproject.toml`, and its READMEs record why it is kept. No plan owns migrating it.
DISPATCH_ALLOWLIST: dict[str, int] = {
    "research/offline_train_eval.py": 10,
    "research/pytorch_experiments/dynamic_model_training.py": 3,
}

# --- #33/#6: architecture-name literals -------------------------------------------------------

#: Decoder module names spelled as string literals. These belong in `config/registry/architecture.py`
#: as data (`ArchitectureSpec.projection_names` / `attention_module_name`) and nowhere else.
#:
#: This guard exists because #33 found **six** of them in `peft/mars/model.py` alone — the attention
#: lookup, the hidden-states hook, three `register_proj_hook` calls and the `projection_type` ladder —
#: plus a seventh in `peft/mapping.py`. Every one of them was invisible on a decoder and wrong on an
#: encoder, and the failure mode was a SILENT no-op: MARS degraded to unshared adapters with no error.
#: A grep guard is the cheap way to keep them from creeping back, and it immediately found a live
#: defect the first time it ran (`training_export.py`'s `--lora_target` still defaulted to the
#: decoder-specific `["q_proj", "k_proj"]` the registry had replaced, silently overriding it).
#:
#: Deliberately NOT included: `query`/`key`/`value`. They are BERT's projection names but also
#: ordinary English used throughout the RAG and hub code, so they would drown the signal.
ARCHITECTURE_LITERAL_PATTERNS = (
    r"[\"']self_attn[\"']",
    r"[\"'][qkvo]_proj[\"']",
    r"[\"'](gate|up|down)_proj[\"']",
    r"[\"'][qkv]_lin[\"']",
)

#: Files allowed to spell them, and why.
ARCHITECTURE_LITERAL_ALLOWLIST: dict[str, int] = {
    # The vendored ONNX Runtime GenAI builder. Upstream code, treated as upstream (see #6/#7).
    "src/mobiletransformers/inference/builder.py": 10_000,
    # `artifacts/validation.py` is GONE from this list as of 2026-08-10: the spec is threaded in
    # (`ONNXModelGenerator(architecture_spec=...)` -> `_attention_module_name()`), which is the fix the
    # note here asked for rather than a wider allowance. Do not re-add it.
}

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

#: 2026-08-14: widened from `("src", "tests")` — the secret guard could not see `research/`,
#: `scripts/` or `spikes/`, which is precisely where a quick experiment script would paste a key.
SCAN_DIRS = ("src", "tests", *LEGACY_ROOTS, *NON_PACKAGE_PY_ROOTS)


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
    """Ratchet over first-party Python outside the wheel: no NEW dispatch, known ones only decrease."""
    roots = [REPO_ROOT / r for r in (*LEGACY_ROOTS, *NON_PACKAGE_PY_ROOTS)]
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


def test_no_architecture_literals_outside_the_registry() -> None:
    """Per-architecture module names are registry DATA; a literal elsewhere is a latent encoder bug.

    Ratchet, like the dispatch guard: an allow-list entry may only shrink. The registry itself is
    excluded because it is where these names are supposed to live.
    """
    src = REPO_ROOT / "src"
    registry = "src/mobiletransformers/config/registry/"

    counts: dict[str, int] = {}
    for hit in _grep(ARCHITECTURE_LITERAL_PATTERNS, [src], ("*.py",)):
        rel = _relative(hit)
        if rel.startswith(registry):
            continue
        counts[rel] = counts.get(rel, 0) + 1

    unlisted = {p: n for p, n in counts.items() if p not in ARCHITECTURE_LITERAL_ALLOWLIST}
    assert not unlisted, (
        "hardcoded architecture module names outside config/registry/ — put them on the "
        f"ArchitectureSpec row instead (projection_names / attention_module_name):\n{unlisted}"
    )

    for path, allowed in ARCHITECTURE_LITERAL_ALLOWLIST.items():
        actual = counts.get(path, 0)
        assert actual <= allowed, f"{path}: architecture literals grew from {allowed} to {actual}"


# --- G2: stage-path concatenation ---------------------------------------------------------------

#: Building a package STAGE path by appending its name to a string.
#:
#: A package has two on-disk layouts — the hub's `variants/<id>/train` and the flat device cache's
#: `<cacheDir>/<repoId>/train` — and the manifest declares the first in `variant.paths`. Before G2,
#: exactly ONE consumer in the repo read those declarations (`cli/federated.py`); every other site
#: spelled the join by hand, in Python and Kotlin, and the two layouts were routinely confused.
#:
#: That confusion is not hypothetical: the #35 simulation looked for `<package>/train/` — the CACHE
#: layout — inside a hub package, and ORT reported `INVALID_ARGUMENT : Invalid fd was supplied: -1`,
#: naming no file. It cost a cycle.
#:
#: Resolve through `artifacts/package_paths.py::PackagePaths` (Python) or `packages/PackagePaths.kt`
#: (Kotlin) instead. **C++ is deliberately NOT scanned**: it never resolves a stage — Kotlin hands it
#: an already-resolved directory over JNI, and its joins (`inference_dir + "/weight_handoff_map.json"`)
#: append a FILENAME to a resolved dir, which is not this defect.
STAGE_PATH_PATTERNS = (
    # Python: `something / "train"`, `something / "inference"`, `something / "embedding"`
    r'/\s*"(train|inference|embedding)"',
    # Kotlin: `File(x, "train")`, `"$cacheDir/$repoId/train"`, `"…/inference"`
    r'File\([^)]*,\s*"(train|inference|embedding|tokenizer)"\s*\)',
    r'"\$[A-Za-z_{][^"]*/(train|inference|embedding)(/|")',
)

#: repo-relative path -> hits currently tolerated. ENTRIES MAY ONLY SHRINK.
#:
#: `export/pipeline.py` is the package PRODUCER: it creates `variants/<id>/<stage>` on disk, so it is
#: the one place that legitimately writes the layout rather than reading it. It is listed rather than
#: exempted by rule so that any growth still has to be argued for.
#: The two PRODUCERS are listed rather than exempted by rule, so growth still has to be argued for:
#: `export/pipeline.py` creates `variants/<id>/<stage>` on disk, and `ModelPackageInstaller.kt` is the
#: function that CONVERTS the hub layout into the flat cache layout. Something has to write each
#: layout down once; everything else reads it through PackagePaths.
#:
#: The two RAG sites were the recorded debt here, and they are now ZERO: `PackagePaths` grew
#: `embeddingDatabase`/`embeddingTokenizer` (sub-paths of the embedding stage, created at ingest time
#: rather than shipped) and `ORTRetriever`/`ORTVectorDatabase` resolve through it. The prose comment
#: they carried had already drifted from the code — it described the store as `<repo>/database/` while
#: the code wrote `<repo>/embedding/database/` — which is precisely the drift this guard exists for.
_SDK = "android/MobileTransformers/MobileTransformers/src/main/java/com/martinkorelic/mobiletransformers"

STAGE_PATH_ALLOWLIST: dict[str, int] = {
    "src/mobiletransformers/export/pipeline.py": 8,
    f"{_SDK}/packages/ModelPackageInstaller.kt": 1,
}

#: Files that necessarily spell a layout: the resolvers themselves and their tests.
_RESOLVER_FILES = ("package_paths.py", "PackagePaths.kt", "PackagePathsTest.kt", "test_package_paths.py")


def test_no_stage_path_concatenation() -> None:
    """Stage directories come from PackagePaths, not from appending a stage name to a string.

    Covers Kotlin as well as Python — the guards historically included only `*.py`/`*.cpp`/`*.h`, and
    Kotlin is where most of these sites lived.
    """
    scan = [REPO_ROOT / "src", REPO_ROOT / "android"]
    counts: dict[str, int] = {}
    for hit in _grep(STAGE_PATH_PATTERNS, scan, ("*.py", "*.kt")):
        rel = _relative(hit)
        if rel.endswith(_RESOLVER_FILES):
            continue
        # Test fixtures build synthetic packages on purpose.
        if rel.startswith("tests/") or "/androidTest/" in rel or "/src/test/" in rel:
            continue
        counts[rel] = counts.get(rel, 0) + 1

    unlisted = {p: n for p, n in counts.items() if p not in STAGE_PATH_ALLOWLIST}
    assert not unlisted, (
        "stage paths built by string concatenation — resolve them through PackagePaths "
        f"(artifacts/package_paths.py / packages/PackagePaths.kt) instead:\n{unlisted}"
    )

    for path, allowed in STAGE_PATH_ALLOWLIST.items():
        actual = counts.get(path, 0)
        assert actual <= allowed, f"{path}: stage-path concatenation grew from {allowed} to {actual}"
