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


# --- #17/#19: the showcase app must use the public facade, nothing else -------------------------

#: The sample app's Kotlin sources. `MobileTransformersApp` is the reference example every external
#: consumer is pointed at, so anything it reaches for is, in practice, public API.
_APP = "android/MobileTransformers/MobileTransformersApp/src/main"

#: Library sources, used to DERIVE the banned set rather than hand-list it.
_SDK_MAIN = f"{_SDK}"

#: Files whose declarations are internal machinery by construction. A type declared in one of these
#: is banned from app code no matter what it is called.
#:
#: Deriving from the declaring FILE is the point. The obvious guard — ban `ORT*`, `*Native`,
#: `*Repository` by name — has a hole big enough to drive the old app through: it also imported
#: `DeviceOptions`, `SamplingOptions`, `SchedulerConfig`, `InferenceProgress`, `TrainingProgress` and
#: `RagResult`, none of which match any of those patterns, yet all six are declared inside
#: `ORTGenerationConfig.kt` / `ORTTrainingConfig.kt` / `ORTProgress.kt` and exist only to build or
#: receive an `ORT*` config. A name-based guard would have reported "migrated" with the app still
#: wired to the engine. This cannot rot as new types are added to those files.
_INTERNAL_DECL_FILE_GLOB = "ORT*.kt"

#: Name patterns that are internal regardless of where they are declared.
_INTERNAL_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*(Native|Repository)$")

_KOTLIN_DECL_RE = re.compile(
    r"\b(?:data class|sealed class|enum class|value class|abstract class|open class|class|interface|"
    r"fun interface|object)\s+([A-Z][A-Za-z0-9_]*)"
)


def _banned_sdk_symbols() -> set[str]:
    """Every library type the sample app must not import, derived from the library sources."""
    sdk = REPO_ROOT / _SDK_MAIN
    banned: set[str] = set()
    for source in sdk.rglob("*.kt"):
        declared = set(_KOTLIN_DECL_RE.findall(source.read_text(encoding="utf-8")))
        # (1) anything declared in an ORT*.kt file, (2) anything *Native / *Repository anywhere.
        if source.match(_INTERNAL_DECL_FILE_GLOB):
            banned |= declared
        banned |= {name for name in declared if _INTERNAL_NAME_RE.match(name)}
    return banned


def test_the_sample_app_uses_only_the_public_facade() -> None:
    """`MobileTransformersApp` may not reach past the facade into the engine layer.

    #17/#19 built a public SDK surface — `MobileTransformers.fromPretrained`,
    `MobileTransformerModel`, the `config/` types — and then nothing exercised it: the app that ships
    with the library drove `LLMRepository`/`TrainingRepository`/`RagRepository`/`InferenceRepository`
    and the `ORT*` configs directly, so the ergonomics of the documented API had never met a real
    screen and there was no worked example of the thing consumers are told to adopt.

    This guard is what makes "migrated" checkable rather than claimed. It is deliberately a source
    grep over imports: an app that cannot *name* an internal type cannot reach one.

    Note the third rule — the `internal/` package. Kotlin's `internal` is module-scoped, and the app
    is a separate Gradle module, so the compiler already stops it; the check is here so the intent is
    stated in one place with the other two rather than depending on a module boundary staying put.
    """
    app = REPO_ROOT / _APP
    if not app.is_dir():
        pytest.skip("Android sample app not present in this checkout")

    banned = _banned_sdk_symbols()
    assert banned, "the banned set is empty — this guard would pass vacuously"

    sources = sorted(app.rglob("*.kt"))
    assert sources, f"no Kotlin sources under {_APP} — the guard would pass vacuously"

    violations: list[str] = []
    for source in sources:
        rel = str(source.relative_to(REPO_ROOT))
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("import com.martinkorelic.mobiletransformers"):
                continue
            symbol = stripped.removeprefix("import ").split(" as ")[0].rsplit(".", 1)[-1]
            if ".internal." in stripped:
                violations.append(f"{rel}:{lineno}: {stripped}  (internal package)")
            elif symbol in banned:
                violations.append(f"{rel}:{lineno}: {stripped}  ({symbol} is engine-layer)")

    assert not violations, (
        "the sample app reaches past the public facade. Use `MobileTransformers.fromPretrained` and "
        "the `config/` types instead; if the facade cannot express what the screen needs, FIX THE "
        "FACADE and record it against #17/#19 — a reach-around is the debt reappearing under a new "
        "name:\n" + "\n".join(violations)
    )


# --- #21/#22: the library manifest must declare the permissions its own code needs ----------------

#: The library manifest. Permissions belong HERE, not in the consuming app: an integrator who calls
#: only the documented entry point cannot be expected to discover that it opens a socket.
_SDK_MANIFEST = "android/MobileTransformers/MobileTransformers/src/main/AndroidManifest.xml"

#: Permission -> the library code path that cannot run without it. Each entry names a call an app
#: reaches through the PUBLIC facade, which is what makes the permission the library's to declare.
_REQUIRED_PERMISSIONS: dict[str, str] = {
    "android.permission.INTERNET": (
        "HubDownloader/PackageDownloader (OkHttp) and AdapterUploader — reached from "
        "MobileTransformers.fromPretrained whenever the package is not already installed"
    ),
    "android.permission.FOREGROUND_SERVICE": "TrainingScheduler's foreground WorkManager worker (#34)",
    "android.permission.POST_NOTIFICATIONS": "the foreground worker's mandatory progress notification",
    "android.permission.WAKE_LOCK": "held by WorkManager for the duration of a training chunk",
}


def test_the_library_manifest_declares_the_permissions_its_own_code_needs() -> None:
    """Every permission the SDK's own code paths require must be in the LIBRARY manifest.

    This guard exists because ``INTERNET`` was absent for the entire life of the #21 Hub-download
    feature. The whole stack — ``HubResolver``, ``DownloadPlanner``, ``PackageDownloader`` (streaming,
    HTTP Range-resume, sha256 verify-and-retry), ``PackageDownloadWorker``, ``ModelPackageInstaller`` —
    was written, reviewed and JVM-tested against MockWebServer, and **could never have run on a
    device**: the first real GET throws ``SecurityException``. MockWebServer talks to localhost from
    the JVM, where no Android permission applies, so the test suite could not see it either.

    That is the general shape worth guarding: a permission is invisible to every automated test that
    is not an instrumented one, so nothing else in this repo can catch its absence.
    """
    manifest = REPO_ROOT / _SDK_MANIFEST
    assert manifest.is_file(), f"{_SDK_MANIFEST} is missing — this guard would pass vacuously"
    declared = set(
        re.findall(r'<uses-permission\s+android:name="([^"]+)"', manifest.read_text(encoding="utf-8"))
    )
    assert declared, f"{_SDK_MANIFEST} declares no permissions at all — the guard would pass vacuously"

    missing = {name: reason for name, reason in _REQUIRED_PERMISSIONS.items() if name not in declared}
    assert not missing, (
        f"{_SDK_MANIFEST} is missing permissions the library's OWN code needs. A consumer calling the "
        "public facade cannot be expected to discover these, and no JVM test can catch their absence "
        "— only a device run can:\n"
        + "\n".join(f"  {name} — {reason}" for name, reason in sorted(missing.items()))
    )


# --- portability: no machine-specific absolute paths in tracked files ---------------------------

#: Absolute paths that only resolve on one developer's machine. `/home/<user>` and `/Users/<user>`
#: leak an identity and break for everyone else; `/opt/android-studio` is a *Linux Android Studio*
#: install location that is simply absent on macOS, on a CI runner, and under any standalone JDK.
_MACHINE_PATH_PATTERNS = (
    re.compile(r"/home/[a-z][a-z0-9_-]*/"),
    re.compile(r"/Users/[A-Za-z][A-Za-z0-9_-]*/"),
    re.compile(r"/opt/android-studio"),
)

#: repo-relative path -> why this file is allowed to name one. **Deliberately tiny.** Each entry is a
#: documented fallback of last resort, not a dependency: every one of them probes for a portable
#: answer first and reaches the literal only when nothing else is found.
_MACHINE_PATH_ALLOWLIST: dict[str, str] = {
    "Makefile": "JAVA_HOME fallback, after probing PATH for a JDK 17+",
    "scripts/lib/java_home.sh": "the shared JAVA_HOME probe — the one place the fallback is defined",
    "spikes/genai_external_swap/README.md": "a spike's recorded command line, not a code path",
    # A guard cannot avoid spelling out what it forbids: the patterns, the docstring explaining them
    # and the can-it-fail samples all contain literal matches. This is gotcha 8 in the operational
    # notes ("grep guards match docstrings") in its unavoidable form — hence the separate
    # `test_the_machine_path_guard_can_actually_fail`, which is what keeps this file honest given
    # that it exempts itself.
    "tests/unit/test_guards.py": "defines the patterns and the samples proving they still match",
}

#: Extensions worth scanning. Binary and vendored trees are excluded by `git ls-files` naturally
#: (they are gitignored), so this is about keeping the scan cheap and the failures readable.
_MACHINE_PATH_SUFFIXES = (
    ".py",
    ".sh",
    ".kt",
    ".kts",
    ".java",
    ".cpp",
    ".h",
    ".hpp",
    ".cmake",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".properties",
    ".gradle",
    ".xml",
    ".cff",
)


def _tracked_files() -> list[str]:
    """Tracked paths only. `git ls-files` rather than `rglob` so gitignored trees — `build/`, the
    vendored `jniLibs/`, every `.venv` — are excluded by construction rather than by an exclude list
    that rots. The same choice `tests/unit/test_docs.py` documents."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_no_machine_specific_absolute_paths_in_tracked_files() -> None:
    """A fresh clone on another machine must not inherit this one's filesystem layout.

    Written for the portability pass: `/opt/android-studio/jbr` was the *only* JAVA_HOME fallback in
    the Makefile and four scripts, so on macOS or a CI runner Gradle failed with a Java-version error
    naming neither JAVA_HOME nor the file that set it. They now share `scripts/lib/java_home.sh`,
    which probes PATH first — and this guard is what stops the literal spreading back out.

    Verified able to fail: reverting any of those five files to its hardcoded form turns this red.
    """
    violations: list[str] = []
    for rel in _tracked_files():
        if not rel.endswith(_MACHINE_PATH_SUFFIXES) or rel in _MACHINE_PATH_ALLOWLIST:
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in _MACHINE_PATH_PATTERNS:
                if pattern.search(line):
                    violations.append(f"  {rel}:{lineno}: {line.strip()[:110]}")

    assert not violations, (
        "machine-specific absolute paths in tracked files — a fresh clone on another machine cannot "
        "use these. Probe for the value (see scripts/lib/java_home.sh) or take it from the "
        "environment; add to _MACHINE_PATH_ALLOWLIST only for a documented last-resort fallback:\n"
        + "\n".join(violations)
    )


def test_machine_path_allowlist_entries_still_exist() -> None:
    """A stale allow-list entry must fail, so this ratchet cannot silently rot."""
    for rel in _MACHINE_PATH_ALLOWLIST:
        assert (REPO_ROOT / rel).is_file(), (
            f"_MACHINE_PATH_ALLOWLIST names {rel}, which no longer exists — drop the entry"
        )


def test_the_machine_path_guard_can_actually_fail() -> None:
    """The guard's own regexes, exercised against a literal. A grep guard whose pattern silently
    stopped matching passes forever and asserts nothing — the failure mode this repo calls
    "scanning the void"."""
    samples = ("/home/someone/Projects/x", "/Users/Someone/Projects/x", "/opt/android-studio/jbr")
    for sample in samples:
        assert any(p.search(sample) for p in _MACHINE_PATH_PATTERNS), f"no pattern matches {sample}"
    assert not any(p.search("~/Android/Sdk") for p in _MACHINE_PATH_PATTERNS)
    assert not any(p.search("$HOME/Android/Sdk") for p in _MACHINE_PATH_PATTERNS)


# --- internal plan identifiers must not reach a reader ------------------------------------------

#: `#12`, `#37` etc. — the numbering of this project's internal implementation plans. They are
#: meaningless to anyone outside the build, and `agent_docs/` (where they were defined) is no longer
#: even in the repository, so every one of them is now a reference to nothing.
_PLAN_ID = re.compile(r"(^|[^0-9A-Za-z_&])#[0-9]{1,2}\b")

#: Paths into the untracked planning material. These are dead references by construction.
_PLAN_PATH = re.compile(r"agent_docs/|IMPLEMENTATION_ORDER|\d\d_code_plans/")

#: Areas that are CLEAN and must stay clean. A hit here fails the build.
#:
#: These were chosen because they are what a reader outside this project actually encounters: the
#: shipped documentation, the sample app that an integrator reads as the reference consumer, and —
#: most importantly — anything a *user* can see at runtime.
#: 2026-08-17, second pass: the first pass guarded `docs/` and the app and left everything else to
#: the ratchet below — which covers `src`, `android/.../MobileTransformers/src` and `scripts` and
#: therefore covered NONE of the repo root, the CI workflows, `config/` or `examples/`. The owner
#: found `#35` still sitting in `pyproject.toml`. A ratchet only holds the ground it is pointed at,
#: and "everything else" was nobody's ground. `"."` closes that: it scans tracked files at the repo
#: ROOT plus any directory not otherwise listed, so a new top-level file is covered by default
#: rather than by someone remembering to add it here.
_PLAN_ID_CLEAN_AREAS = (
    "docs",
    "android/MobileTransformers/MobileTransformersApp/src/main",
    ".github",
    "config",
    "examples",
    ".",
)

#: repo-relative dir -> plan-identifier hits currently tolerated. ENTRIES MAY ONLY SHRINK.
#:
#: These are internal implementation comments. The 2026-08-17 pass cleaned the reader-facing surfaces
#: above and stopped there, deliberately: mechanically deleting `(#8 schema)` from prose leaves
#: `(schema)`, and a whole-tree regex rewrite of comments was attempted on 2026-08-16 and had to be
#: fully reverted — a test suite cannot validate a rewrite of comments, because comments do not affect
#: behaviour, so every gate stayed green over 300 files of mangled prose. Each of these is a sentence
#: that has to be re-written by a human, not a substitution.
#:
#: The ratchet's job is that the numbers only fall.
_PLAN_ID_ALLOWLIST: dict[str, int] = {
    "src": 175,
    "android/MobileTransformers/MobileTransformers/src": 383,
    "scripts": 18,
}

#: File types this guard reads.
#:
#: The 2026-08-17 first pass omitted `.toml`, `.yml`, `.yaml`, `.json`, `.properties` and `.cff` — so
#: `pyproject.toml` and all three CI workflows were **structurally invisible** to it. The guard read
#: as if it covered the repo and did not, which is worse than an absent guard: the owner found `#35`
#: in `pyproject.toml` by eye, on a tree this test had just passed.
#:
#: The lesson is the one this repo keeps relearning about ratchets — a scan is only as wide as its
#: filter, and a filter that silently excludes is indistinguishable from a clean result. Any config
#: format that carries comments belongs here.
_PLAN_ID_SUFFIXES = (
    ".py",
    ".kt",
    ".kts",
    ".java",
    ".cpp",
    ".h",
    ".hpp",
    ".md",
    ".sh",
    ".xml",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".properties",
    ".cff",
    ".cmake",
    "Makefile",
    "CMakeLists.txt",
)


def _scan_tracked(area: str, pattern: re.Pattern[str]) -> list[str]:
    """`pattern` hits in TRACKED files under `area`, as `path:line: text`.

    Tracked-only, via `git ls-files`, and the reason is a bug this guard shipped with for one run: an
    `rglob` sweep read `android/**/build/tmp/kapt3/stubs/**/*.java` — generated Kotlin stubs that
    copy every KDoc comment verbatim — and reported ~60 phantom hits in build output nobody can edit.
    Anything gitignored is out by construction, which is the same choice `test_docs.py` documents.
    """
    prefix = "" if area == "." else f"{area}/"
    hits: list[str] = []
    for rel in _tracked_files():
        if not rel.startswith(prefix) or not rel.endswith(_PLAN_ID_SUFFIXES):
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"  {rel}:{lineno}: {line.strip()[:100]}")
    return hits


#: Trees whose plan-identifier debt is ACCEPTED and tracked by `_PLAN_ID_ALLOWLIST` instead, plus the
#: gate spikes. Excluded from the repo-root sweep so the two do not double-report the same lines.
#:
#: `spikes/` is deliberately out of both: they are recorded gate experiments, referenced from the
#: operational notes by the identifiers they were run under, and rewriting that prose would make the
#: record harder to follow rather than easier.
_PLAN_ID_ROOT_EXCLUDES = (
    "src/",
    "android/",
    "scripts/",
    "docs/",
    "tests/",
    "spikes/",
    "research/",
    ".github/",
    "config/",
    "examples/",
)


def _plan_id_hits(area: str) -> list[str]:
    hits = _scan_tracked(area, _PLAN_ID)
    if area == ".":
        hits = [h for h in hits if not h.strip().startswith(_PLAN_ID_ROOT_EXCLUDES)]
    return hits


@pytest.mark.parametrize("area", _PLAN_ID_CLEAN_AREAS)
def test_no_plan_identifiers_in_reader_facing_areas(area: str) -> None:
    """The shipped docs and the sample app must not cite this project's internal plan numbering."""
    if not (REPO_ROOT / area).is_dir():
        pytest.skip(f"{area} not present in this checkout")
    hits = _plan_id_hits(area)
    assert not hits, (
        f"internal plan identifiers in {area} — these mean nothing to a reader and point at "
        "`agent_docs/`, which is not in the repository. Rewrite the sentence without the "
        "identifier:\n" + "\n".join(hits)
    )


def test_no_plan_identifiers_in_user_visible_strings() -> None:
    """The strongest form of this rule: a *user* must never see one.

    Three did reach here — one of them inside `ModelNotInstalledException`'s message, which is a
    string an integrator hits on their first wrong call. Scoped to double-quoted Kotlin literals in
    `main` source sets, which is where every user-facing message in the SDK lives.
    """
    android = REPO_ROOT / "android"
    if not android.is_dir():
        pytest.skip("Android sources not present in this checkout")

    literal = re.compile(r'"[^"\n]*"')
    hits: list[str] = []
    for path in sorted(android.rglob("src/main/**/*.kt")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in literal.findall(line):
                # `"#$rank"` and friends are Kotlin string templates rendering a real number.
                if _PLAN_ID.search(match) and "$" not in match:
                    hits.append(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {match[:100]}")
    assert not hits, "a plan identifier is reachable in a user-visible string:\n" + "\n".join(hits)


def test_no_dead_references_to_the_untracked_planning_material() -> None:
    """`agent_docs/` was untracked on 2026-08-17, so any pointer to it from shipped code is dead.

    Covers the plan directories too (`00_code_plans/…`, `IMPLEMENTATION_ORDER`): they live in the same
    untracked tree, so a comment citing one sends a reader to an address that does not exist in the
    repository they cloned.

    `tests/` is exempt because several of its comments *explain* the untracking and necessarily name
    the directory — the same unavoidable self-reference the machine-path guard has.
    """
    hits: list[str] = []
    for area in ("src", "docs", "scripts", "android", "."):
        if area == ".":
            # CHANGELOG.md is exempt: it is a historical record, and the 0.2.0 entry that *announces*
            # the untracking cannot describe it without naming the directory. Its one genuinely dead
            # pointer (an older entry citing `agent_docs/HANDOFF.md` for a parameter count) was
            # removed rather than exempted.
            hits += [
                h for h in _scan_tracked(".", _PLAN_PATH) if h.strip().startswith(("README.md", "Makefile"))
            ]
        elif (REPO_ROOT / area).is_dir():
            hits += _scan_tracked(area, _PLAN_PATH)
    assert not hits, (
        "reference(s) to the untracked planning material from shipped files. That tree is not in the "
        "repository, so these point at nothing for anyone who clones it:\n" + "\n".join(hits)
    )


def test_plan_identifier_debt_only_shrinks() -> None:
    """Internal comments still carry them; the count may only fall. See `_PLAN_ID_ALLOWLIST`."""
    for area, allowed in _PLAN_ID_ALLOWLIST.items():
        if not (REPO_ROOT / area).is_dir():
            continue
        actual = len(_plan_id_hits(area))
        assert actual <= allowed, (
            f"{area}: {actual} plan identifiers, allowance {allowed} — a NEW one was introduced. "
            "Write the comment without the identifier."
        )
        assert actual == allowed, (
            f"{area}: down to {actual} plan identifiers (allowance {allowed}) — lower "
            "_PLAN_ID_ALLOWLIST so the ratchet holds"
        )
