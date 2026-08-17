#!/usr/bin/env python3
"""Upload the gitignored build artifacts to the Hub, so a fresh clone can provision itself.

    scripts/publish_build_artifacts.py --dry-run     # verify hashes, print the plan, upload nothing
    scripts/publish_build_artifacts.py               # verify, then upload what is missing
    scripts/publish_build_artifacts.py --force       # re-upload even if the remote already matches

WHY THIS EXISTS. ~956 MB of prebuilt binaries and one source-built wheel cannot live in git, so a
`git clone` produces a tree that cannot build the Android SDK at all. `scripts/fetch_native_deps.sh`
is the consumer side of that problem and has been complete for a while; it was waiting on somewhere
to fetch *from*. This is that somewhere.

WHY A DATASET REPO. It needs to be reachable by `curl -fL` with no credentials and no CLI, because
it sits before the first build on a machine that has nothing. An anonymous
`https://huggingface.co/datasets/<id>/resolve/main/<file>` request answers 302-to-CDN then 200,
which is exactly what the fetch script already follows. A Storage Bucket would be the more natural
"pile of build outputs" home, but its documented access paths are the `hf` CLI, the Python API and
an S3-compatible endpoint — none of which a bootstrap script should have to depend on.

THE HASHES ARE CHECKED BEFORE UPLOAD, NOT AFTER. `third_party/android/manifest.json` and
`third_party/onnxruntime/manifest.json` are what `fetch_native_deps.sh` verifies downloads against,
so a file whose local bytes do not match its recorded hash would be published as a permanently
broken download — every consumer would fail the checksum and no amount of retrying would help. The
manifests are the contract; this refuses to publish anything that already violates it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mobiletransformers.config.settings import get_settings  # noqa: E402

#: The dataset repo the manifest's `baseUrl` points at. Created by hand, deliberately: making a
#: public repo is an outward-facing act and not something a script should do as a side effect.
DEFAULT_REPO = "mobiletransformers/build-artifacts"

ANDROID_MANIFEST = REPO_ROOT / "third_party" / "android" / "manifest.json"
ORT_MANIFEST = REPO_ROOT / "third_party" / "onnxruntime" / "manifest.json"


@dataclass(frozen=True)
class Artifact:
    """One file to publish, with the hash its own manifest already claims for it."""

    path: Path
    sha256: str
    size: int
    required: bool
    note: str

    @property
    def name(self) -> str:
        return self.path.name


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def collect_artifacts() -> list[Artifact]:
    """Everything to publish, read from the two manifests rather than hardcoded here.

    Hardcoding the filenames would put a third copy of them in the tree, and the version is in each
    name — so a 0.3.0 bundle would upload under a 0.2.0 name the day someone forgot this file.
    """
    android = _load_json(ANDROID_MANIFEST)
    ort = _load_json(ORT_MANIFEST)

    artifacts = [
        Artifact(
            path=REPO_ROOT / "build" / "dist" / bundle["filename"],
            sha256=bundle["sha256"],
            size=bundle["size"],
            required=bundle["required"],
            note=bundle["note"],
        )
        for bundle in android["bundles"]
    ]

    wheel = ort["wheel"]
    artifacts.append(
        Artifact(
            path=REPO_ROOT / "third_party" / "wheels" / wheel["filename"],
            sha256=wheel["sha256"],
            size=int(wheel.get("size") or 0),
            required=False,
            note="Source-built ONNX Runtime Training wheel (cp312, linux_x86_64). Only an export "
            "that produces a TRAINING stage needs it; inference-only exports and the whole "
            "Android side do not.",
        )
    )
    return artifacts


def sha256_of(path: Path) -> str:
    """Streamed, because one of these is 632 MB and `read_bytes()` would hold it all."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(artifacts: list[Artifact]) -> tuple[list[Artifact], list[str]]:
    """Split into (publishable, problems). A missing OPTIONAL artifact is not a problem."""
    ready: list[Artifact] = []
    problems: list[str] = []

    for art in artifacts:
        if not art.path.is_file():
            message = f"{art.name}: not found at {art.path.relative_to(REPO_ROOT)}"
            if art.required:
                problems.append(message)
            else:
                print(f"  skip    {art.name} — not present locally (optional)")
            continue

        actual = sha256_of(art.path)
        if actual != art.sha256:
            problems.append(
                f"{art.name}: sha256 mismatch\n"
                f"           manifest {art.sha256}\n"
                f"           actual   {actual}"
            )
            continue

        print(f"  ok      {art.name} ({art.path.stat().st_size / 1e6:.0f} MB)")
        ready.append(art)

    return ready, problems


DATASET_CARD = """\
---
license: other
tags:
  - build-artifacts
  - not-a-dataset
---

# MobileTransformers — build artifacts

**This is not a dataset.** It is the set of build inputs that cannot live in git, published here so
that a `git clone` of
[MobileTransformers](https://github.com/martinkorelic/mobiletransformers) can provision itself.

Nothing here is downloaded by the Android app or by any model package. These files are consumed by
one script, at development time:

```bash
scripts/fetch_native_deps.sh              # the natives bundle — required to build the Android SDK
TRAINING=1 scripts/fetch_native_deps.sh   # + the ORT-training wheel, for exporting a training stage
SYMBOLS=1  scripts/fetch_native_deps.sh   # + unstripped binaries, for symbolicating a native crash
```

That script reads `third_party/android/manifest.json`, downloads what is missing, checks the archive
sha256, unpacks it, and then checks every unpacked file's sha256 individually. Both halves matter: the
archive hash proves the download, the per-file hashes prove the unpack, and a half-populated
`jniLibs/` is the failure mode that produces a linker error naming a symbol rather than a file.

## Contents

| file | size | needed for |
| --- | --- | --- |
| `mobiletransformers-natives-0.2.0-arm64-v8a.tar.zst` | 63 MB | **Required** to build the Android SDK: ONNX Runtime (training build), the GenAI engine, the tokenizer static libs, and vendored headers. |
| `onnxruntime_training-1.23.0+cpu-cp312-cp312-linux_x86_64.whl` | 632 MB | Only to export a **training** stage. Source-built, cp312/linux_x86_64 only — it is not on PyPI. |
| `mobiletransformers-natives-0.2.0-arm64-v8a-debug-symbols.tar.zst` | 261 MB | Optional. The unstripped originals of the shipped `.so` files. Android's build strips them at packaging, so these cost nothing at runtime and are the only way to read a native stack trace. |

Every sha256, size, provenance and role is recorded in
[`third_party/android/manifest.json`](https://github.com/martinkorelic/mobiletransformers/blob/main/third_party/android/manifest.json)
and
[`third_party/onnxruntime/manifest.json`](https://github.com/martinkorelic/mobiletransformers/blob/main/third_party/onnxruntime/manifest.json).
Verify by hand with `sha256sum` if you prefer; the fetch script does it for you either way.

## Licensing

These are builds of third-party projects — ONNX Runtime, onnxruntime-genai, tokenizers-cpp, protobuf
— each under its own upstream licence. See
[`THIRD_PARTY_NOTICES.md`](https://github.com/martinkorelic/mobiletransformers/blob/main/THIRD_PARTY_NOTICES.md).
The `license: other` tag above reflects that this repo is a mixed bundle of upstream artifacts rather
than a single licensed work.

## arm64-v8a only

There is no x86_64 build, so the SDK does not run on a standard Android emulator. `libonnxruntime.so`
and the tokenizer archives were never built for it; restoring x86_64 means building ONNX Runtime
Training and tokenizers-cpp for that ABI first.
"""


def main(argv: list[str] | None = None, *, uploader: Callable[..., Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"dataset repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--dry-run", action="store_true", help="verify and print the plan; upload nothing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="upload every artifact even if the remote already has a file of that name",
    )
    parser.add_argument("--skip-card", action="store_true", help="do not write the dataset card")
    args = parser.parse_args(argv)

    print(f"verifying local artifacts against the manifests ({ANDROID_MANIFEST.name}, {ORT_MANIFEST.name})")
    ready, problems = verify(collect_artifacts())

    if problems:
        print("\nrefusing to publish:\n  " + "\n  ".join(problems), file=sys.stderr)
        print(
            "\nA file whose bytes do not match its manifest hash would be published as a "
            "permanently broken download — every consumer fails the checksum and retrying cannot "
            "help. Fix the file or the manifest before publishing.",
            file=sys.stderr,
        )
        return 1
    if not ready:
        print("nothing to publish", file=sys.stderr)
        return 1

    total = sum(a.path.stat().st_size for a in ready)
    print(f"\n{len(ready)} artifact(s), {total / 1e6:.0f} MB total -> {args.repo}")

    if args.dry_run:
        for art in ready:
            print(f"  [dry-run] would upload {art.name}")
        print("[dry-run] nothing was uploaded")
        return 0

    # Explicit token, and the ORG one. `huggingface_hub` falls back to $HF_TOKEN and then to the
    # cached CLI login, so an org upload with no token argument can authenticate as the wrong
    # identity and look exactly like success.
    token = get_settings().require_org_token()

    if uploader is None:
        from huggingface_hub import create_repo, upload_file

        # The repo is expected to exist (it is created by hand — see DEFAULT_REPO). exist_ok makes
        # a re-run a no-op rather than an error, and covers a fresh org bootstrapping itself.
        create_repo(args.repo, repo_type="dataset", exist_ok=True, token=token)
        uploader = upload_file

    if not args.skip_card:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            card = Path(tmp) / "README.md"
            card.write_text(DATASET_CARD, encoding="utf-8")
            uploader(
                path_or_fileobj=str(card),
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="dataset",
                token=token,
                commit_message="Describe what these artifacts are and who consumes them",
            )
        print("  uploaded README.md (dataset card)")

    for art in ready:
        print(f"  uploading {art.name} ({art.path.stat().st_size / 1e6:.0f} MB)…", flush=True)
        uploader(
            path_or_fileobj=str(art.path),
            path_in_repo=art.name,
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
            commit_message=f"Add {art.name}",
        )
        print(f"  uploaded  {art.name}")

    print(f"\npublished to https://huggingface.co/datasets/{args.repo}")
    print("Set `baseUrl` in third_party/android/manifest.json to:")
    print(f"  https://huggingface.co/datasets/{args.repo}/resolve/main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
