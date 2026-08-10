"""One resolver for every stage path in a package. No consumer appends a stage name to a string.

## Why this exists

A package has **two** on-disk layouts, and until now both were built by string concatenation at every
call site:

===================  ==========================================  ==========================
layout               shape                                        who produces it
===================  ==========================================  ==========================
hub package          ``variants/<variantId>/{train,inference,     ``export/pipeline.py``
                     embedding}`` + ``shared/tokenizer``
device cache (flat)  ``<cacheDir>/<repoId>/{train,inference,      ``ModelPackageInstaller``,
                     embedding,tokenizer}``                        ``scripts/device_package.sh``
===================  ==========================================  ==========================

The manifest has always *declared* the hub layout — ``variant.paths`` — but exactly one consumer in the
whole repo read it (``cli/federated.py``); roughly twenty others spelled the join by hand, in four
languages. The #35 simulation lost a cycle to precisely this: the client looked for ``<package>/train/``
(the cache layout) in a hub package, and ORT reported ``INVALID_ARGUMENT : Invalid fd was supplied: -1``,
naming no file.

This is the layer-identity problem in a second namespace. The fix is the same one ``cpp/layer_name.h``
applied there: **one place that knows the spelling**, and a guard that keeps it that way.

## The rule

Ask a :class:`PackagePaths` for a stage. Never write ``dir / "train"``.

Mirrored in Kotlin by ``packages/PackagePaths.kt``; the two must agree, because the same package is
read by both.

**C++ deliberately has no mirror.** It never resolves a stage — Kotlin hands it an already-resolved
directory over JNI, and its joins (``inference_dir + "/weight_handoff_map.json"``) append a *filename*
to a resolved directory, which is not this defect. Adding a third copy for symmetry would create the
very duplication this module exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mobiletransformers.exceptions import ManifestError

#: Stage keys as they appear in the manifest's ``paths`` map. These are wire names.
STAGE_INFERENCE = "inference"
STAGE_TRAIN = "train"
STAGE_EMBEDDING = "embedding"
STAGE_TOKENIZER = "tokenizer"

#: Every stage a package can declare, in a deterministic order.
STAGES: tuple[str, ...] = (STAGE_INFERENCE, STAGE_TRAIN, STAGE_EMBEDDING, STAGE_TOKENIZER)

#: Directory name each stage takes in the FLAT device-cache layout.
#:
#: Note ``embedding`` keeps its name while ``tokenizer`` moves from ``shared/tokenizer`` to a sibling —
#: the cache layout is not simply the hub layout with the ``variants/<id>/`` prefix removed, which is
#: exactly why open-coding it in nine places produced two different answers.
_CACHE_DIRNAMES: dict[str, str] = {
    STAGE_INFERENCE: "inference",
    STAGE_TRAIN: "train",
    STAGE_EMBEDDING: "embedding",
    STAGE_TOKENIZER: "tokenizer",
}

#: Filename of the weight handoff map inside the inference stage.
WEIGHT_HANDOFF_FILENAME = "weight_handoff_map.json"


@dataclass(frozen=True)
class PackagePaths:
    """Resolved absolute stage directories for one package in one layout.

    Build with :meth:`for_hub` or :meth:`for_cache`; do not construct directly unless you are writing a
    third layout, in which case add a factory here rather than joining strings at the call site.
    """

    root: Path
    #: Stage key -> absolute directory. Only stages the layout actually declares are present.
    stages: dict[str, Path]
    #: Which layout produced this, for error messages that need to say so.
    layout: str

    @classmethod
    def for_hub(cls, package_dir: str | Path, variant: object) -> PackagePaths:
        """Resolve against a hub package using the variant's **declared** ``paths``.

        :param variant: a ``SelectedVariant`` (anything exposing a ``paths`` mapping). The manifest is
            the source of truth here — a variant may legitimately place a stage somewhere this module
            would not have guessed, and re-deriving ``variants/<id>/<stage>`` would silently ignore it.
        """
        package_dir = Path(package_dir)
        declared = getattr(variant, "paths", None)
        if not isinstance(declared, dict):
            raise ManifestError(
                "variant declares no `paths` map; a package built before the manifest carried per-variant "
                "paths cannot be resolved — re-export it."
            )
        stages = {stage: package_dir / rel for stage, rel in declared.items() if isinstance(rel, str) and rel}
        return cls(root=package_dir, stages=stages, layout="hub")

    @classmethod
    def for_cache(cls, cache_dir: str | Path, repo_id: str) -> PackagePaths:
        """Resolve against the FLAT on-device cache layout.

        ``repo_id`` must already be sanitized (``/`` -> ``__``) if it came from a hub id; that mapping is
        owned by ``hub/package_format.py::sanitize_repo_id`` and mirrored in Kotlin, and is deliberately
        not repeated here.
        """
        base = Path(cache_dir) / repo_id
        return cls(
            root=base,
            stages={stage: base / name for stage, name in _CACHE_DIRNAMES.items()},
            layout="cache",
        )

    # -- accessors ----------------------------------------------------------

    def stage(self, name: str) -> Path:
        """The directory for ``name``, or :class:`ManifestError` naming what is available.

        Fails closed rather than returning a plausible path that does not exist: a silently-wrong stage
        directory surfaces later as an unrelated-looking IO error, which is the failure mode this module
        was written to end.
        """
        if name not in STAGES:
            raise ManifestError(f"unknown stage {name!r}; known stages are {list(STAGES)}")
        try:
            return self.stages[name]
        except KeyError:
            raise ManifestError(
                f"this {self.layout} package does not declare a {name!r} stage "
                f"(declared: {sorted(self.stages)})"
            ) from None

    @property
    def inference(self) -> Path:
        return self.stage(STAGE_INFERENCE)

    @property
    def train(self) -> Path:
        return self.stage(STAGE_TRAIN)

    @property
    def embedding(self) -> Path:
        return self.stage(STAGE_EMBEDDING)

    @property
    def tokenizer(self) -> Path:
        return self.stage(STAGE_TOKENIZER)

    @property
    def weight_handoff(self) -> Path:
        """The handoff map, which lives inside the inference stage in both layouts."""
        return self.inference / WEIGHT_HANDOFF_FILENAME

    def has(self, name: str) -> bool:
        """Whether the layout declares ``name`` at all (says nothing about what is on disk)."""
        return name in self.stages
