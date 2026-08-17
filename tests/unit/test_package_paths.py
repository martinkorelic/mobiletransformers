"""The two package layouts resolve from one place, and an undeclared stage fails closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from mobiletransformers.artifacts.package_paths import STAGES, PackagePaths
from mobiletransformers.exceptions import ManifestError


class _Variant:
    """Stand-in for `SelectedVariant` — the resolver only needs `paths`."""

    def __init__(self, paths: dict[str, str]) -> None:
        self.paths = paths


HUB_PATHS = {
    "inference": "variants/cpu-int4/inference",
    "train": "variants/cpu-int4/train",
    "embedding": "variants/cpu-int4/embedding",
    "tokenizer": "shared/tokenizer",
}


def test_hub_layout_uses_the_manifests_declared_paths() -> None:
    paths = PackagePaths.for_hub("/pkg", _Variant(HUB_PATHS))

    assert paths.train == Path("/pkg/variants/cpu-int4/train")
    assert paths.inference == Path("/pkg/variants/cpu-int4/inference")
    # The tokenizer is SHARED across variants — it is not under variants/<id>/. Re-deriving the hub
    # layout as "variants/<id>/<stage>" would get this one wrong, which is why the manifest decides.
    assert paths.tokenizer == Path("/pkg/shared/tokenizer")


def test_hub_layout_honours_a_variant_that_places_a_stage_unusually() -> None:
    # The manifest is the source of truth, not a convention this module re-implements.
    odd = dict(HUB_PATHS, train="somewhere/else/train")

    assert PackagePaths.for_hub("/pkg", _Variant(odd)).train == Path("/pkg/somewhere/else/train")


def test_cache_layout_is_flat_and_declares_every_stage() -> None:
    paths = PackagePaths.for_cache("/cache", "org__model")

    assert paths.train == Path("/cache/org__model/train")
    assert paths.inference == Path("/cache/org__model/inference")
    assert paths.embedding == Path("/cache/org__model/embedding")
    # Flat: the tokenizer is a sibling here, NOT under shared/. This is the difference that made the
    # #35 simulation look for <package>/train/ in a hub package and get "Invalid fd was supplied: -1".
    assert paths.tokenizer == Path("/cache/org__model/tokenizer")
    assert all(paths.has(stage) for stage in STAGES)


def test_the_two_layouts_disagree_which_is_the_whole_point() -> None:
    hub = PackagePaths.for_hub("/pkg", _Variant(HUB_PATHS))
    cache = PackagePaths.for_cache("/pkg", "model")

    assert hub.train != cache.train


def test_weight_handoff_sits_inside_inference_in_both_layouts() -> None:
    hub = PackagePaths.for_hub("/pkg", _Variant(HUB_PATHS))
    cache = PackagePaths.for_cache("/cache", "model")

    assert hub.weight_handoff == Path("/pkg/variants/cpu-int4/inference/weight_handoff_map.json")
    assert cache.weight_handoff == Path("/cache/model/inference/weight_handoff_map.json")


def test_an_undeclared_stage_fails_closed_naming_what_exists() -> None:
    paths = PackagePaths.for_hub("/pkg", _Variant({"inference": "variants/v/inference"}))

    assert not paths.has("train")
    with pytest.raises(ManifestError) as excinfo:
        _ = paths.train
    # The message must name the missing stage AND what is available; a bare KeyError sends the reader
    # hunting through four languages' worth of path joins.
    assert "train" in str(excinfo.value)
    assert "inference" in str(excinfo.value)


def test_an_unknown_stage_name_is_rejected_rather_than_silently_missing() -> None:
    paths = PackagePaths.for_cache("/cache", "model")

    with pytest.raises(ManifestError, match="unknown stage"):
        paths.stage("trian")  # typo, not a legitimately absent stage


def test_a_variant_without_paths_fails_closed_telling_you_to_re_export() -> None:
    class _Old:
        pass

    with pytest.raises(ManifestError, match="re-export"):
        PackagePaths.for_hub("/pkg", _Old())
