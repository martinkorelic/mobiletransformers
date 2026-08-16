"""Unit tests for HandoffMap.validate() invariants + the canonical check_compat (#8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mobiletransformers.artifacts.handoff_map import (
    HANDOFF_MAP_READER_VERSION,
    HandoffEntry,
    HandoffMap,
    TrainableTensorCodec,
)
from mobiletransformers.artifacts.versioning import SchemaVersionError, check_compat
from mobiletransformers.config.constants import HandoffMode
from mobiletransformers.exceptions import HandoffError

_CASES = json.loads((Path(__file__).parent.parent / "fixtures" / "check_compat_cases.json").read_text())[
    "cases"
]


def _good_entry(idx: int = 0) -> HandoffEntry:
    name = f"model.layers.{idx}.attn.q_proj.MatMul.weight"
    return HandoffEntry(
        training_base_layer_name=f"backbone.model.layers.{idx}.self_attn.q_proj.base_layer",
        dtype="float16",
        shape=(4096, 4096),
        merged_tensor_names={"weight": name},
        inference_initializer_names={"weight": name},
        external_data_location={"weight": name + ".bin"},
    )


# --- check_compat (shared cross-language fixture) --------------------------------------------------
@pytest.mark.parametrize("case", _CASES, ids=[c["why"] for c in _CASES])
def test_check_compat_matches_shared_fixture(case: dict) -> None:
    if case["expect"] == "accept":
        check_compat(case["doc"], case["minReader"], case["reader"])  # must not raise
    else:
        with pytest.raises(SchemaVersionError):
            check_compat(case["doc"], case["minReader"], case["reader"])


def test_reader_version_tracks_the_schema_minor() -> None:
    """1.1 added `adapterDtypes`/`adapterShapes`, which is ADDITIVE.

    `minReaderVersion` deliberately stays 1.0: a 1.0 reader ignores unknown fields by the canonical
    rule and keeps working, and maps written at 1.0 still load (they simply cannot describe their
    adapter factors, which `codec_tensor_specs` reports as a fail-closed error rather than a guess).
    """
    assert HANDOFF_MAP_READER_VERSION == "1.1"
    assert HandoffMap().schema_version == "1.1"
    assert HandoffMap().min_reader_version == "1.0"


# --- validate() invariants ------------------------------------------------------------------------
def test_valid_external_initializer_map_passes() -> None:
    HandoffMap(entries=[_good_entry(0), _good_entry(1)]).validate()


def test_merged_must_equal_inference_name() -> None:
    e = _good_entry()
    e.merged_tensor_names = {"weight": "model.layers.0.attn.q_proj.MatMul.WRONG"}
    with pytest.raises(HandoffError, match="mergedTensorNames"):
        HandoffMap(entries=[e]).validate()


def test_quantized_scale_from_base_layer_name_is_rejected() -> None:
    # The documented bug: scale derived from base_layer_name instead of the observed inference init.
    seed = "model.layers.0.attn.q_proj.MatMul"
    e = HandoffEntry(
        training_base_layer_name="backbone.model.layers.0.self_attn.q_proj.base_layer",
        dtype="int4",
        shape=(4096, 2048),
        merged_tensor_names={
            "weight_quantized": f"{seed}.qweight",
            "scale": f"{seed}.scales",
            "zero_point": f"{seed}.qzeros",
        },
        inference_initializer_names={
            "weight_quantized": f"{seed}.qweight",
            "scale": f"{seed}.scales",
            "zero_point": f"{seed}.qzeros",
        },
        external_data_location={
            "weight_quantized": f"{seed}.qweight.bin",
            "scale": f"{seed}.scales.bin",
            "zero_point": f"{seed}.qzeros.bin",
        },
        quantization={
            "weightQuantizedName": f"{seed}.qweight",
            # BUG: derived from base_layer_name, not the observed inference init
            "scaleName": "backbone.model.layers.0.self_attn.q_proj.base_layer.weight_scale",
            "zeroPointName": f"{seed}.qzeros",
        },
    )
    with pytest.raises(HandoffError, match="not derived from base_layer_name"):
        HandoffMap(entries=[e]).validate()


# --- per-role on-disk dtype/shape (the raw-bytes device loader's only source) ----------------------
def test_per_role_dtype_shape_round_trip() -> None:
    e = _good_entry()
    e.tensor_dtypes = {"weight": "float16"}
    e.tensor_shapes = {"weight": (4096, 4096)}
    restored = HandoffEntry.from_dict(e.to_dict())
    assert restored.tensor_dtypes == {"weight": "float16"}
    assert restored.tensor_shapes == {"weight": (4096, 4096)}
    assert restored.dtype_for("weight") == "float16"
    assert restored.shape_for("weight") == (4096, 4096)


def test_per_role_lookup_falls_back_to_entry_level() -> None:
    """Maps written before tensorDtypes/tensorShapes existed still resolve their single role."""
    e = _good_entry()
    assert not e.tensor_dtypes and not e.tensor_shapes
    assert e.dtype_for("weight") == "float16"
    assert e.shape_for("weight") == (4096, 4096)


def test_tensor_specs_report_each_roles_own_dtype_and_shape() -> None:
    """A scale tensor is not shaped like the weight it scales — reporting the weight's was wrong."""
    e = _good_entry()
    seed = "model.layers.0.attn.q_proj.MatMul"
    e.merged_tensor_names = {"weight_quantized": f"{seed}.qweight", "scale": f"{seed}.scales"}
    e.inference_initializer_names = dict(e.merged_tensor_names)
    e.external_data_location = {r: f"{n}.bin" for r, n in e.merged_tensor_names.items()}
    e.tensor_dtypes = {"weight_quantized": "uint8", "scale": "float16"}
    e.tensor_shapes = {"weight_quantized": (4096, 2048), "scale": (4096, 32)}

    by_role = {spec.role: spec for spec in e.tensor_specs()}
    assert (by_role["weight_quantized"].dtype, by_role["weight_quantized"].shape) == (
        "uint8",
        (4096, 2048),
    )
    assert (by_role["scale"].dtype, by_role["scale"].shape) == ("float16", (4096, 32))


def test_role_missing_per_role_dtype_is_rejected() -> None:
    e = _good_entry()
    e.tensor_shapes = {"weight": (4096, 4096)}  # shapes declared, dtypes not
    with pytest.raises(HandoffError, match="missing tensorDtypes"):
        HandoffMap(entries=[e]).validate()


def test_quantized_entry_without_per_role_dtype_shape_is_rejected() -> None:
    """The entry-level dtype/shape describes only the weight-like role, so it cannot stand in here."""
    seed = "model.layers.0.attn.q_proj.MatMul"
    e = _good_entry()
    e.merged_tensor_names = {
        "weight_quantized": f"{seed}.qweight",
        "scale": f"{seed}.scales",
        "zero_point": f"{seed}.qzeros",
    }
    e.inference_initializer_names = dict(e.merged_tensor_names)
    e.external_data_location = {r: f"{n}.bin" for r, n in e.merged_tensor_names.items()}
    e.quantization = {
        "weightQuantizedName": f"{seed}.qweight",
        "scaleName": f"{seed}.scales",
        "zeroPointName": f"{seed}.qzeros",
    }
    with pytest.raises(HandoffError, match="must declare per-role tensorDtypes/tensorShapes"):
        HandoffMap(entries=[e]).validate()


def test_duplicate_external_location_rejected() -> None:
    a, b = _good_entry(0), _good_entry(1)
    b.external_data_location = dict(a.external_data_location)  # collide
    with pytest.raises(HandoffError, match="duplicate externalDataLocation"):
        HandoffMap(entries=[a, b]).validate()


def test_duplicate_inference_name_rejected() -> None:
    a, b = _good_entry(0), _good_entry(1)
    b.inference_initializer_names = dict(a.inference_initializer_names)
    b.merged_tensor_names = dict(a.merged_tensor_names)
    with pytest.raises(HandoffError, match="duplicate inferenceInitializerName"):
        HandoffMap(entries=[a, b]).validate()


def test_model_input_mode_fails_closed_v1() -> None:
    with pytest.raises(HandoffError, match="not supported"):
        HandoffMap(entries=[_good_entry()], handoff_mode=HandoffMode.MODEL_INPUT).validate()


def test_adapter_mode_fails_closed_v1() -> None:
    with pytest.raises(HandoffError, match="not supported"):
        HandoffMap(entries=[_good_entry()], handoff_mode=HandoffMode.ADAPTER).validate()


def test_unsupported_major_fails_closed() -> None:
    with pytest.raises(SchemaVersionError):
        HandoffMap(entries=[_good_entry()], schema_version="2.0").validate()


def test_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "weight_handoff_map.json"
    HandoffMap(entries=[_good_entry(0), _good_entry(1)]).save(path)
    loaded = HandoffMap.load(path)
    assert len(loaded.entries) == 2
    assert loaded.handoff_mode == HandoffMode.EXTERNAL_INITIALIZER


class _ArchSpec:
    """Minimal stand-in for the #6 architecture-registry row (only the rewrite field is read)."""

    def __init__(self, attention_module_name: str = "self_attn") -> None:
        self.attention_module_name = attention_module_name


def test_candidate_seeds_cover_both_attention_spellings():
    """Two inference exporters name the attention module differently; the seed must accept both.

    The legacy `inference/builder.py` graphs (and the `weight_merger.cpp:904` mirror) use `attn`;
    the Optimum export that #7 made the front door keeps HF-canonical `self_attn`. Seeding only the
    rewritten spelling meant no trainable tensor in an Optimum-produced package could ever be matched,
    so `export_inference_package` failed with "inference/training naming drifted" and no handoff map
    could be built for the packages the project actually ships.
    """
    seeds = TrainableTensorCodec.candidate_inference_names(
        "backbone.model.layers.0.self_attn.q_proj.base_layer", _ArchSpec()
    )
    assert "model.layers.0.attn.q_proj.MatMul" in seeds
    assert "model.layers.0.self_attn.q_proj.MatMul" in seeds


def test_canonical_name_still_returns_the_cpp_mirror_spelling():
    """`canonical_inference_name` is mirrored in C++, so its result must not change."""
    assert (
        TrainableTensorCodec.canonical_inference_name(
            "backbone.model.layers.0.self_attn.q_proj.base_layer", _ArchSpec()
        )
        == "model.layers.0.attn.q_proj.MatMul"
    )


def test_candidate_seeds_are_deduped_when_the_module_is_already_attn():
    seeds = TrainableTensorCodec.candidate_inference_names(
        "model.layers.0.attn.q_proj.base_layer", _ArchSpec(attention_module_name="attn")
    )
    assert seeds == ("model.layers.0.attn.q_proj.MatMul",)


# --- transpose policy: observed, not declared (2026-08-14) -------------------------------------


def test_transpose_policy_is_observed_from_the_adapter_and_weight_shapes() -> None:
    """The orientation of the on-disk weight must be *derived*, never defaulted.

    Regression guard for the most expensive defect this project has had. ``ObservedInit.transposed``
    was a declared field nothing ever assigned, so ``transposePolicy`` was ``no_transpose`` by
    omission on every package ever produced. The on-device merge honoured it and wrote every merged
    weight TRANSPOSED, which is invisible to shape checks (``q_proj`` is square), to element counts
    (``v_proj`` has the same count either way) and to L2/absmax (both transpose-invariant).

    The shapes below are the real ones from a SmolLM2-135M export.
    """
    from mobiletransformers.artifacts.handoff_map import (
        ALREADY_TRANSPOSED,
        NO_TRANSPOSE,
        derive_transpose_policy,
    )

    lora = {"adapter_A": (8, 576), "adapter_B": (192, 8)}
    # v_proj: B @ A is (192, 576) while the graph stores (576, 192) -> the on-disk tensor IS the
    # transpose. This is the case that was silently wrong.
    assert derive_transpose_policy((576, 192), lora) == ALREADY_TRANSPOSED
    # ... and the same factors against a weight already in merger orientation need no conversion.
    assert derive_transpose_policy((192, 576), lora) == NO_TRANSPOSE

    # Square weights genuinely cannot decide their own orientation; the function must not pretend to.
    square = {"adapter_A": (8, 576), "adapter_B": (576, 8)}
    assert derive_transpose_policy((576, 576), square) == NO_TRANSPOSE

    # Undescribed adapters -> nothing observable, keep the historical value rather than invent one.
    assert derive_transpose_policy((4, 3), {}) == NO_TRANSPOSE


def test_a_delta_that_cannot_be_added_to_its_weight_is_refused() -> None:
    """Factors whose product is neither the weight shape nor its transpose are incoherent.

    The merge would be adding tensors that cannot be added, so the map must refuse to describe the
    layer rather than emit a contract no consumer can honour.
    """
    from mobiletransformers.artifacts.handoff_map import derive_transpose_policy

    with pytest.raises(ValueError, match="neither the on-disk weight shape"):
        derive_transpose_policy((10, 20), {"adapter_A": (8, 576), "adapter_B": (192, 8)})


def test_one_package_cannot_mix_two_weight_orientations() -> None:
    """A square layer inherits the orientation the non-square layers prove; disagreement fails closed."""
    from mobiletransformers.artifacts.handoff_map import (
        ALREADY_TRANSPOSED,
        NO_TRANSPOSE,
        HandoffEntry,
        resolve_package_transpose_policy,
    )

    def entry(shape: tuple[int, int], policy: str, adapters: bool = True) -> HandoffEntry:
        return HandoffEntry(
            training_base_layer_name="l",
            dtype="float32",
            shape=shape,
            tensor_dtypes={"weight": "float32"},
            tensor_shapes={"weight": shape},
            checkpoint_names={"weight": "w"},
            adapter_dtypes={},
            adapter_shapes={"adapter_A": (8, 576), "adapter_B": (192, 8)} if adapters else {},
            merger_output_names={"weight": "merged_weight"},
            merged_tensor_names={"weight": "w"},
            inference_initializer_names={"weight": "w"},
            external_data_location={"weight": "w.bin"},
            transpose_policy=policy,
        )

    # One decidable (non-square) entry settles the package, including for the square one.
    decided = [entry((576, 192), ALREADY_TRANSPOSED), entry((576, 576), NO_TRANSPOSE)]
    assert resolve_package_transpose_policy(decided) == ALREADY_TRANSPOSED

    # Two non-square entries that disagree cannot both be right.
    with pytest.raises(ValueError, match="disagree about weight orientation"):
        resolve_package_transpose_policy(
            [entry((576, 192), ALREADY_TRANSPOSED), entry((192, 576), NO_TRANSPOSE)]
        )

    # Nothing decidable -> the historical default, not a guess.
    assert resolve_package_transpose_policy([entry((576, 576), NO_TRANSPOSE, adapters=False)]) == NO_TRANSPOSE


def test_the_derivation_agrees_with_a_real_exported_package() -> None:
    """Run the derivation over a REAL export's shapes, not a fixture the same author invented.

    The three tests above use synthetic shapes, so they prove the function is self-consistent — not
    that it describes a package this project actually produces. That gap is how the original defect
    survived: the fixtures agreed with the broken code because both said ``no_transpose``.

    This reads whatever export is on disk (skipping when there is none, so it never blocks a clean
    checkout) and asserts the derivation reaches ``already_transposed_for_inference`` — the same
    answer ``weight_merger.cpp`` independently *observes* at merge time from the tensors themselves
    (logged as ``merge orientation: transpose_for_inference=1``). Two implementations in two languages
    agreeing on real data is the assertion; a change that breaks either side breaks the agreement.
    """
    import json

    maps = sorted(Path("build").glob("**/weight_handoff_map.json"))
    if not maps:
        pytest.skip("no exported package on disk (run scripts/device_package.sh)")

    from mobiletransformers.artifacts.handoff_map import (
        ALREADY_TRANSPOSED,
        derive_transpose_policy,
    )

    # Take the first package that can actually ANSWER the question, not simply the first on disk.
    #
    # Orientation is only observable from a non-square adapted weight, and two real cases produce
    # none: an encoder whose attention projections are square (all-MiniLM-L6-v2 adapts
    # `query`/`value` at 384x384), and a package exported for inference only, whose map has no
    # entries at all. Both are perfectly good packages — they just cannot settle this question — so
    # picking `maps[0]` made the suite's result depend on which export happened to sort first.
    chosen: Path | None = None
    entries: list = []
    for candidate in maps:
        rows = json.loads(candidate.read_text()).get("entries") or []
        if any(e["shape"][0] != e["shape"][1] for e in rows):
            chosen, entries = candidate, rows
            break
    if chosen is None:
        pytest.skip(
            "no export on disk has a non-square adapted weight, so orientation is unobservable "
            f"(looked at {len(maps)} package(s))"
        )

    # The square layers (q_proj, [576,576]) genuinely cannot be decided alone and must NOT be
    # over-claimed; the non-square ones (v_proj) are what settles the package.
    decidable = {
        e["inferenceInitializerNames"]["weight"]: derive_transpose_policy(e["shape"], e["adapterShapes"])
        for e in entries
        if e["shape"][0] != e["shape"][1]
    }
    assert decidable, "export has no non-square adapted weight, so orientation is unobservable"
    assert set(decidable.values()) == {ALREADY_TRANSPOSED}, (
        f"derivation disagrees with the orientation the device merge observes: {decidable}"
    )
