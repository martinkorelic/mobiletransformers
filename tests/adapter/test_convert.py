"""#22 export + PEFT-vs-native gate + adapter card (pure Python, no torch/peft)."""

from __future__ import annotations

import pytest

from mobiletransformers.adapter.convert import materialize_peft_weights, to_peft_layout
from mobiletransformers.adapter.export import export_adapter_from_cache
from mobiletransformers.adapter.model_card import assert_required_sections, render_adapter_card
from mobiletransformers.exceptions import ExportError
from tests.adapter._helpers import TRAINABLE, make_cache

_LORA_ROLES = {"adapter_A": "l.lora_A", "adapter_B": "l.lora_B"}
_MARS_ROLES = {"shared_A": "m.shared_A", "adapter_B": "m.mars_B"}


def test_export_builds_package_from_cache(tmp_path):
    cache = make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA_ROLES)
    pkg = export_adapter_from_cache(cache)
    assert pkg.base_model_id == "org/base-model"
    assert pkg.peft_method == "lora" and pkg.rank == 8 and pkg.alpha == 16.0
    assert set(pkg.checkpoint_component_roles) == {"adapter_A", "adapter_B"}
    assert any(t.external_data_location == f"{TRAINABLE}.bin" for t in pkg.tensors)


def test_lora_with_factors_gates_to_peft(tmp_path):
    pkg = export_adapter_from_cache(
        make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA_ROLES)
    )
    layout = to_peft_layout(pkg)
    assert layout is not None
    assert layout.adapter_config["peft_type"] == "LORA"
    assert layout.adapter_config["r"] == 8 and layout.adapter_config["lora_alpha"] == 16.0
    assert layout.adapter_config["base_model_name_or_path"] == "org/base-model"


def test_lora_without_factors_falls_to_native(tmp_path):
    # Checkpoint no longer carries A/B factors (only the merged weight) -> None (native mode).
    pkg = export_adapter_from_cache(make_cache(tmp_path / "c", peft_method="lora", component_roles={}))
    assert to_peft_layout(pkg) is None


def test_mars_never_peft(tmp_path):
    pkg = export_adapter_from_cache(
        make_cache(tmp_path / "c", peft_method="mars", component_roles=_MARS_ROLES)
    )
    assert pkg.mars_optimization_level == 2
    assert to_peft_layout(pkg) is None


def test_missing_rank_alpha_falls_to_native(tmp_path):
    pkg = export_adapter_from_cache(
        make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA_ROLES, rank=None, alpha=None)
    )
    assert to_peft_layout(pkg) is None


def test_adapter_card_has_mandatory_sections(tmp_path):
    pkg = export_adapter_from_cache(
        make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA_ROLES)
    )
    card = render_adapter_card(pkg, mode="peft", base_model_license="Apache-2.0")
    assert "Privacy warning" in card
    assert "Apache-2.0" in card
    assert "lora" in card and "Rank: 8" in card
    assert_required_sections(card, pkg)  # no raise


def test_adapter_card_assert_fails_on_missing_section(tmp_path):
    pkg = export_adapter_from_cache(
        make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA_ROLES)
    )
    with pytest.raises(ExportError, match="privacy warning"):
        assert_required_sections("no disclosures here", pkg)


def test_materialize_peft_weights_writes_safetensors(tmp_path):
    """The numpy->torch->safetensors path with an injected factor reader (no ORT checkpoint needed)."""
    np = pytest.importorskip("numpy")
    st = pytest.importorskip("safetensors")  # ensures torch+safetensors present
    pytest.importorskip("torch")

    pkg = export_adapter_from_cache(
        make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA_ROLES)
    )
    layout = to_peft_layout(pkg)
    assert layout is not None

    factors = {
        "l.lora_A": np.arange(8 * 3, dtype=np.float32).reshape(8, 3),  # (rank, in_features)
        "l.lora_B": np.arange(4 * 8, dtype=np.float32).reshape(4, 8),  # (out_features, rank)
    }
    dest = tmp_path / "out"
    materialize_peft_weights(pkg, layout, str(dest), factor_reader=lambda _dir, _names: factors)

    from safetensors.numpy import load_file

    loaded = load_file(str(dest / "adapter_model.safetensors"))
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    assert set(loaded) == {f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"}
    assert np.array_equal(loaded[f"{prefix}.lora_A.weight"], factors["l.lora_A"])
    assert np.array_equal(loaded[f"{prefix}.lora_B.weight"], factors["l.lora_B"])
    _ = st


def test_materialize_peft_weights_fails_closed_on_missing_factor(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")

    pkg = export_adapter_from_cache(
        make_cache(tmp_path / "c", peft_method="lora", component_roles=_LORA_ROLES)
    )
    layout = to_peft_layout(pkg)
    assert layout is not None
    with pytest.raises(ExportError, match="missing required LoRA factor"):
        materialize_peft_weights(pkg, layout, str(tmp_path / "out"), factor_reader=lambda _d, _n: {})
