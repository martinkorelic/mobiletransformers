"""#33 self-check 3: MARS transfer onto encoder attention layers — **verified, not assumed**.

Run:  uv run --python 3.12 --group ort-training-local --no-default-groups \\
          pytest tests/integration/test_mars_encoder_transfer.py -q

Needs torch + transformers + peft; **no network and no HF token** — every model here is built from a
tiny locally-constructed config, so this is fast and deterministic.

## Why the assertions look the way they do

A silent no-op and a successful transfer are indistinguishable from the outside, which is exactly the
failure this file exists to catch. Before the fix, `peft/mars/model.py` hardcoded the Llama naming in
five places:

* the attention module was found by `isinstance(m, type(model.model.layers[0].self_attn))` —
  `BertForSequenceClassification` has no `.model.layers` at all;
* the shared outputs were read from `kwargs["hidden_states"]` — `BertAttention` passes them
  positionally;
* the projections were looked up as `q_proj`/`k_proj`/`v_proj` — BERT names them
  `query`/`key`/`value`, nested one level deeper under `attention.self`;
* `projection_type` came from `"q_proj" in target_name`, which matched nothing on an encoder, leaving
  `is_standalone=True` — i.e. MARS **silently degraded to unshared adapters**;
* `_replace_module` grouped `qkv`/`mlp` by the same decoder literals, so the shared adapter was never
  wired to the wrapped module even when one existed.

So asserting "the call returned" proves nothing. These tests assert **counts** (how many modules
actually carry a shared adapter) and, across the seam, that perturbing the *shared* parameters
changes the model's logits — which cannot happen unless the shared adapter is genuinely on the
compute graph.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="needs torch (ort-training-local / export profile)")
pytest.importorskip("transformers", reason="needs transformers")
pytest.importorskip("peft", reason="needs peft")

from peft import PeftType, get_peft_model  # noqa: E402

# Same compat shim as `export/training_export.py`: peft 0.15 renamed the PeftType -> tuner-class
# registry. Mirrored rather than imported from there, because that module pulls optimum.
try:  # pragma: no cover - one branch per peft line
    from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING  # noqa: E402
except ImportError:  # pragma: no cover
    from peft.peft_model import PEFT_TYPE_TO_TUNER_MAPPING as PEFT_TYPE_TO_MODEL_MAPPING  # noqa: E402

from mobiletransformers.peft.mars.config import MarsConfig  # noqa: E402
from mobiletransformers.peft.mars.layer import Linear as MarsLinear  # noqa: E402
from mobiletransformers.peft.mars.model import MarsModel  # noqa: E402


def _register_mars_peft_type() -> None:
    """Register MARS with peft the same way `export/training_export.py` does at import time.

    Duplicated deliberately: importing that module here would pull optimum in, and the point of this
    file is to test the PEFT wrap, not the export stack.
    """
    PeftType.MARS = "MARS"  # type: ignore[attr-defined]
    PeftType._value2member_map_["MARS"] = "MARS"
    PEFT_TYPE_TO_MODEL_MAPPING[PeftType("MARS")] = MarsModel


_register_mars_peft_type()

NUM_LAYERS = 2


def _tiny_encoder():
    """A 2-layer BERT classifier built from a config — no download, no token."""
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        intermediate_size=37,
        max_position_embeddings=64,
        num_labels=2,
    )
    torch.manual_seed(0)
    return BertForSequenceClassification(config)


def _tiny_decoder():
    """The decoder regression twin — same size, same wrap, `q_proj`/`v_proj` naming."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def _wrap(model, target_modules):
    config = MarsConfig(
        peft_type="MARS",
        r=4,
        alpha=4,
        onnx_export=True,
        target_modules=list(target_modules),
        task_type=None,
    )
    return get_peft_model(model, config, adapter_name="mars")


def _distinct_shared_adapters(model):
    """The DISTINCT `SharedAttentionAdapter` objects in the tree — one per attention block.

    Counting `hasattr(m, "shared_qkv")` would over-count: `_replace_module` also gives every wrapped
    projection a back-reference to its block's adapter (that back-reference is what makes the adapter
    shared at all), so a 2-layer model with 2 targets per layer reports 6 holders of 2 objects.
    """
    from mobiletransformers.peft.mars.layer import SharedAttentionAdapter

    by_id = {}
    for _name, module in model.named_modules():
        if isinstance(module, SharedAttentionAdapter):
            by_id[id(module)] = module
    return list(by_id.values())


def _shared_adapter_holders(model):
    """Every module carrying a `shared_qkv` reference: the anchors plus the wrapped projections."""
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "shared_qkv")]


def _wrapped_projections(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MarsLinear)]


# --- the count assertions: a no-op fails here, a returned call does not -----------------------


@pytest.mark.parametrize(
    ("build", "targets", "expected_names"),
    [
        (_tiny_encoder, ("query", "value"), {"query", "value"}),
        (_tiny_decoder, ("q_proj", "v_proj"), {"q_proj", "v_proj"}),
    ],
    ids=["encoder", "decoder-regression"],
)
def test_shared_adapter_is_attached_to_every_attention_block(build, targets, expected_names):
    """One `SharedAttentionAdapter` per attention block — counted, not assumed."""
    model = _wrap(build(), targets)

    # Exactly one shared adapter per attention block. The anchor it is attached to is the module that
    # DIRECTLY owns the projections: `self_attn` on a decoder, `attention.self` on BERT.
    adapters = _distinct_shared_adapters(model)
    assert len(adapters) == NUM_LAYERS, f"expected {NUM_LAYERS} shared QKV adapters, got {len(adapters)}"

    wrapped = _wrapped_projections(model)
    # Two targets per layer (the Wq/Wv LoRA convention).
    assert len(wrapped) == NUM_LAYERS * len(expected_names)
    assert {n.rsplit(".", 1)[-1] for n, _ in wrapped} == expected_names

    # Every wrapped projection holds a reference to one of those adapters — the anchors plus the
    # wrapped projections, and nothing else, carry `shared_qkv`.
    holders = _shared_adapter_holders(model)
    assert len(holders) == NUM_LAYERS + len(wrapped)


@pytest.mark.parametrize(
    ("build", "targets"),
    [(_tiny_encoder, ("query", "value")), (_tiny_decoder, ("q_proj", "v_proj"))],
    ids=["encoder", "decoder-regression"],
)
def test_wrapped_projections_are_shared_not_standalone(build, targets):
    """`is_standalone=False` is what distinguishes MARS from a degraded per-module LoRA.

    This is the assertion the old code would have failed on an encoder while looking entirely
    healthy: `projection_type` stayed `None`, so every module silently became standalone.
    """
    model = _wrap(build(), targets)
    wrapped = _wrapped_projections(model)
    assert wrapped, "no MARS Linear was created at all"

    for name, module in wrapped:
        assert module.projection_type in ("q", "v"), (
            f"{name}: projection_type={module.projection_type!r} — the module name never resolved to "
            "a projection role, which silently downgrades MARS to unshared adapters"
        )
        assert module.is_standalone is False, f"{name}: adapter is standalone, i.e. NOT shared"
        # The shared adapter must actually be reachable from the wrapped module, not merely exist
        # somewhere in the tree — `_replace_module` is what wires this, and it used to miss.
        assert hasattr(module, "shared_qkv"), f"{name}: no reference to the block's shared adapter"


# --- the across-the-seam assertion ------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "targets", "forward"),
    [
        (
            _tiny_encoder,
            ("query", "value"),
            lambda m: m(input_ids=torch.arange(8, dtype=torch.long).reshape(1, 8)).logits,
        ),
        (
            _tiny_decoder,
            ("q_proj", "v_proj"),
            lambda m: m(input_ids=torch.arange(8, dtype=torch.long).reshape(1, 8)).logits,
        ),
    ],
    ids=["encoder", "decoder-regression"],
)
def test_shared_parameters_are_on_the_compute_graph(build, targets, forward):
    """Perturbing only the SHARED parameters must change the logits.

    This is the seam. Counting wrapped modules proves the wrap happened; it does not prove the shared
    adapter's output ever reaches the base layer. If the transfer had degraded to standalone
    adapters, `shared_qkv` would be dead weight and these logits would be byte-identical.

    `up_project` is zero-initialised (so a freshly wrapped model is output-identical to its base),
    which is why it is filled first — otherwise the whole adapter branch multiplies to zero and the
    test would pass for the wrong reason.
    """
    model = _wrap(build(), targets)
    model.eval()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "up_project" in name:
                param.fill_(0.05)

        before = forward(model).clone()

        shared = [p for n, p in model.named_parameters() if "shared_qkv" in n]
        assert shared, "no shared_qkv parameters exist at all"
        for param in shared:
            param.add_(0.5)

        after = forward(model)

    assert not torch.allclose(before, after), (
        "perturbing the shared QKV adapter did not move the logits — the shared adapter is not on "
        "the compute graph, i.e. the MARS transfer is a silent no-op"
    )


# --- the adapter mapping the codec consumes ---------------------------------------------------


@pytest.mark.parametrize(
    ("build", "targets"),
    [(_tiny_encoder, ("query", "value")), (_tiny_decoder, ("q_proj", "v_proj"))],
    ids=["encoder", "decoder-regression"],
)
def test_adapter_mapping_has_the_same_shape_for_encoder_and_decoder(build, targets):
    """#33's spike gate: the mapping must join cleanly, and every projection must know what it shares.

    `named_modules()` surfaces a shared object exactly once, so only one projection per attention
    block finds `shared_qkv` in its own subtree; the rest get a back-pointer from the builder's
    fallback. That fallback used to test `"v_proj" in base_layer_name` — a decoder literal — so on an
    encoder BERT's `value` silently ended up with no `shared_A` at all while `query` had one, i.e.
    the two halves of a layer disagreed about whether they shared a tensor.

    The assertion is therefore that **every** entry names the shared pair, and that both projections
    of a layer name the **same** one. Encoder and decoder must agree on this shape.
    """
    from mobiletransformers.config.constants import PEFTMethod
    from mobiletransformers.config.registry.peft import build_adapter_mapping

    model = _wrap(build(), targets)
    mapping = build_adapter_mapping(PEFTMethod.MARS, model)

    assert len(mapping) == NUM_LAYERS * len(targets)
    assert all(key.endswith(".base_layer") for key in mapping), "codec keys are base-layer paths"
    # Every entry has its own up-projection and knows its shared pair.
    assert all("adapter_B" in entry for entry in mapping.values())
    assert all("shared_A" in entry for entry in mapping.values()), (
        "a wrapped projection with no `shared_A` does not know it shares a tensor — the codec would "
        "emit it as an independent adapter"
    )
    assert all("intermediate" in entry for entry in mapping.values())

    # Exactly one shared pair per layer, named identically by both of that layer's projections.
    distinct_shared = {entry["shared_A"] for entry in mapping.values()}
    assert len(distinct_shared) == NUM_LAYERS, (
        f"expected {NUM_LAYERS} distinct shared_A tensors (one per layer), got {sorted(distinct_shared)}"
    )


# --- fail-closed, rather than silently applying decoder naming --------------------------------


def test_unknown_architecture_fails_closed_instead_of_assuming_decoder_naming():
    from mobiletransformers.exceptions import UnsupportedModelError

    model = _tiny_encoder()
    # An architecture the registry does not know: previously this silently used `self_attn`/`q_proj`
    # and produced a wrap that looked fine and shared nothing.
    model.__class__ = type("TotallyMadeUpForSequenceClassification", (type(model),), {})
    with pytest.raises(UnsupportedModelError, match="unsupported architecture"):
        _wrap(model, ("query", "value"))
