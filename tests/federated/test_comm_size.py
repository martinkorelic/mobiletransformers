"""#35: per-round communication size is measured, bounded, and rank-r rather than merged-weight sized.

The size question is the whole reason the vocabulary decision existed, so it gets a test that states
the trade-off rather than a magic byte count.
"""

from __future__ import annotations

from mobiletransformers.federated.adapter_record import FederatedAdapterRecord
from tests.federated._helpers import make_arrays, make_handoff


def test_comm_size_bounded_and_accounts_for_payload():
    handoff = make_handoff()
    arrays = make_arrays()
    rec = FederatedAdapterRecord.from_handoff(handoff, arrays, base_model_id="org/base", peft_method="lora")

    # Payload = the four float32 ADAPTER FACTORS: (2,3) + (4,2) + (2,5) + (6,2) = 36 floats.
    raw_payload = sum(a.nbytes for a in arrays)
    assert raw_payload == 36 * 4

    size = rec.comm_size_bytes()
    # size = 4 (header length) + JSON header + raw payload; must exceed the payload and stay small.
    assert size > raw_payload + 4
    assert size < raw_payload + 4096  # header is tiny for a 4-tensor LoRA record


def test_rank_r_exchange_is_far_smaller_than_merged_at_real_dimensions():
    """Why #35 chose rank-r factors over merged weights, pinned as arithmetic.

    The toy fixture above cannot show this — at `r=2` with `d` of 3..6 the factors are no smaller than
    the weight, which is exactly right and exactly why the fixture is a poor place to assert it. At
    real dimensions the ratio is `d_in * d_out / (r * (d_in + d_out))`.

    Numbers are SmolLM2-135M's adapted layers, the model this project actually ships: `q_proj` is
    576x576 and `v_proj` is 576x192, r=8, 30 layers, two adapted modules per layer.
    """
    rank = 8
    layers = 30
    adapted = [(576, 576), (576, 192)]  # q_proj, v_proj

    merged_floats = layers * sum(d_in * d_out for d_in, d_out in adapted)
    factor_floats = layers * sum(rank * (d_in + d_out) for d_in, d_out in adapted)

    # 60 merged tensors vs 120 rank-r factors — more tensors, far fewer numbers.
    assert merged_floats == 30 * (576 * 576 + 576 * 192)

    # Cross-check against a number the pipeline records independently: the shipped SmolLM2 package's
    # `trainable_parameter_count`. If these two ever disagree, one of them is describing a different
    # adapter set than the other — which is precisely the confusion #35 was stuck in.
    assert factor_floats == 460_800

    ratio = merged_floats / factor_floats
    assert ratio > 25, f"expected a large saving at r={rank}, got {ratio:.1f}x"  # measured 28.8x

    # And the saving grows as rank shrinks relative to the dimensions.
    smaller_rank_floats = layers * sum(4 * (d_in + d_out) for d_in, d_out in adapted)
    assert merged_floats / smaller_rank_floats > ratio
