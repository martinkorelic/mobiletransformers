"""The on-device tokenizer config: what goes in it, and which object each field comes from.

``mobiletransformers_tokenizer_config.json`` is the only file the Android Native tokenizer reads for
``vocab_size``, and that number bounds the sampler's argmax scan over the logits row. A value larger
than the embedding table lets the sampler return an id with no embedding row, which ORT then fails on
in the next step's ``Gather``. So these are correctness tests, not formatting ones.
"""

from __future__ import annotations

import pytest

from mobiletransformers.export.tokenizer_export import build_device_tokenizer_config


class FakeConfig:
    """A stand-in for a transformers config: attributes only, nothing else is used."""

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


class FakeTokenizer:
    def __init__(self, vocab_size: int, added: dict[str, int] | None = None, **ids):
        self._vocab = {f"tok{i}": i for i in range(vocab_size)}
        self._added = added or {}
        for key, value in ids.items():
            setattr(self, key, value)

    def get_vocab(self):
        return self._vocab

    def get_added_vocab(self):
        return self._added


def gemma3_text_config() -> FakeConfig:
    """``google/functiongemma-270m-it`` as its config.json actually reads."""
    return FakeConfig(
        model_type="gemma3_text",
        vocab_size=262144,
        num_hidden_layers=18,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=32768,
        bos_token_id=2,
        eos_token_id=[1, 50],
        pad_token_id=0,
    )


def gemma3_generation_config() -> FakeConfig:
    """Its generation_config.json: token ids and nothing else — no architecture fields at all."""
    return FakeConfig(bos_token_id=2, eos_token_id=[1, 50, 106], pad_token_id=0)


def test_vocab_size_comes_from_the_model_not_the_tokenizer():
    """The regression that produced ``idx=262145 ... range [-262144,262143]`` on device.

    FunctionGemma's tokenizer declares ``<image_soft_token>`` (262144) and ``<end_of_image>``
    (262145) above a 262144-row embedding table. Sizing the sampler from the tokenizer let it read
    two floats past the end of every logits row and return an id with no embedding.
    """
    added = {"<image_soft_token>": 262144, "<end_of_image>": 262145}
    tokenizer = FakeTokenizer(vocab_size=262144, added=added)

    payload = build_device_tokenizer_config(gemma3_text_config(), gemma3_generation_config(), tokenizer)

    assert payload["model"]["vocab_size"] == 262144, (
        "the emitted vocab size must be the number of embedding rows; anything the tokenizer can "
        "address above that has no row and must never be sampled"
    )


def test_architecture_fields_come_from_the_model_config_not_the_defaults():
    """A ``GenerationConfig`` has none of these, so reading it wrote 12/12/12/2048/'unknown'."""
    payload = build_device_tokenizer_config(gemma3_text_config(), gemma3_generation_config(), None)

    model = payload["model"]
    assert model["num_hidden_layers"] == 18
    assert model["num_attention_heads"] == 4
    assert model["num_key_value_heads"] == 1
    assert model["context_length"] == 32768
    assert model["type"] == "gemma3_text"


def test_a_null_bos_in_the_generation_config_falls_through_to_the_model():
    """``getattr(cfg, 'bos_token_id', 2)`` returns None for a declared-null field, not 2."""
    generation = FakeConfig(bos_token_id=None, eos_token_id=[1, 50, 106], pad_token_id=None)

    payload = build_device_tokenizer_config(gemma3_text_config(), generation, None)

    assert payload["model"]["bos_token_id"] == 2
    assert payload["model"]["pad_token_id"] == 0


def test_the_generation_config_wins_for_eos():
    """It commonly stops on more tokens than the model config lists — ``106`` is ``<end_of_turn>``."""
    payload = build_device_tokenizer_config(gemma3_text_config(), gemma3_generation_config(), None)

    assert payload["model"]["eos_token_id"] == [1, 50, 106]


def test_a_nested_text_config_is_used_for_the_decoder_shape():
    """Multimodal configs describe the composite at the top level and the decoder underneath."""
    composite = FakeConfig(
        model_type="gemma3",
        text_config=FakeConfig(
            model_type="gemma3_text",
            vocab_size=262144,
            num_hidden_layers=18,
            num_attention_heads=4,
            max_position_embeddings=32768,
            eos_token_id=[1, 50],
        ),
    )

    payload = build_device_tokenizer_config(composite, None, None)

    assert payload["model"]["vocab_size"] == 262144
    assert payload["model"]["num_hidden_layers"] == 18


def test_a_model_config_without_a_vocab_size_is_an_export_failure():
    """Failing the export beats emitting a guess that only fails once it is on a phone."""
    with pytest.raises(ValueError, match="vocab_size"):
        build_device_tokenizer_config(FakeConfig(model_type="mystery"), None, FakeTokenizer(vocab_size=1000))


def test_pad_falls_back_to_the_first_eos_when_nothing_declares_one():
    payload = build_device_tokenizer_config(
        FakeConfig(model_type="llama", vocab_size=32000, eos_token_id=[2, 32001]),
        None,
        None,
    )

    assert payload["model"]["pad_token_id"] == 2


def test_smollm2_still_emits_what_it_did_before():
    """The control: the model this was silently wrong for, but which happened to work.

    Its tokenizer and its config agree on 49152, so the old tokenizer-derived path produced the right
    number by luck. The architecture fields it used to get wrong are now right, and the one field the
    device actually depends on is unchanged — this must not move.
    """
    config = FakeConfig(
        model_type="llama",
        vocab_size=49152,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        max_position_embeddings=8192,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )

    payload = build_device_tokenizer_config(config, None, FakeTokenizer(vocab_size=49152))

    assert payload["model"]["vocab_size"] == 49152
    assert payload["model"]["num_hidden_layers"] == 30
    assert payload["model"]["type"] == "llama"
