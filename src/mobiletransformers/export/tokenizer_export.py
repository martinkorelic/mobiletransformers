"""On-device tokenizer config emit (``mobiletransformers_tokenizer_config.json``).

Migrated from ``tools/tokenizer_export.py`` (Migration Map S1). ``transformers`` is imported inside the
functions so the module stays importable in the core environment — the export pipeline imports it
eagerly and must not drag transformers in on a ``--dry-run``.

### The file this writes is load-bearing, and it used to be mostly wrong

``ORTTokenizerNative`` reads ``model.vocab_size`` out of this file and hands it to
``performInferenceStep``, which scans exactly that many floats of the logits row to pick the next
token. So the number here is not documentation — it defines the set of token ids the sampler may
return.

The previous version built the whole block from whatever ``GenerationConfig.from_pretrained``
returned, falling back to ``AutoConfig`` only when that *raised*. A ``GenerationConfig`` carries none
of the architecture fields, so for every model with a ``generation_config.json`` each
``getattr(config, ..., default)`` silently took the default: 12 heads, 12 layers, context 2048, type
``"unknown"`` — and ``vocab_size`` fell through to ``len(tokenizer.get_vocab())``.

That last one is the one that bites. ``len(tokenizer)`` counts **added tokens**, which need not be
backed by embedding rows. FunctionGemma's tokenizer declares ``<image_soft_token>`` (262144) and
``<end_of_image>`` (262145) above a 262144-row embedding table, so the emitted vocab_size was 262146
and the sampler read two floats past the end of every logits row. When that garbage won the argmax
the id was fed straight back as the next input and ORT failed the embedding lookup with

    Gather node ... indices element out of data bounds, idx=262145 ... range [-262144,262143]

SmolLM2 survived it only because its added tokens happen to fit inside its declared vocabulary — the
same code was wrong there too and nothing could see it.

So: the architecture facts come from the **model** config and the token ids from the **generation**
config, each from the object that actually declares them, and a vocab size that cannot be sourced
from the model config is an error rather than a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# What the device does when a field is absent. Only ever reached for a model config that genuinely
# omits the field; they are NOT a fallback for reading the wrong object.
_DEFAULT_CONTEXT_LENGTH = 2048
_DEFAULT_ATTENTION_HEADS = 12
_DEFAULT_HIDDEN_LAYERS = 12


def _first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """The first of ``names`` that is present **and not None** on ``obj``.

    ``getattr(cfg, "bos_token_id", 2)`` returns ``None`` — not 2 — for a config that declares the
    attribute as null, which is how ``bos_token_id: null`` reached a package exported from a model
    whose real BOS is 2.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _text_config(config: Any) -> Any:
    """The sub-config carrying the decoder's shape.

    Multimodal configs (Gemma 3, Llava, Qwen-VL) put ``num_hidden_layers``/``vocab_size`` on a nested
    ``text_config`` and leave the top level describing the composite. Reading the top level yields a
    config with none of the fields, i.e. all the defaults again.
    """
    nested = getattr(config, "text_config", None)
    return nested if nested is not None and getattr(nested, "vocab_size", None) is not None else config


def build_device_tokenizer_config(
    model_config: Any,
    generation_config: Any = None,
    tokenizer: Any = None,
) -> dict:
    """Assemble the ``mobiletransformers_tokenizer_config.json`` payload.

    Pure, and deliberately takes duck-typed objects rather than transformers classes, so the field
    routing that this module got wrong for its whole life is testable without loading a model.

    :param model_config: an ``AutoConfig`` — the authority for architecture and ``vocab_size``.
    :param generation_config: a ``GenerationConfig``, when the model ships one. Authority for the
        token ids *only*: its ``eos_token_id`` is frequently a longer list than the model config's
        (FunctionGemma adds ``106``/``<end_of_turn>``), and stopping on more of them is correct.
    :param tokenizer: consulted only for token ids the configs leave unset. **Never** for
        ``vocab_size`` — see the module docstring.
    :raises ValueError: when no ``vocab_size`` can be read off the model config. Emitting a guess is
        what produced an out-of-bounds sampler, and a package that fails to export is strictly better
        than one that fails on the phone.
    """
    text = _text_config(model_config)

    vocab_size = _first_attr(text, "vocab_size")
    if vocab_size is None:
        raise ValueError(
            "the model config declares no vocab_size, so the number of embedding rows the sampler "
            "may address is unknown. It must NOT be taken from len(tokenizer): added tokens are not "
            "guaranteed to have embedding rows, and a sampler allowed to return an id above the "
            "table fails inside ORT's Gather with an out-of-bounds index."
        )

    # Token ids: the generation config wins where it has an opinion, because that is the object that
    # describes how the model is meant to be decoded.
    sources = [c for c in (generation_config, text, model_config) if c is not None]

    def token_id(*names: str) -> Any:
        for source in sources:
            value = _first_attr(source, *names)
            if value is not None:
                return value
        return None

    eos_token_id = token_id("eos_token_id")
    if eos_token_id is None and tokenizer is not None:
        eos_token_id = _first_attr(tokenizer, "eos_token_id")

    bos_token_id = token_id("bos_token_id")
    if bos_token_id is None and tokenizer is not None:
        bos_token_id = _first_attr(tokenizer, "bos_token_id")

    pad_token_id = token_id("pad_token_id")
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = _first_attr(tokenizer, "pad_token_id")
    if pad_token_id is None:
        # Padding with EOS is the transformers convention for models that declare no pad token.
        pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) and eos_token_id else eos_token_id

    num_attention_heads = _first_attr(text, "num_attention_heads", default=_DEFAULT_ATTENTION_HEADS)

    payload = {
        "model": {
            "bos_token_id": bos_token_id,
            "context_length": _first_attr(
                text,
                "max_position_embeddings",
                "max_sequence_length",
                "n_positions",
                default=_DEFAULT_CONTEXT_LENGTH,
            ),
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": _first_attr(
                text, "num_hidden_layers", "n_layer", default=_DEFAULT_HIDDEN_LAYERS
            ),
            "num_key_value_heads": _first_attr(text, "num_key_value_heads", default=num_attention_heads),
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "type": _first_attr(text, "model_type", default="unknown"),
            "vocab_size": vocab_size,
        }
    }

    # Not fatal, but worth naming: it is the exact discrepancy that used to be written out as truth.
    declared = None
    if tokenizer is not None:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        added = getattr(tokenizer, "get_added_vocab", None)
        if callable(get_vocab):
            try:
                declared = len(get_vocab())
                if callable(added):
                    declared = max(declared, max((v for v in added().values()), default=-1) + 1)
            except Exception:  # noqa: BLE001 - a tokenizer that cannot enumerate is not an export failure
                declared = None
    if declared is not None and declared != vocab_size:
        print(
            f"note: the tokenizer addresses {declared} ids but the model config declares "
            f"{vocab_size} embedding rows. Using {vocab_size} — ids above it have no row and must "
            "never be sampled."
        )

    return payload


def export_tokenizer_config(model_name_or_path, output_dir="build", hf_token=None, trust_remote_code=True):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig  # noqa: PLC0415

    """
    Export tokenizer and config files from HuggingFace model.

    Args:
        model_name_or_path (str): HuggingFace model name or local path
        output_dir (str): Output directory (default: "build")
        hf_token (str): HuggingFace token for private models
        trust_remote_code (bool): Whether to trust remote code

    Returns:
        dict: The generated config dictionary
    """

    # Create output directories
    tokenizer_dir = Path(output_dir) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load tokenizer and config
        print(f"Loading tokenizer from {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, token=hf_token, trust_remote_code=trust_remote_code
        )

        # The MODEL config, always: it is the only object that knows how many embedding rows exist.
        model_config = AutoConfig.from_pretrained(
            model_name_or_path, token=hf_token, trust_remote_code=trust_remote_code
        )

        # The generation config is additive and optional — a model without one is not an error.
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_name_or_path, token=hf_token, trust_remote_code=trust_remote_code
            )
        except Exception as exc:  # noqa: BLE001 - absence is the common case, not a failure
            print(
                f"No generation config for {model_name_or_path} ({exc}); using the model config's token ids."
            )
            generation_config = None

        # Save tokenizer files to build/tokenizer directory
        print(f"Saving tokenizer files to {tokenizer_dir}...")
        tokenizer.save_pretrained(tokenizer_dir)

        mobiletransformers_config = build_device_tokenizer_config(
            model_config=model_config,
            generation_config=generation_config,
            tokenizer=tokenizer,
        )

        # Save the main config file
        config_path = Path(output_dir) / "tokenizer" / "mobiletransformers_tokenizer_config.json"
        print(f"Saving main config to {config_path}...")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(mobiletransformers_config, f, indent=4, ensure_ascii=False)

        print("Export completed successfully!")
        print("Files saved:")
        print(f"  - Main config: {config_path}")
        print(f"  - Tokenizer files: {tokenizer_dir}")

        # List tokenizer files that were saved
        tokenizer_files = list(tokenizer_dir.glob("*.json"))
        for file in tokenizer_files:
            print(f"    - {file.name}")

        return mobiletransformers_config

    except Exception as e:
        print(f"Error exporting tokenizer: {str(e)}")
        raise


def export_tokenizer_config_advanced(
    model_name_or_path, output_dir="build", hf_token=None, trust_remote_code=True, extra_config_overrides=None
):
    """
    Advanced version with additional configuration options.

    Args:
        model_name_or_path (str): HuggingFace model name or local path
        output_dir (str): Output directory
        hf_token (str): HuggingFace token
        trust_remote_code (bool): Whether to trust remote code
        extra_config_overrides (dict): Additional config values to override

    Returns:
        dict: The generated config dictionary
    """

    config = export_tokenizer_config(model_name_or_path, output_dir, hf_token, trust_remote_code)

    # Apply any overrides
    if extra_config_overrides:
        for key, value in extra_config_overrides.items():
            if key in config["model"]:
                config["model"][key] = value
                print(f"Override applied: {key} = {value}")

        # Save updated config
        config_path = Path(output_dir) / "mobiletransformers_tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    return config
