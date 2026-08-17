"""Loading serialized adapter weights back onto a PEFT-wrapped model.

Migration Map S8 side-effect: this lived in ``research/utils.py``, but the only caller is the packaged
`evaluation.eval_adapter_models` evaluator. A packaged module importing `research.` works from a
checkout and fails from an installed wheel — `research/` is not part of the distribution — so the
helper moved rather than the import being allow-listed. `research/utils.py` re-exports it.
"""

from __future__ import annotations

import os
from typing import Any


def load_mars_adapters(model: Any, adapter_path: str) -> Any:
    """Load a safetensors adapter file onto ``model`` in place and return it.

    ``strict=False``: the file holds only the adapted tensors, so the base model's own parameters are
    legitimately "missing" from it. That also means a wholly mismatched file loads silently — the
    caller is responsible for pairing an adapter with the model it was trained against.
    """
    from safetensors.torch import load_file

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    adapter_state_dict = load_file(adapter_path)
    model.base_model.model.load_state_dict(adapter_state_dict, strict=False)
    return model
