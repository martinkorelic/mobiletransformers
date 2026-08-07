"""HuggingFace ``Trainer`` callbacks.

Migrated from ``tools/utils.py`` (Migration Map S1).

``transformers`` is imported at module level because ``MemoryLoggerCallback`` SUBCLASSES
``TrainerCallback`` — a base class must exist when the class body is evaluated, so there is no honest
way to defer it. This module is therefore importable only under the ``train`` profile, and nothing in
the package imports it eagerly (``tests/unit/test_import_weight.py`` guards the top-level import).
torch/psutil stay function-local since those are only used inside the methods.
"""

from __future__ import annotations

from transformers import TrainerCallback


class MemoryLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.pre_backward_memory = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        import psutil  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            # Log GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
            logs["gpu_memory_allocated_MB"] = allocated
            logs["gpu_memory_reserved_MB"] = reserved

            # Log memory usage before backward pass
            if self.pre_backward_memory:
                logs["gpu_memory_allocated_MB_pre_bp"] = self.pre_backward_memory[
                    "gpu_memory_allocated_MB_pre_bp"
                ]
        else:
            # Log CPU memory usage using psutil
            mem = psutil.virtual_memory()
            logs["cpu_memory_used_MB"] = mem.used / 1024**2  # Convert to MB
            # Log memory usage before backward pass
            if self.pre_backward_memory:
                logs["cpu_memory_used_MB_pre_bp"] = self.pre_backward_memory["cpu_memory_used_MB_pre_bp"]

    def on_optimizer_step(self, args, state, control, **kwargs):
        import psutil  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            # Log GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            # reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
            self.pre_backward_memory["gpu_memory_allocated_MB_pre_bp"] = allocated
            # self.pre_backward_memory["gpu_memory_reserved_MB_pre_bs"] = reserved
        else:
            # Log CPU memory usage using psutil
            mem = psutil.virtual_memory()
            self.pre_backward_memory["cpu_memory_used_MB_pre_bp"] = mem.used / 1024**2  # Convert to MB
