"""Data-driven registries: the single source of truth for every closed dispatch choice.

A closed set of choices is **data** (an enum member + a registry row), never an ``if/elif`` chain in
business logic. Adding a PEFT method, architecture, or merger variant is a registry entry, not a new
branch. Extended with more registries (task, execution-provider,
document-loader, export-frontend) as their consumers land.
"""

from __future__ import annotations

from mobiletransformers.config.registry.architecture import (
    ARCHITECTURE_REGISTRY,
    ArchitectureSpec,
    resolve_architecture,
)
from mobiletransformers.config.registry.merger import (
    MergerSpec,
    build_merger_model,
    resolve_merger,
)
from mobiletransformers.config.registry.peft import (
    PEFT_REGISTRY,
    AdapterComponent,
    PEFTMethodSpec,
    get_peft_spec,
)
from mobiletransformers.config.registry.task import (
    TASK_REGISTRY,
    TaskSpec,
    get_task_spec,
)

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "ArchitectureSpec",
    "resolve_architecture",
    "PEFT_REGISTRY",
    "PEFTMethodSpec",
    "AdapterComponent",
    "get_peft_spec",
    "MergerSpec",
    "resolve_merger",
    "build_merger_model",
    "TASK_REGISTRY",
    "TaskSpec",
    "get_task_spec",
]
