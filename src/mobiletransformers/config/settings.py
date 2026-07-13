"""Typed, env-driven runtime settings — the single owner of secrets and machine paths.

Decision (final): stdlib ``dataclass`` loader; do NOT add ``pydantic-settings``. ``pydantic>=2``
is a core dependency for the typed *tunable* config models (00_code_plans/09), but secrets stay
dependency-light and env-only.

``get_settings()`` is the ONLY place the environment is read for secrets. Business logic calls
``get_settings().hf_token`` / ``.require_hf_token()`` and never reads secret env vars directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of secrets + machine-specific paths, read from the environment."""

    hf_token: str | None
    hf_cache: Path | None
    # Azure OpenAI (evaluation only)
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_deployment_name: str | None
    azure_model_name: str | None
    azure_api_version: str | None
    # Gemini (openehr evaluation)
    gemini_api_key: str | None

    def require_hf_token(self) -> str:
        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set (mobiletransformers.config.settings). "
                "Export it or add it to your .env file."
            )
        return self.hf_token


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton (env is read once, then cached)."""
    load_dotenv()  # preserves the legacy config.py load_dotenv() behavior

    def _path(value: str | None) -> Path | None:
        return Path(value) if value else None

    return Settings(
        hf_token=os.environ.get("HF_TOKEN"),
        hf_cache=_path(os.environ.get("HF_CACHE")),
        azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME"),
        azure_model_name=os.environ.get("AZURE_MODEL_NAME"),
        azure_api_version=os.environ.get("AZURE_API_VERSION"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    )
