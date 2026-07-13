"""Config-layering tests: precedence (CLI > env > YAML > default), settings caching, and shims."""

from __future__ import annotations

import importlib
import warnings

import pytest

from mobiletransformers.config import resolve
from mobiletransformers.config.settings import Settings, get_settings


# --- precedence: CLI > env > YAML > default ---------------------------------------
def test_resolve_cli_wins():
    assert resolve("cli", "env", "yaml", "default") == "cli"


def test_resolve_env_when_no_cli():
    assert resolve(None, "env", "yaml", "default") == "env"


def test_resolve_yaml_when_no_cli_env():
    assert resolve(None, None, "yaml", "default") == "yaml"


def test_resolve_default_when_nothing_else():
    assert resolve(None, None, None, "default") == "default"


def test_resolve_all_none():
    assert resolve(None, None, None, None) is None


# --- settings: env-driven, cached -------------------------------------------------
def test_get_settings_reads_env(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "tok-123")
    monkeypatch.setenv("HF_CACHE", "/tmp/hf")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-xyz")
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.hf_token == "tok-123"
    assert str(settings.hf_cache) == "/tmp/hf"
    assert settings.gemini_api_key == "gem-xyz"
    assert settings.require_hf_token() == "tok-123"
    get_settings.cache_clear()


def test_get_settings_is_cached(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "first")
    first = get_settings()
    monkeypatch.setenv("HF_TOKEN", "second")  # ignored: lru_cache returns the same object
    second = get_settings()
    assert first is second
    assert second.hf_token == "first"
    get_settings.cache_clear()


def test_require_hf_token_raises_when_missing(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN is not set"):
        get_settings().require_hf_token()
    get_settings.cache_clear()


# --- legacy import compatibility (deprecation shims) ------------------------------
def test_legacy_config_shim_imports_and_warns():
    import config as legacy_config

    importlib.reload(legacy_config)  # ensure the DeprecationWarning path runs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(legacy_config)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    # experiment constants still resolve through the shim
    assert legacy_config.TASK_EPOCHS["boolq"] == 2
    assert legacy_config.BATCH_SIZE == 32
    assert hasattr(legacy_config, "AZURE_API_VERSION")


def test_legacy_parser_config_shim_imports_and_warns():
    from tools import parser_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(parser_config)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert parser_config.TRAIN_CONFIG == "TRAIN_BUILDER"
    assert parser_config.TASK_NAME_TO_DATASET["boolq"] == "google/boolq"
