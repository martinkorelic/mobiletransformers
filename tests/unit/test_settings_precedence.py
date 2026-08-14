"""Config-layering tests: precedence (CLI > env > YAML > default), settings caching, and shims."""

from __future__ import annotations

from pathlib import Path

import pytest

from mobiletransformers.config import resolve
from mobiletransformers.config import settings as settings_module
from mobiletransformers.config.settings import Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Neutralize a real `.env` for every test in this module.

    `get_settings()` calls `load_dotenv()`, which writes `.env` into `os.environ` — so a
    `monkeypatch.delenv("HF_TOKEN")` is silently undone, and these tests measured the developer's
    machine rather than the precedence rules they are named after. Since `.env` is the DOCUMENTED
    place to put `HF_TOKEN` (`config/settings.py` says so in its own error message), the suite went
    red for anyone who followed the documentation.

    Patched at the settings module rather than deleting the file, so nothing touches the developer's
    real secrets.
    """
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_require_hf_token_raises_when_missing(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN is not set"):
        get_settings().require_hf_token()
    get_settings.cache_clear()


# --- legacy import compatibility (deprecation shims) ------------------------------
# Both shims are GONE. `tools/parser_config.py` went with the `tools/` root in S9;
# the root `config.py` was deleted 2026-08-14 once its last two importers were repointed
# (`evaluation/mobile/recommendation_eval.py` -> `get_settings()`, which also fixed a
# ModuleNotFoundError from an installed wheel, and `research/offline_train_eval.py` ->
# `mobiletransformers.config.constants`). The constants they re-exported are covered by the
# symbol golden; the secrets are covered by the precedence tests above.
def test_no_root_config_shim_remains():
    """The root `config.py` must stay deleted — it shadowed the package name and was not in the wheel."""
    assert not (REPO_ROOT / "config.py").exists()


def test_experiment_constants_resolve_from_the_package():
    from mobiletransformers.config.constants import BATCH_SIZE, TASK_EPOCHS

    assert TASK_EPOCHS["boolq"] == 2
    assert BATCH_SIZE == 32
