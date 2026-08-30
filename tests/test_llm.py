"""Tests for the LLM model factory and onboarding credential verification."""
import pytest

from openoutfind.core import llm


def test_build_llm_model_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm.build_llm_model("nope:some-model", "key")


def test_verify_llm_credentials_ok(monkeypatch):
    monkeypatch.setattr(llm, "_ping_model", lambda *a, **k: None)
    assert llm.verify_llm_credentials("anthropic:claude", "sk-key") is None


def test_verify_llm_credentials_reports_a_provider_refusal(monkeypatch):
    from pydantic_ai.exceptions import ModelHTTPError

    def boom(*a, **k):
        raise ModelHTTPError(status_code=401, model_name="claude", body="invalid api key")

    monkeypatch.setattr(llm, "_ping_model", boom)
    assert "invalid api key" in llm.verify_llm_credentials("anthropic:claude", "bad")


def test_verify_llm_credentials_reports_an_unusable_model_id(monkeypatch):
    # `build_llm_model` rejects the id before any request is made; that is an answer
    # about a value the operator set, not a bug.
    def boom(*a, **k):
        raise ValueError("AI_MODEL 'mystery' has no provider prefix.")

    monkeypatch.setattr(llm, "_ping_model", boom)
    assert "no provider prefix" in llm.verify_llm_credentials("mystery", "sk-key")


def test_verify_llm_credentials_does_not_swallow_a_bug(monkeypatch):
    """A library incompatibility must not read as a rejected key.

    This is the `anthropic` 1.0.0 failure: it dropped `temperature`, pydantic-ai
    passed it anyway, and the `TypeError` came back as
    `bad_config: OPENOUTFIND_LLM_API_KEY` — sending the operator after a key that
    was fine. Only what a configuration can cause is an answer here.
    """
    def boom(*a, **k):
        raise TypeError("create() got an unexpected keyword argument 'temperature'")

    monkeypatch.setattr(llm, "_ping_model", boom)
    with pytest.raises(TypeError, match="temperature"):
        llm.verify_llm_credentials("anthropic:claude", "sk-key")


def test_every_provider_sdk_is_silenced_at_debug():
    """A new provider must arrive with its SDK on the silenced list.

    Otherwise `--log-level debug` becomes unreadable the first time someone switches
    models: every LLM SDK dumps the whole request body at DEBUG (`Request options:
    {'method': 'post', ...}`, one screen per call), which buries the discovery walk's
    reasoning that the flag exists to show. This failed for `anthropic` in the wild —
    only `openai` was listed.
    """
    from openoutfind.core.llm import _PROVIDER_BUILDERS
    from openoutfind.core.logging import SILENCED_LOGGERS

    # The SDK package each provider actually imports, where it differs from the key.
    sdk_for = {"mistral": "mistralai", "openai_compatible": "openai"}
    missing = [
        provider for provider in _PROVIDER_BUILDERS
        if sdk_for.get(provider, provider) not in SILENCED_LOGGERS
    ]
    assert not missing, f"providers whose SDK logger is not silenced: {missing}"
