"""Tests for the LLM model factory and the credential check every run makes.

The suite stubs ``verify_llm_credentials`` globally so nothing reaches a real provider;
these are the tests *of* that function, so they ask for the real one.
"""
import json
import typing

import httpx2
import pytest

from openoutfind.core import llm

pytestmark = pytest.mark.no_llm_mock


def _reply(text):
    """An Anthropic Messages API response that answers *text*."""
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "some-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })


def test_build_llm_model_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm.build_llm_model("nope:some-model", "key")


@pytest.mark.parametrize("provider", list(llm._PROVIDER_BUILDERS))
def test_every_provider_builds_offline(provider):
    """Each builder imports and constructs its pydantic-ai model without a request.

    pydantic-ai renamed `OpenAIModel` to `OpenAIChatModel`, and both OpenAI builders
    kept importing the old name: every `openai:*` and `openai_compatible:*` install
    died on an `ImportError` while the Anthropic ones carried on (issue #959).
    """
    model = llm.build_llm_model(f"{provider}:some-model", "sk-key", "https://llm.invalid/v1")
    assert model.model_name == "some-model"


def test_openai_compatible_requires_an_api_base():
    with pytest.raises(ValueError, match="openai_compatible"):
        llm.build_llm_model("openai_compatible:some-model", "sk-key")


def test_anthropic_request_survives_union_message_param(monkeypatch):
    """A full Anthropic request goes out and its answer comes back on `anthropic` 2.x.

    2.0 made `BetaMessageParam` a `Union` that pydantic-ai calls as a constructor, so
    every Anthropic request died with `TypeError: Cannot instantiate typing.Union` —
    OpenOutSend's sequences stopped on it for a day.
    """
    from anthropic import AsyncAnthropic
    from pydantic_ai import Agent
    # Imported before the client is patched: the provider module builds a type union
    # over `AsyncAnthropic` at import time.
    import pydantic_ai.models.anthropic  # noqa: F401
    import pydantic_ai.providers.anthropic  # noqa: F401

    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return _reply("pong")

    client = AsyncAnthropic(
        api_key="sk-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda **kwargs: client)

    model = llm.build_llm_model("anthropic:some-model", "sk-key")
    result = llm.run_agent_sync(Agent(model).run("ping"))

    assert result.output == "pong"
    assert requests[0]["messages"] == [
        {"role": "user", "content": [{"text": "ping", "type": "text"}]},
    ]


def test_message_param_shim_leaves_a_constructible_type_alone(monkeypatch):
    import pydantic_ai.models.anthropic as pydantic_ai_anthropic

    class MessageParam(typing.TypedDict):
        role: str

    monkeypatch.setattr(pydantic_ai_anthropic, "BetaMessageParam", MessageParam)
    llm._accept_union_message_param()

    assert pydantic_ai_anthropic.BetaMessageParam is MessageParam


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
