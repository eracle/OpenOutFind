# tests/test_readiness.py
"""`check_ready` — everything a run has to have been given, checked before it spends.

This is the gate every verb passes through, and `find` and `check` both patch it out to
test what comes after, so nothing else in the suite exercises its body. These call it for
real: an install that is missing something must learn all of it from one error, and an
install that has everything must not be asked a question, because there is nobody there.
"""
from unittest.mock import patch

import pytest

from openoutfind.core import readiness
from openoutfind.core.errors import ErrorType, OpenOutFindError

READY = {
    "product_docs": "a self-hosted CI dashboard",
    "campaign_target": "engineering leaders at growth-stage SaaS",
    "ai_model": "anthropic:claude-sonnet-4-5",
    "llm_api_key": "sk-test",
    "bettercontact_api_key": "bc-test",
    "operator_country_code": "us",
}


@pytest.fixture
def ready(configure, monkeypatch):
    """Everything a run needs, as the environment it would arrive in."""
    monkeypatch.setenv(readiness.OPERATOR_EMAIL, "me@example.com")
    monkeypatch.setenv(readiness.ACCEPT_LEGAL_NOTICE, "true")
    return configure(**READY)


@pytest.mark.django_db
class TestAGivenRunIsReady:
    def test_nothing_is_missing_and_nothing_is_asked(self, ready):
        assert readiness.missing_variables() == {}
        assert readiness.check_ready() is None

    def test_the_operator_becomes_a_row_written_once(self, ready, monkeypatch):
        from django.contrib.auth.models import User

        readiness.check_ready()
        user = User.objects.get()
        assert user.email == "me@example.com"

        # Identity is a record, not an answer: renaming the variable must not rename the
        # person a campaign belongs to.
        monkeypatch.setenv(readiness.OPERATOR_EMAIL, "someone.else@example.com")
        readiness.check_ready()

        assert User.objects.get().email == "me@example.com"

    def test_the_email_is_only_asked_for_while_there_is_no_operator(self, ready, monkeypatch):
        readiness.check_ready()
        monkeypatch.delenv(readiness.OPERATOR_EMAIL)

        assert readiness.missing_variables() == {}

    def test_the_model_is_pinged_every_run(self, ready):
        """Nothing is stored, so nothing is trusted from last time: a key rotated out
        from under a timer fails here rather than mid-pass with a lead in hand."""
        with patch("openoutfind.core.llm.verify_llm_credentials",
                   return_value=None) as verify:
            readiness.check_ready()
            readiness.check_ready()

        assert verify.call_count == 2

    def test_a_refused_key_is_a_typed_error_naming_the_model(self, ready):
        with patch("openoutfind.core.llm.verify_llm_credentials",
                   return_value="401 invalid x-api-key"):
            with pytest.raises(OpenOutFindError) as raised:
                readiness.check_ready()

        assert raised.value.error_type == ErrorType.BAD_CONFIG
        assert "anthropic:claude-sonnet-4-5" in str(raised.value)


@pytest.mark.django_db
class TestItNamesEverythingMissingAtOnce:
    def test_a_bare_environment_names_all_four_groups(self, site_config):
        missing = readiness.missing_variables()

        assert set(missing) == {"campaign", "llm", "bettercontact", "account"}
        with pytest.raises(OpenOutFindError) as raised:
            readiness.check_ready()

        assert raised.value.error_type == ErrorType.ONBOARDING_INCOMPLETE
        for variable in ("OPENOUTFIND_PRODUCT_DOCS", "OPENOUTFIND_LLM_API_KEY",
                         "OPENOUTFIND_BETTERCONTACT_API_KEY", "OPENOUTFIND_OPERATOR_COUNTRY"):
            assert variable in str(raised.value)

    def test_one_missing_value_leaves_the_rest_satisfied(self, ready, monkeypatch):
        monkeypatch.delenv("OPENOUTFIND_BETTERCONTACT_API_KEY")

        assert readiness.missing_variables() == {
            "bettercontact": ["OPENOUTFIND_BETTERCONTACT_API_KEY"]}


@pytest.mark.django_db
class TestTheLegalNotice:
    """It is a gate, and it is asked on every run — an install cannot inherit somebody
    else's agreement by inheriting their database."""

    def test_silence_is_not_acceptance(self, ready, monkeypatch):
        monkeypatch.delenv(readiness.ACCEPT_LEGAL_NOTICE)

        assert readiness.missing_variables() == {"account": [readiness.ACCEPT_LEGAL_NOTICE]}

    def test_a_no_is_not_acceptance(self, ready, monkeypatch):
        monkeypatch.setenv(readiness.ACCEPT_LEGAL_NOTICE, "false")

        assert readiness.missing_variables() == {"account": [readiness.ACCEPT_LEGAL_NOTICE]}

    def test_an_operator_row_does_not_carry_it_forward(self, ready, monkeypatch):
        readiness.check_ready()
        monkeypatch.delenv(readiness.ACCEPT_LEGAL_NOTICE)

        assert readiness.ACCEPT_LEGAL_NOTICE in readiness.missing_variables()["account"]

    def test_a_value_that_is_neither_stops_the_run(self, ready, monkeypatch):
        """A bad value is a different thing from an absent one: naming the variable as
        missing would print one the operator has already set."""
        monkeypatch.setenv(readiness.ACCEPT_LEGAL_NOTICE, "sure")

        with pytest.raises(OpenOutFindError) as raised:
            readiness.missing_variables()

        assert raised.value.error_type == ErrorType.BAD_CONFIG


@pytest.mark.django_db
class TestTheNewsletter:
    """Consent is an explicit yes, and it is acted on once — when the operator row is
    created, not on every run that carries the variable."""

    def test_it_subscribes_on_an_explicit_yes(self, ready, monkeypatch):
        monkeypatch.setenv(readiness.NEWSLETTER, "yes")

        with patch("openoutfind.core.newsletter.subscribe_to_newsletter") as subscribe:
            readiness.check_ready()
            readiness.check_ready()

        subscribe.assert_called_once_with("me@example.com")

    def test_silence_subscribes_nobody(self, ready):
        with patch("openoutfind.core.newsletter.subscribe_to_newsletter") as subscribe:
            readiness.check_ready()

        subscribe.assert_not_called()
