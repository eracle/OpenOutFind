# tests/test_bootstrap.py
"""`validate_operator` — the gate every verb passes through before it works.

`find` and `init` both patch this out to test what comes after it, so nothing else
in the suite ever imports its body. That is how it shipped calling a `campaigns()`
that the one-ICP cut had already deleted: green tests, and an `ImportError` on the
first real `find`. These call it for real.
"""
import pytest

from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.management.bootstrap import validate_operator


@pytest.fixture
def configured(site_config):
    site_config.llm_api_key = "sk-test"
    site_config.save()
    return site_config


@pytest.mark.django_db
class TestItPassesAConfiguredInstall:
    def test_a_key_and_an_operator_are_the_whole_gate(self, configured, operator):
        assert validate_operator() is None


@pytest.mark.django_db
class TestItNamesWhatIsMissing:
    def test_no_llm_key_is_incomplete_onboarding(self, site_config, operator):
        with pytest.raises(OpenOutFindError) as raised:
            validate_operator()

        assert raised.value.error_type == ErrorType.ONBOARDING_INCOMPLETE
        assert "LLM API key" in str(raised.value)

    def test_no_operator_is_incomplete_onboarding(self, configured):
        from django.contrib.auth.models import User

        User.objects.all().delete()

        with pytest.raises(OpenOutFindError) as raised:
            validate_operator()

        assert raised.value.error_type == ErrorType.ONBOARDING_INCOMPLETE
        assert "operator" in str(raised.value)
