# tests/test_init.py
"""`init` — the setup phase, given a name and an exit code of its own.

Everything here already happened inside `find`; what is asserted is that it can now
happen *deliberately*, report what it created, and cost nothing. `find` keeps its own
boot path, so the headless one-command install is not what these tests are about —
that is `tests/test_find.py`.
"""
import io
import json
from unittest.mock import patch

import pytest
from django.core.management import call_command

from openoutfind.core.errors import ErrorType, OpenOutFindError


@pytest.fixture
def bootstrapped():
    """Migrating and onboarding have their own tests; `init` is asserted on what it
    reports about the campaign that came out the other side."""
    with patch("openoutfind.core.management.commands.init.ensure_database"), \
            patch("openoutfind.core.management.commands.init.ensure_onboarded"), \
            patch("openoutfind.core.management.commands.init.validate_operator"), \
            patch("openoutfind.core.logging.print_banner"):
        yield


def _run(*args) -> str:
    out = io.StringIO()
    call_command("init", *args, stdout=out)
    return out.getvalue()


@pytest.mark.django_db
class TestItReportsWhatWasCreated:
    def test_the_campaign_is_named_on_stdout(self, campaign, bootstrapped):
        assert campaign.name in _run()

    def test_json_is_one_object_a_program_can_read(self, campaign, bootstrapped):
        described = json.loads(_run("--json"))

        assert described["name"] == campaign.name
        assert described["product_docs_chars"] == len(campaign.product_docs)
        assert described["exportable"] == 0

    def test_it_spends_nothing(self, campaign, bootstrapped):
        """`init` is the verb you can always run: no discovery, no qualification, no
        lookup — so nothing it does can cost a credit or an LLM call."""
        with patch("openoutfind.core.cycle.run_one_action") as action:
            _run()

        action.assert_not_called()

    def test_running_it_twice_is_not_an_error(self, campaign, bootstrapped):
        """Re-running setup should tell you where you are, not refuse."""
        assert campaign.name in _run()
        assert campaign.name in _run()


@pytest.mark.django_db
class TestTheFileFlags:
    """The two long fields come from files because shell-quoting a page of markdown is
    a way to corrupt it quietly."""

    def test_a_file_supplies_the_product_description(self, tmp_path, monkeypatch, bootstrapped):
        monkeypatch.delenv("OPENOUTFIND_PRODUCT_DESCRIPTION", raising=False)
        doc = tmp_path / "product.md"
        doc.write_text("# A self-hosted CI dashboard\n\nFor small dev teams.\n")

        from openoutfind.core.management.commands.init import _seed_environment
        _seed_environment({"product_docs": str(doc), "target": None, "name": None})

        import os
        assert "self-hosted CI dashboard" in os.environ["OPENOUTFIND_PRODUCT_DESCRIPTION"]

    def test_an_explicit_export_beats_the_file(self, tmp_path, monkeypatch, bootstrapped):
        """A flag never silently overrides something the operator already set."""
        monkeypatch.setenv("OPENOUTFIND_PRODUCT_DESCRIPTION", "from the environment")
        doc = tmp_path / "product.md"
        doc.write_text("from the file")

        from openoutfind.core.management.commands.init import _seed_environment
        _seed_environment({"product_docs": str(doc), "target": None, "name": None})

        import os
        assert os.environ["OPENOUTFIND_PRODUCT_DESCRIPTION"] == "from the environment"

    def test_a_missing_file_is_a_typed_error_naming_the_flag(self, campaign, bootstrapped):
        with pytest.raises(OpenOutFindError) as exc:
            call_command("init", "--product-docs", "/nope/absent.md", stdout=io.StringIO())

        assert exc.value.error_type == ErrorType.BAD_CONFIG
        assert "--product-docs" in str(exc.value)

    def test_an_empty_file_is_refused(self, tmp_path, campaign, bootstrapped):
        """An empty file would hydrate nothing and send the operator to the wizard with
        no idea why."""
        doc = tmp_path / "product.md"
        doc.write_text("   \n")

        with pytest.raises(OpenOutFindError) as exc:
            call_command("init", "--product-docs", str(doc), stdout=io.StringIO())

        assert exc.value.error_type == ErrorType.BAD_CONFIG
