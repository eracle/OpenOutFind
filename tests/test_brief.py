# tests/test_brief.py
"""The brief — the rules every prompt is written from, stored once per store."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core import agent_qualify
from openoutfind.core.pipeline import brief as brief_module


@pytest.fixture(autouse=True)
def _reset_agent_qualify():
    """A contextvar carries state past a test's own scope otherwise."""
    yield
    agent_qualify._active.set(False)


@pytest.fixture
def described(configure, operator):
    return configure(product_docs="We bake bread ovens.", campaign_target="Bakeries in Italy.",
                     ai_model="anthropic:claude-sonnet-5-5")


@pytest.mark.django_db
class TestAdopting:
    def test_a_given_brief_replaces_the_one_in_force(self):
        from openoutfind.core.models import Brief

        brief_module.adopt("A fit is: bakers.")
        brief_module.adopt("  A fit is: bakers and pastry chefs.\n")

        assert Brief.objects.count() == 1
        assert brief_module.stored().rules == "A fit is: bakers and pastry chefs."
        assert brief_module.stored().written_by == brief_module.GIVEN

    def test_the_same_brief_again_leaves_the_row_alone(self):
        brief_module.adopt("A fit is: bakers.")
        first = brief_module.stored().pk

        brief_module.adopt("A fit is: bakers.")

        assert brief_module.stored().pk == first


@pytest.mark.django_db
class TestEnsuringRules:
    def test_a_stored_brief_is_read_without_asking_the_model(self, described):
        brief_module.adopt("A fit is: bakers.")

        with patch.object(brief_module, "_write_with_llm") as write:
            assert brief_module.ensure_rules(described) == "A fit is: bakers."

        write.assert_not_called()

    def test_the_model_writes_one_once_and_it_is_kept(self, described):
        with patch.object(brief_module, "_write_with_llm",
                          return_value=" Not a fit: florists. ") as write:
            first = brief_module.ensure_rules(described)
            second = brief_module.ensure_rules(described)

        assert first == second == "Not a fit: florists."
        write.assert_called_once()
        assert brief_module.stored().written_by == "anthropic:claude-sonnet-5-5"

    def test_a_failed_write_stores_nothing_and_propagates(self, described):
        with patch.object(brief_module, "_write_with_llm", side_effect=RuntimeError("outage")):
            with pytest.raises(RuntimeError):
                brief_module.ensure_rules(described)

        assert brief_module.stored() is None

    def test_no_icp_text_and_agent_mode_write_nothing(self, described):
        from dataclasses import replace

        with patch.object(brief_module, "_write_with_llm") as write:
            agent_qualify.enable(None)
            assert brief_module.ensure_rules(described) == ""
            agent_qualify._active.set(False)
            blank = replace(described, product_docs="", campaign_target="")
            assert brief_module.ensure_rules(blank) == ""

        write.assert_not_called()


class TestPrompts:
    def _render(self, name, **values):
        import jinja2

        from openoutfind.core.conf import PROMPTS_DIR

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
        return env.get_template(name).render(product_docs="ovens", campaign_target="bakeries",
                                             profile_text="baker", seniorities=[], count=3,
                                             existing=[], **values)

    @pytest.mark.parametrize("name", ["qualify_lead.j2", "icp_filters.j2", "anchor_profiles.j2"])
    def test_every_prompt_carries_the_rules(self, name):
        assert "Not a fit: florists." in self._render(name, rules="Not a fit: florists.")

    def test_without_rules_the_verdict_prompt_has_no_rules_section(self):
        assert "The rules decide" not in self._render("qualify_lead.j2", rules="")


@pytest.mark.django_db
class TestTheVerdictReadsTheBrief:
    def test_the_qualifier_is_handed_the_stored_rules(self, described):
        from openoutfind.core.ml.qualifier import BayesianQualifier
        from openoutfind.core.pipeline.qualify import run_qualification
        from openoutfind.crm.models import Lead

        Lead.objects.create(profile_url="https://www.linkedin.com/in/alice/",
                            profile_text="baker at forno",
                            embedding=np.ones(384, dtype=np.float32).tobytes())
        brief_module.adopt("A fit is: bakers.")

        with patch("openoutfind.core.ml.qualifier.qualify_with_llm",
                   return_value=(1, "Bakes for Forno.")) as qualify, \
                patch("openoutfind.core.db.leads.promote_lead_to_deal"):
            run_qualification(described, BayesianQualifier(seed=42))

        assert qualify.call_args.kwargs["rules"] == "A fit is: bakers."

    def test_an_agent_is_handed_the_rules_with_the_candidate(self, described):
        from openoutfind.core.pipeline.qualify import _candidate_payload
        from openoutfind.crm.models import Lead

        lead = Lead.objects.create(profile_url="https://www.linkedin.com/in/bob/",
                                   profile_text="baker")
        assert _candidate_payload(lead)["rules"] == ""

        brief_module.adopt("A fit is: bakers.")
        assert _candidate_payload(lead)["rules"] == "A fit is: bakers."
