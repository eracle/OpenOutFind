# tests/test_discovery.py
"""Discovery slice — mock the BetterContact transport (`search_and_poll`) and
the embedder, so no network or ONNX model is needed."""
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from openoutfind import discovery
from openoutfind.core.pipeline.icp import WrittenSearch
from openoutfind.crm.models import Company


class TestSeniorityVocabulary:
    """`lead_seniority` is the one filter family with a closed vocabulary, and a
    value outside it returns an empty page instead of an error — so a written search
    constrains it in the schema."""

    def test_a_written_search_rejects_a_level_lead_finder_does_not_know(self):
        with pytest.raises(ValidationError):
            WrittenSearch(title="founder", seniority="other")

    def test_a_written_search_accepts_every_real_level(self):
        for level in discovery.LEAD_SENIORITIES:
            assert WrittenSearch(title="founder", seniority=level).seniority == level

    def test_vocabulary_is_derived_from_the_type_so_prompt_and_schema_cannot_drift(self):
        # The prompts render this tuple; the schema validates the same Literal.
        assert "mid-level" in discovery.LEAD_SENIORITIES
        assert "other" not in discovery.LEAD_SENIORITIES
        assert len(discovery.LEAD_SENIORITIES) == 12


_TRANSPORT = "openoutfind.enrichment.bettercontact.search_and_poll"


class TestSearch:
    def test_returns_leads_and_sends_the_filters(self):
        rows = [{"contact_full_name": "Alice"}]
        with patch(_TRANSPORT, return_value=({"leads": rows}, 1.2)) as call:
            fetched = discovery.search("k", {"lead_seniority": {"include": ["owner"]}},
                                       limit=200, offset=0, first_poll_s=1.5)

        assert fetched.page.leads == rows
        assert fetched.ready_seconds == 1.2
        api_key, url, body, first_poll = call.call_args.args
        assert (api_key, url, first_poll) == ("k", discovery.LEAD_FINDER_URL, 1.5)
        assert body == {"filters": {"lead_seniority": {"include": ["owner"]}},
                        "limit": 200, "offset": 0}

    def test_no_leads_key_is_empty_list(self):
        with patch(_TRANSPORT, return_value=({"status": "terminated"}, 1.0)):
            assert discovery.search("k", {}, 200, 0, 1.5).page.leads == []

    def test_surfaces_leads_found_at_offset_zero(self):
        # The count separates a genuinely empty query from a transport artifact: a
        # burst can answer a 71M-lead query with an empty page.
        body = {"leads": [], "summary": {"leads_found": 71403396}}
        with patch(_TRANSPORT, return_value=(body, 1.0)):
            page = discovery.search("k", {}, 200, 0, 1.5).page
        assert page.leads == [] and page.leads_found == 71403396

    def test_count_is_none_past_offset_zero(self):
        # leads_found is only trustworthy at offset 0 — past the end of ANY result set
        # the API reports 0, so it must never read as "empty" there.
        body = {"leads": [], "summary": {"leads_found": 0}}
        with patch(_TRANSPORT, return_value=(body, 1.0)):
            assert discovery.search("k", {}, 200, 400, 1.5).page.leads_found is None


class TestSourceFieldsFor:
    def test_the_headline_is_not_kept_as_job_title_vocabulary(self):
        """A headline is marketing prose, and its words are not job titles. It still
        reaches ``profile_text`` for the qualifier — only the vocabulary refuses it."""
        row = {
            "contact_job_title": "CTO",
            "contact_headline": "Building AI-powered agents for users",
            "contact_location_country": "United States",
        }
        stored = discovery.source_fields_for(row)

        assert stored == {
            "contact_job_title": "CTO",
            "contact_location_country": "United States",
        }
        assert "users" in discovery.profile_text_for(row)

    def test_seniority_is_kept_so_a_seniority_term_can_be_matched(self):
        stored = discovery.source_fields_for({"contact_seniority": "director"})

        assert stored == {"contact_seniority": "director"}

    def test_company_tags_are_kept_as_text_not_a_list_repr(self):
        stored = discovery.source_fields_for(
            {"company_keywords": ["cannabis", "thc vape", ""]})

        assert stored == {"company_keywords": "cannabis, thc vape"}


class TestAsPlace:
    def test_a_place_is_re_cased_for_the_index(self):
        """Lead Finder reports `United states` and matches only `United States`."""
        assert discovery.as_place("United states") == "United States"

    def test_a_connective_stays_lowercase(self):
        """Measured: `Bosnia and Herzegovina` counts 36,067, `Bosnia And Herzegovina` 0."""
        assert discovery.as_place("Bosnia and herzegovina") == "Bosnia and Herzegovina"


class TestProfileTextFor:
    def test_joins_fields_in_order_lowercased(self):
        row = {
            "contact_headline": "Head of Growth", "contact_industry": "SaaS",
            "contact_job_title": "CMO", "company_name": "Acme",
            "contact_seniority": "Founder", "company_industry": "B2B",
            "contact_location_state": "California",
            "contact_location_country": "United States",
        }
        assert discovery.profile_text_for(row) == (
            "head of growth saas cmo acme founder b2b california united states"
        )

    def test_tolerates_missing_and_null_fields(self):
        assert discovery.profile_text_for({"contact_headline": "Hi", "company_name": None}) == "hi"

    def test_skips_absent_fields_without_padding(self):
        # a sparse row stays short rather than padding out to a rich row's shape
        assert discovery.profile_text_for({"contact_job_title": "CEO"}) == "ceo"

    def test_drops_the_fabricated_company_free_text(self):
        # Lead Finder staples a fuzzy-matched company record onto every row (1-4
        # distinct per 100-row page), so these carry no per-lead signal to rank on.
        row = {
            "contact_job_title": "Founder",
            "company_description": "Meta's mission is to build the future of human connection",
            "company_keywords": ["bee keeper", "chaplin", "dive master"],
            "contact_location": "Berlin",
        }
        assert discovery.profile_text_for(row) == "founder"


class TestPersonFor:
    """The row's identity, kept as columns for the export rather than as text.

    Discovery knows one ``contact_full_name`` and nothing finer — first/last arrive
    later from the enrichment provider's own response, never from a split here.
    """

    def test_reads_the_name_and_title_the_provider_reports(self):
        row = {"contact_full_name": "Ada Lovelace", "contact_job_title": "CTO"}
        assert discovery.person_for(row) == {"full_name": "Ada Lovelace", "job_title": "CTO"}

    def test_an_unreported_field_is_none_not_an_empty_string(self):
        # One representation of "they didn't tell us", so the column never holds both.
        assert discovery.person_for({}) == {"full_name": None, "job_title": None}

    def test_null_and_empty_and_padded_values_all_normalise(self):
        row = {"contact_full_name": "  Ada Lovelace  ", "contact_job_title": ""}
        assert discovery.person_for(row) == {"full_name": "Ada Lovelace", "job_title": None}

    def test_no_first_or_last_name_is_invented(self):
        # The whole point: a split here would end up in a sequencer's {{first_name}}.
        assert set(discovery.person_for({"contact_full_name": "Ada Lovelace"})) == {
            "full_name", "job_title"}


@pytest.mark.django_db
class TestCompanyFor:
    def test_creates_the_company_and_keys_it_on_the_domain(self):
        company = discovery.company_for({"company_name": "Acme", "company_domain": "acme.com"})

        assert (company.name, company.domain, company.key) == ("Acme", "acme.com", "acme.com")

    def test_two_leads_at_one_firm_share_a_row(self):
        first = discovery.company_for({"company_name": "Acme", "company_domain": "Acme.com"})
        second = discovery.company_for({"company_name": "Acme", "company_domain": "acme.com"})

        assert first.pk == second.pk  # the key lowercases, so casing cannot fork the row

    def test_keeps_the_size_band(self):
        company = discovery.company_for({
            "company_name": "Acme", "company_domain": "acme.com",
            "company_employees_range_start": 51, "company_employees_range_end": "200"})

        assert (company.employees_min, company.employees_max) == (51, 200)
        assert company.size_text() == "51–200 employees (LinkedIn)"

    def test_a_later_band_replaces_the_stored_one_and_a_missing_one_does_not(self):
        discovery.company_for({"company_name": "Acme", "company_domain": "acme.com",
                               "company_employees_range_start": 11,
                               "company_employees_range_end": 50})
        discovery.company_for({"company_name": "Acme", "company_domain": "acme.com",
                               "company_employees_range_start": 1001,
                               "company_employees_range_end": 5000})
        company = discovery.company_for({"company_name": "Acme", "company_domain": "acme.com"})

        assert company.size_text() == "1,001–5,000 employees (LinkedIn)"

    def test_a_band_that_is_not_a_number_is_not_reported(self):
        company = discovery.company_for({"company_name": "Acme",
                                         "company_employees_range_start": "n/a"})

        assert company.size_text() == "not reported"

    def test_an_open_band_reads_as_a_floor(self):
        company = discovery.company_for({"company_name": "Acme",
                                         "company_employees_range_start": 10001})

        assert company.size_text() == "10,001+ employees (LinkedIn)"

    def test_a_company_with_no_domain_keys_on_its_name(self):
        company = discovery.company_for({"company_name": "Acme"})

        assert company.key == "name:acme"

    def test_a_row_naming_no_company_stores_nothing(self):
        assert discovery.company_for({"contact_job_title": "CTO"}) is None
        assert discovery.company_for({"company_name": "", "company_domain": None}) is None

    def test_a_placeholder_employer_stores_nothing_even_with_a_domain(self):
        """The provider fuzzy-matches "Stealth startup" to a real domain; storing it
        would pool every stealth founder under somebody else's website."""
        for name in ("Stealth startup", "Self-employed", "None at this time"):
            assert discovery.company_for(
                {"company_name": name, "company_domain": "harmonic.ai"}) is None
        assert not Company.objects.exists()

    def test_a_name_merely_containing_a_placeholder_word_is_a_company(self):
        company = discovery.company_for({"company_name": "Stealth Labs",
                                         "company_domain": "stealthlabs.io"})

        assert company.name == "Stealth Labs"


class TestEmbedProfile:
    def test_the_vector_is_the_profile_alone(self):
        """No query terms ride it: a search ranked on its people's GP scores must not be
        scored on its own words."""
        with patch("openoutfind.core.ml.embeddings.embed_text", return_value=np.ones(384)) as embed:
            discovery.embed_profile("head of growth acme")
        embed.assert_called_once_with("head of growth acme")


class TestDescribeFilters:
    """The log rendering of a Lead Finder filter set. Pure (no colour) so the
    call sites own presentation and these stay readable."""

    def test_renders_the_families_a_mutation_varies(self):
        assert discovery.describe_filters({
            "company_headcount_min": 1, "company_headcount_max": 20,
            "lead_job_title": {"include": ["Founder", "CTO"], "exact_match": False},
            "lead_location": {"include": ["United States"]},
        }) == "headcount 1–20 · job_title Founder, CTO · location United States"

    def test_collapses_the_two_headcount_bounds_into_one_range(self):
        """min/max are two keys describing one thing; they read as one."""
        out = discovery.describe_filters({"company_headcount_min": 1, "company_headcount_max": 20})
        assert out == "headcount 1–20"

    def test_marks_an_open_ended_headcount_bound(self):
        assert discovery.describe_filters({"company_headcount_min": 50}) == "headcount 50–?"
        assert discovery.describe_filters({"company_headcount_max": 50}) == "headcount ?–50"

    def test_keeps_exact_match_because_it_changes_what_matches(self):
        assert discovery.describe_filters(
            {"lead_job_title": {"include": ["SDR"], "exact_match": True}}
        ) == "job_title SDR (exact)"

    def test_strips_the_lead_prefix(self):
        """Every family we search is ``lead_*`` (the only ``company_*`` keys are the
        two headcount bounds, consumed by the range renderer) — so only ``lead_`` is
        stripped; an unmodeled key stays verbatim (see the unknown-key test)."""
        assert discovery.describe_filters({
            "lead_department": {"include": ["Sales"]},
            "lead_skills": {"include": ["negotiation"]},
        }) == "department Sales · skills negotiation"

    def test_empty_filters_say_so_rather_than_render_blank(self):
        """An all-unset proposal means 'the LLM is dry' — it must not read as a query."""
        assert discovery.describe_filters({}) == "(no filters)"

    def test_survives_an_empty_include_list(self):
        assert discovery.describe_filters({"lead_skills": {"include": []}}) == "skills (none)"

    def test_renders_an_unknown_key_rather_than_dropping_it(self):
        """Filters are free-form dicts; a key we don't model must still be visible."""
        assert discovery.describe_filters({"some_new_filter": "x"}) == "some_new_filter x"


class TestDescribeCompanyAxis:
    def test_the_company_axis_reads_as_company(self):
        from openoutfind.core.pipeline.query import Terms

        rendered = discovery.describe_filters(
            Terms.written(title="purchasing", tag="cannabis").to_filters())

        assert "company cannabis" in rendered
        assert "job_title purchasing" in rendered


class TestWrittenSearchAsTerms:
    """A written search keeps what goes together: the title's words, the tag's phrase."""

    def test_the_tag_searches_the_company_and_the_title_the_person(self):
        from openoutfind.core.pipeline.icp import WrittenSearch, as_terms

        terms = as_terms(WrittenSearch(title="Head of Procurement", tag="Cannabis Extraction",
                                       state="california", seniority="head"),
                         "united states")

        assert terms.title == {"head", "procurement"}
        assert terms.tag == ("cannabis", "extraction")
        assert terms.place == "California, United States"
        assert terms.seniority == "head"

    def test_no_state_is_the_whole_market(self):
        from openoutfind.core.pipeline.icp import WrittenSearch, as_terms

        assert as_terms(WrittenSearch(title="founder"), "united kingdom").place == "United Kingdom"
