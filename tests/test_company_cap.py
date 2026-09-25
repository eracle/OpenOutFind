# tests/test_company_cap.py
"""``OPENOUTFIND_MAX_PER_COMPANY`` — no company supplies more than N fits.

Covers ``core.company_cap`` and the two places it is read: discovery (a full company's
row is not accepted) and the qualification scan (a full company's stored lead is never
judged). The readiness check that refuses a malformed value is here too.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openoutfind.core import company_cap
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.models import Keyword, QueryNode
from openoutfind.core.pipeline import discover as discover_mod
from openoutfind.core.pipeline import select
from openoutfind.core.pipeline.discover import discover
from openoutfind.core.pipeline.qualify import fetch_qualification_candidates
from openoutfind.crm.models import Company, Deal, DealState, Lead, Outcome
from openoutfind.discovery import Page, company_key_for

ACME = "acme.com"


def _company(domain=ACME):
    return Company.objects.create(key=domain, name=domain.split(".")[0], domain=domain)


def _lead(url, company=None):
    return Lead.objects.create(
        profile_url=url,
        profile_text="head of quality at a food plant",
        embedding=np.ones(384, dtype=np.float32).tobytes(),
        company=company,
    )


def _fit(url, company, state=DealState.QUALIFIED):
    lead = _lead(url, company)
    outcome = Outcome.WRONG_FIT if state == DealState.FAILED else ""
    Deal.objects.create(lead=lead, state=state, outcome=outcome)
    return lead


def _row(url, company_name="Acme", domain=ACME):
    return {
        "contact_linkedin_profile_url": url,
        "contact_job_title": "quality manager",
        "contact_headline": "quality at acme",
        "company_name": company_name,
        "company_domain": domain,
    }


# ── the setting ──────────────────────────────────────────────────


class TestLimit:
    def test_unset_is_no_cap(self, configure):
        configure()

        assert company_cap.limit() is None
        assert company_cap.full_company_keys() == frozenset()

    def test_a_whole_number_is_the_cap(self, configure):
        configure(max_per_company="2")

        assert company_cap.limit() == 2

    @pytest.mark.parametrize("value", ["0", "two", "-1", "1.5"])
    def test_anything_else_is_bad_config(self, configure, value):
        configure(max_per_company=value)

        with pytest.raises(OpenOutFindError) as raised:
            company_cap.limit()

        assert raised.value.error_type == ErrorType.BAD_CONFIG
        assert "OPENOUTFIND_MAX_PER_COMPANY" in str(raised.value)


class TestFullCompanies:
    def test_a_company_is_full_at_the_cap(self, db, configure):
        configure(max_per_company="2")
        acme, other = _company(), _company("other.com")
        _fit("https://x/1", acme)
        _fit("https://x/2", acme)
        _fit("https://x/3", other)

        assert company_cap.full_company_keys() == frozenset({ACME})

    def test_rejections_and_disqualified_leads_do_not_count(self, db, configure):
        configure(max_per_company="2")
        acme = _company()
        _fit("https://x/1", acme)
        _fit("https://x/2", acme, state=DealState.FAILED)
        disqualified = _fit("https://x/3", acme)
        disqualified.disqualified = True
        disqualified.save()

        assert company_cap.full_company_keys() == frozenset()

    def test_leads_with_no_company_never_fill_anything(self, db, configure):
        configure(max_per_company="1")
        _fit("https://x/1", None)
        _fit("https://x/2", None)

        assert company_cap.full_company_keys() == frozenset()


# ── the qualification scan ───────────────────────────────────────


class TestQualificationScan:
    def test_a_full_companys_stored_leads_are_never_judged(self, db, configure):
        configure(max_per_company="2")
        acme, other = _company(), _company("other.com")
        _fit("https://x/1", acme)
        _fit("https://x/2", acme)
        waiting_at_acme = _lead("https://x/3", acme)
        elsewhere = _lead("https://x/4", other)
        no_company = _lead("https://x/5")

        candidates = fetch_qualification_candidates()

        assert elsewhere in candidates
        assert no_company in candidates
        assert waiting_at_acme not in candidates
        assert Lead.objects.filter(pk=waiting_at_acme.pk).exists()  # kept, not purged

    def test_below_the_cap_the_company_is_still_open(self, db, configure):
        configure(max_per_company="2")
        acme = _company()
        _fit("https://x/1", acme)
        waiting = _lead("https://x/2", acme)

        assert waiting in fetch_qualification_candidates()

    def test_no_cap_leaves_every_candidate(self, db, configure):
        configure()
        acme = _company()
        for n in range(3):
            _fit(f"https://x/{n}", acme)
        waiting = _lead("https://x/9", acme)

        assert waiting in fetch_qualification_candidates()


# ── discovery ────────────────────────────────────────────────────


def _node(pairs):
    node = QueryNode.objects.create(token_key=select.token_key(pairs))
    node.keywords.set(Keyword.rows_for(pairs))
    return node


class TestDiscovery:
    def test_a_full_companys_rows_are_not_accepted_and_the_node_still_advances(
        self, db, configure,
    ):
        configure(bettercontact_api_key="k", product_docs="p", campaign_target="t",
                  max_per_company="2")
        acme = _company()
        _fit("https://x/1", acme)
        _fit("https://x/2", acme)
        node = _node([("lead_job_title", "quality")])
        rows = [_row("https://linkedin.com/in/acme-third"),
                _row("https://linkedin.com/in/elsewhere", "Other", "other.com")]

        with patch.object(discover_mod, "_fetch", return_value=Page(rows, 200)):
            assert discover(company_cap.SiteConfig.load()) is True

        urls = set(Lead.objects.values_list("profile_url", flat=True))
        assert "https://linkedin.com/in/elsewhere" in urls
        assert "https://linkedin.com/in/acme-third" not in urls
        node.refresh_from_db()
        assert node.next_offset == select.DISCOVERY_PAGE_SIZE

    def test_without_a_cap_every_live_row_is_accepted(self, db, configure):
        configure(bettercontact_api_key="k", product_docs="p", campaign_target="t")
        acme = _company()
        _fit("https://x/1", acme)
        _fit("https://x/2", acme)
        _node([("lead_job_title", "quality")])

        with patch.object(discover_mod, "_fetch",
                          return_value=Page([_row("https://linkedin.com/in/acme-third")], 9)):
            discover(company_cap.SiteConfig.load())

        assert Lead.objects.filter(profile_url="https://linkedin.com/in/acme-third").exists()


class TestCompanyKeyFor:
    def test_matches_the_key_the_company_is_stored_under(self, db):
        assert company_key_for(_row("u", "Acme", "Acme.com")) == "acme.com"
        assert company_key_for(_row("u", "Acme Foods", None)) == "name:acme foods"

    def test_a_placeholder_or_missing_employer_is_no_company(self, db):
        assert company_key_for(_row("u", "Self employed", "selfemployed.com")) == ""
        assert company_key_for(_row("u", None, None)) == ""


# ── readiness ────────────────────────────────────────────────────


def test_a_malformed_cap_stops_the_run_before_any_work(site_config, configure):
    from openoutfind.core.readiness import check_ready

    configure(bettercontact_api_key="k", product_docs="p", campaign_target="t",
              ai_model="anthropic:claude-haiku-4-5", llm_api_key="k", max_per_company="lots")

    with pytest.raises(OpenOutFindError) as raised:
        check_ready()

    assert raised.value.error_type == ErrorType.BAD_CONFIG
