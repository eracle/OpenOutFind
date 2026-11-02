"""Synthetic paid-source fixtures: no provider, model or hub network calls."""
import json
from unittest.mock import MagicMock

import pytest
import requests

from openoutfind import companyenrich as api
from openoutfind.core import discovery_source
from openoutfind.core.config import SiteConfig
from openoutfind.core.errors import OpenOutFindError
from openoutfind.core.models import CompanyEnrichPage
from openoutfind.core.pipeline.companyenrich import CompanyEnrichSource, from_options
from openoutfind.enrichment.provider import ProviderUnavailable

pytestmark = pytest.mark.django_db(transaction=True)
QUERY = {"domains": ["example.com"], "positionQuery": ["Founder"], "page": 1, "pageSize": 10}


@pytest.fixture
def page():
    return {"items": [{"id": 9223372036854775807, "name": "Alex Example",
                       "first_name": "Alex", "last_name": "Example", "position": "Founder",
                       "seniority": "founder", "socials": {"linkedin_url": "https://www.linkedin.com/in/alex-example/"},
                       "location": {"country": {"code": "US", "name": "United States"}},
                       "experiences": [
                           {"type": "company", "isCurrent": False,
                            "company": {"id": "old", "name": "Old Employer"}},
                           {"type": "company", "isCurrent": True,
                            "company": {"id": "new", "name": "Example", "domain": "example.com", "industry": "Software"}}]}],
            "page": 1, "pageSize": 10, "totalItems": 23, "totalPages": 3}


@pytest.fixture
def http(mocker, page):
    response = MagicMock()
    response.status_code = 200
    response.iter_content.return_value = [json.dumps(page).encode()]
    response.__enter__.return_value = response
    session = MagicMock()
    session.__enter__.return_value = session
    session.post.return_value = response
    mocker.patch.object(api.requests, "Session", return_value=session)
    return session, response


def test_single_request_exact_schema_and_full_page(http, page):
    session, _ = http
    result = api.search("test-secret", QUERY)
    assert result == page
    assert result["items"][0]["id"] == 9223372036854775807
    session.post.assert_called_once_with(
        api.API_URL, json=QUERY,
        headers={"Authorization": "Bearer test-secret", "Accept": "application/json"},
        timeout=(5, 30), allow_redirects=False, stream=True)
    row = api.rows_for(result)[0]
    assert row["company_name"] == "Example"
    assert row["_first_name"] == "Alex"
    assert row["_country_code"] == "US"
    assert row["_discovery_source"] == "companyenrich"
    assert "email" not in row
    assert "contact_headline" not in row


@pytest.mark.parametrize("status,error", [(301, "provider_unavailable"), (400, "provider_unavailable"),
    (401, "provider_auth"), (402, "provider_out_of_credits"), (403, "provider_auth"),
    (429, "provider_rate_limited"), (503, "provider_unavailable")])
def test_http_errors_are_not_empty_results_or_retried(http, status, error):
    session, response = http
    response.status_code = status
    response.text = "test-secret raw private response"
    with pytest.raises(ProviderUnavailable) as caught:
        api.search("test-secret", QUERY)
    assert caught.value.error_type == error
    assert "test-secret" not in str(caught.value)
    assert "private response" not in str(caught.value)
    session.post.assert_called_once()
    response.iter_content.assert_not_called()


def test_transport_failure_is_ambiguous_and_redacted(http):
    session, _ = http
    session.post.side_effect = requests.Timeout("test-secret")
    with pytest.raises(ProviderUnavailable, match="outcome unknown") as caught:
        api.search("test-secret", QUERY)
    assert "test-secret" not in str(caught.value)
    session.post.assert_called_once()


@pytest.mark.parametrize("body", [b"not-json", b"null", b'{"items":{}}',
                                  b'{"items":[{"id":"wrong"}]}', b'{"items":[{"id":true}]}'])
def test_invalid_response(http, body):
    _, response = http
    response.iter_content.return_value = [body]
    with pytest.raises(ProviderUnavailable, match="Invalid"):
        api.search("test-secret", QUERY)


def test_response_size_bound(http, monkeypatch):
    _, response = http
    monkeypatch.setattr(api, "MAX_RESPONSE_BYTES", 4)
    response.iter_content.return_value = [b"12345"]
    with pytest.raises(ProviderUnavailable, match="exceeded"):
        api.search("test-secret", QUERY)


@pytest.mark.parametrize("query", [{}, {"domains": ["example.com"], "pageSize": 101},
    {"domains": ["example.com"], "page": True}, {"companyFilter": {"search": {"query": "software"}}},
    {"countries": ["USA"]}, {"seniority": ["invented"]}, {"department": ["invented"]},
    {"query": "x", "expand": "email"}, {"domains": "example.com"},
    {"atCurrentCompanyAfter": "yesterday"}])
def test_bad_queries_never_reach_http(http, query):
    session, _ = http
    with pytest.raises(OpenOutFindError):
        api.search("test-secret", query)
    session.post.assert_not_called()


def test_optional_exclusion_education_and_date_filters(tmp_path):
    query = {**QUERY, "exclude": {"seniority": ["intern"]},
             "education": {"institutionNameQuery": ["Example University"]},
             "atCurrentCompanyAfter": "2024-01-01T00:00:00Z"}
    path = tmp_path / "query.json"
    path.write_text(json.dumps(query))
    assert api.read_query(str(path)) == query


@pytest.mark.parametrize("url", ["https://evil.example/in/person", "https://www.linkedin.com/company/person",
    "https://www.linkedin.com:bad/in/person", "https://user@www.linkedin.com/in/person", "http://www.linkedin.com/in/person", ""])
def test_unusable_profile_urls_are_skipped(page, url):
    page["items"][0]["socials"]["linkedin_url"] = url
    assert api.rows_for(page) == []


def test_historical_job_is_not_current_company(page):
    page["items"][0]["experiences"][1]["isCurrent"] = False
    row = api.rows_for(page)[0]
    assert row["company_name"] == ""
    assert row["company_domain"] == ""


def test_all_three_flags_and_sufficient_budget_required(tmp_path):
    path = tmp_path / "query.json"
    path.write_text(json.dumps(QUERY))
    assert from_options({}) is None
    with pytest.raises(OpenOutFindError, match="together"):
        from_options({"companyenrich_query": str(path)})
    with pytest.raises(OpenOutFindError, match="at least 20"):
        CompanyEnrichSource(QUERY, "page-one", 19)
    with pytest.raises(OpenOutFindError):
        CompanyEnrichSource(QUERY, "", 20)
    assert CompanyEnrichSource(QUERY, "page-one", 20).maximum_credits == 20


def test_reserved_page_reused_across_invocations_and_names_preserved(http):
    from openoutfind.crm.models import Lead

    session, _ = http
    config = SiteConfig(companyenrich_api_key="test-secret")
    first = CompanyEnrichSource(QUERY, "page-one", 20)
    assert first.discover(config)
    assert not first.discover(config)
    assert not CompanyEnrichSource(QUERY, "page-one", 20).discover(config)
    session.post.assert_called_once()
    record = CompanyEnrichPage.objects.get(request_id="page-one")
    assert record.imported and record.response["totalPages"] == 3
    lead = Lead.objects.get(synthetic=False)
    assert lead.first_name == "Alex" and lead.last_name == "Example"
    assert lead.email is None
    assert lead.country_code == "US"
    assert lead.discovery_source == "companyenrich"
    assert lead.company.domain == "example.com"


def test_ambiguous_failure_never_replays(http):
    session, _ = http
    session.post.side_effect = requests.Timeout("private")
    config = SiteConfig(companyenrich_api_key="test-secret")
    with pytest.raises(ProviderUnavailable):
        CompanyEnrichSource(QUERY, "uncertain", 20).discover(config)
    assert CompanyEnrichPage.objects.get(request_id="uncertain").response is None
    with pytest.raises(ProviderUnavailable, match="already submitted"):
        CompanyEnrichSource(QUERY, "uncertain", 20).discover(config)
    session.post.assert_called_once()
    with pytest.raises(ProviderUnavailable, match="different query"):
        CompanyEnrichSource({**QUERY, "page": 2}, "uncertain", 20).discover(config)
    session.post.assert_called_once()


def test_saved_response_resumes_after_local_embedding_failure(http, mocker):
    session, _ = http
    config = SiteConfig(companyenrich_api_key="test-secret")
    embedding = mocker.patch("openoutfind.core.ml.embeddings.embed_texts", side_effect=RuntimeError("local failure"))
    with pytest.raises(RuntimeError, match="local failure"):
        CompanyEnrichSource(QUERY, "resume", 20).discover(config)
    assert CompanyEnrichPage.objects.get(request_id="resume").response is not None
    import numpy as np
    embedding.side_effect = lambda texts: np.ones((len(texts), 384))
    assert CompanyEnrichSource(QUERY, "resume", 20).discover(config)
    session.post.assert_called_once()


def test_source_selection_is_scoped_and_default_walk_unchanged(mocker):
    from openoutfind.core.pipeline.discover import discover

    source = CompanyEnrichSource(QUERY, "page-one", 20)
    paid = mocker.patch.object(source, "discover", return_value=True)
    walk = mocker.patch("openoutfind.core.pipeline.walk.run_round", return_value=object())
    mocker.patch("openoutfind.enrichment.bettercontact.is_configured", return_value=True)
    config = SiteConfig(product_docs="example", companyenrich_api_key="unused")
    with discovery_source.selected(source):
        assert discover(config)
    assert discovery_source.current() is None
    paid.assert_called_once_with(config)
    walk.assert_not_called()
    assert discover(config)
    walk.assert_called_once()


def test_provenance_survives_duplicate_and_blocks_all_hub_paths(http, mocker):
    from openoutfind.contacts import service
    from openoutfind.core.db.leads import store_rows
    from openoutfind.crm.models import Lead

    known = Lead.objects.create(profile_url="https://www.linkedin.com/in/alex-example",
                                full_name="Original", country_code="US")
    CompanyEnrichSource(QUERY, "page-one", 20).discover(SiteConfig(companyenrich_api_key="test"))
    known.refresh_from_db()
    assert known.discovery_source == "companyenrich"
    assert known.full_name == "Original"
    store_rows([{"contact_linkedin_profile_url": known.profile_url}])
    known.refresh_from_db()
    assert known.discovery_source == "companyenrich"
    token = mocker.patch.object(service, "token_in_hand")
    page_token = mocker.patch.object(service, "_page_token")
    send = mocker.patch.object(service, "_send")
    assert service.resolve(known) is None
    service.contribute(known, ["alex@example.com"], "bettercontact")
    row = {"contact_linkedin_profile_url": known.profile_url,
           "company_linkedin_url": "https://www.linkedin.com/company/example",
           "_discovery_source": "companyenrich"}
    service.share_profiles([row], "US")
    service.share_companies([row], "US")
    token.assert_not_called()
    page_token.assert_not_called()
    send.assert_not_called()


def test_readiness_uses_only_explicit_source_key(configure):
    from openoutfind.core.readiness import missing_variables

    configure(product_docs="product", campaign_target="target", ai_model="openai:test", llm_api_key="fake")
    with discovery_source.selected(CompanyEnrichSource(QUERY, "page-one", 20)):
        assert missing_variables() == {"companyenrich": ["OPENOUTFIND_COMPANYENRICH_API_KEY"]}
    assert missing_variables() == {"bettercontact": ["OPENOUTFIND_BETTERCONTACT_API_KEY"]}


def test_cli_selection_resets_on_exception(tmp_path, mocker):
    from openoutfind.core.management.commands.find import Command

    path = tmp_path / "query.json"
    path.write_text(json.dumps(QUERY))
    command = Command()
    options = vars(command.create_parser("outfind", "find").parse_args([
        "1", "--companyenrich-query", str(path), "--companyenrich-request-id", "page-one",
        "--discovery-credit-budget", "20"]))
    def interrupted(**options):
        assert discovery_source.current().maximum_credits == 20
        raise RuntimeError("interrupted")
    mocker.patch.object(command, "_handle", side_effect=interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        command.handle(**options)
    assert discovery_source.current() is None


def test_cli_to_local_lead_and_export_with_one_paid_request(http, tmp_path, configure, mocker):
    """Exercise the real CLI, readiness, bounded job, ingestion and CSV. Only the
    qualification verdict is supplied locally instead of paying an LLM in a test."""
    import csv
    import io
    from django.core.management import call_command
    from openoutfind.core.db.leads import promote_lead_to_deal
    from openoutfind.core.pipeline.discover import discover
    from openoutfind.crm.models import Lead

    configure(product_docs="Example product", campaign_target="Founders",
              ai_model="openai:test", llm_api_key="test", companyenrich_api_key="test")
    path = tmp_path / "query.json"
    path.write_text(json.dumps(QUERY))
    mocker.patch("openoutfind.core.management.commands.find.ensure_database")
    def qualify_one(config, **kwargs):
        assert not kwargs["buy_addresses"]
        discover(config)
        lead = Lead.objects.filter(discovery_source="companyenrich").first()
        promote_lead_to_deal(lead.profile_url, "Fixture: matches the target")
        return True
    mocker.patch("openoutfind.core.cycle.run_one_action", side_effect=qualify_one)
    output = io.StringIO()
    call_command("find", "1", companyenrich_query=str(path),
                 companyenrich_request_id="cli-page", discovery_credit_budget=20,
                 stdout=output, stderr=io.StringIO(), batch=True)
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert len(rows) == 1
    assert rows[0]["first_name"] == "Alex"
    assert rows[0]["email"] == ""
    assert rows[0]["website"] == "example.com"
    assert rows[0]["reason"] == "Fixture: matches the target"
    http[0].post.assert_called_once()
    assert discovery_source.current() is None


def test_find_zero_with_paid_flags_never_purchases(http, tmp_path, configure, mocker):
    import io
    from django.core.management import call_command

    configure(product_docs="Example product", campaign_target="Founders",
              ai_model="openai:test", llm_api_key="test", companyenrich_api_key="test")
    path = tmp_path / "query.json"
    path.write_text(json.dumps(QUERY))
    mocker.patch("openoutfind.core.management.commands.find.ensure_database")
    call_command("find", "0", companyenrich_query=str(path),
                 companyenrich_request_id="zero-page", discovery_credit_budget=20,
                 stdout=io.StringIO(), stderr=io.StringIO(), batch=True)
    assert not CompanyEnrichPage.objects.exists()
    http[0].post.assert_not_called()


def test_source_rejects_uncommitted_outer_transaction(http):
    from django.db import transaction

    with transaction.atomic(), pytest.raises(RuntimeError, match="autocommit"):
        CompanyEnrichSource(QUERY, "uncommitted", 20).discover(
            SiteConfig(companyenrich_api_key="test"))
    assert not CompanyEnrichPage.objects.exists()
    http[0].post.assert_not_called()


def test_later_free_pages_cannot_clear_local_only_identity(http, mocker):
    from openoutfind.contacts import service
    from openoutfind.crm.models import Lead

    CompanyEnrichSource(QUERY, "private", 20).discover(SiteConfig(companyenrich_api_key="test"))
    token = mocker.patch.object(service, "token_in_hand")
    page_token = mocker.patch.object(service, "_page_token")
    send = mocker.patch.object(service, "_send")
    # Independently fetched free rows have no CompanyEnrich marker and may use a
    # different URL spelling. The persisted local-only identity still wins.
    row = {"contact_linkedin_profile_url": "https://linkedin.com/in/alex-example/?tracking=1",
           "company_linkedin_url": "https://www.linkedin.com/company/example",
           "company_name": "Example"}
    service.share_profiles([row], "US")
    service.share_companies([row], "US")
    duplicate = Lead.objects.create(profile_url=row["contact_linkedin_profile_url"],
                                    discovery_source="bettercontact", country_code="US")
    assert service.resolve(duplicate) is None
    service.contribute(duplicate, ["fixture@example.com"], "bettercontact")
    token.assert_not_called()
    page_token.assert_not_called()
    send.assert_not_called()


def test_unknown_current_employer_and_invalid_country_are_not_guessed(page):
    page["items"][0]["experiences"] = [{"type": "unknown", "isCurrent": True,
                                         "companyName": "Unmatched employer"}]
    page["items"][0]["location"]["country"]["code"] = "USA"
    row = api.rows_for(page)[0]
    assert row["company_name"] == "Unmatched employer"
    assert row["company_domain"] == ""
    assert row["_country_code"] == ""
