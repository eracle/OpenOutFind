"""A single paid source page, separately authorized from the free discovery walk."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from django.db import transaction

from openoutfind import companyenrich
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.enrichment.provider import ProviderUnavailable

logger = logging.getLogger(__name__)


@dataclass
class CompanyEnrichSource:
    query: dict
    request_id: str
    credit_budget: int
    required_field: str = field(default="companyenrich_api_key", init=False)
    _done: bool = field(default=False, init=False)
    _used: bool = field(default=False, init=False)

    def __post_init__(self):
        self.query = companyenrich.validate_query(self.query)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.request_id):
            raise OpenOutFindError(ErrorType.BAD_CONFIG,
                                  "--companyenrich-request-id needs 1–64 letters, digits, _, . or -")
        if type(self.credit_budget) is not int or self.credit_budget < self.maximum_credits:
            raise OpenOutFindError(ErrorType.BAD_CONFIG,
                                  f"--discovery-credit-budget must be at least {self.maximum_credits} "
                                  "(2 CompanyEnrich credits per requested profile)")

    @property
    def maximum_credits(self) -> int:
        return companyenrich.CREDITS_PER_PERSON * self.query["pageSize"]

    def report(self) -> dict:
        return {"source": "companyenrich", "request_id": self.request_id,
                "maximum_credits": self.maximum_credits, "credit_budget": self.credit_budget,
                "page": self.query["page"], "page_size": self.query["pageSize"],
                "used": self._used, "hub_sharing": False}

    def discover(self, site_config) -> bool:
        from openoutfind.core.company_cap import full_company_keys
        from openoutfind.core.db.leads import store_rows
        from openoutfind.core.models import CompanyEnrichPage
        from openoutfind.discovery import company_key_for

        if self._done:
            return False
        if not site_config.companyenrich_api_key:
            raise ProviderUnavailable("Set OPENOUTFIND_COMPANYENRICH_API_KEY", ErrorType.NO_CREDENTIAL)
        # get_or_create's unique request ID arbitrates concurrent invocations. The
        # reservation is committed before HTTP; only its creator may submit the request.
        if transaction.get_connection().in_atomic_block:
            raise RuntimeError("CompanyEnrich discovery requires an autocommit boundary")
        page, created = CompanyEnrichPage.objects.get_or_create(
            request_id=self.request_id,
            defaults={"query": self.query, "maximum_credits": self.maximum_credits})
        if page.query != self.query:
            raise ProviderUnavailable("CompanyEnrich request ID already belongs to a different query",
                                      ErrorType.BAD_CONFIG)
        if created:
            logger.info("CompanyEnrich: one page, at most %d credits; provider data stays local",
                        self.maximum_credits)
            data = companyenrich.search(site_config.companyenrich_api_key, self.query)
            page.response = data
            page.save(update_fields=["response"])
        elif page.response is None:
            raise ProviderUnavailable(
                "CompanyEnrich request was already submitted or its outcome is unknown; "
                "not retried. Inspect provider usage before explicitly authorizing a new request ID.")
        self._used = True
        self._done = True
        if page.imported:
            return False
        full = full_company_keys()
        rows = [row for row in companyenrich.rows_for(page.response)
                if company_key_for(row) not in full]
        # No share_profiles/share_companies. Source provenance persists even for a
        # previously known URL, so a later email resolver cannot give that identity back.
        with transaction.atomic():
            page = CompanyEnrichPage.objects.select_for_update().get(pk=page.pk)
            if page.imported:
                return False
            leads, created_count = store_rows(rows, discovery_source="companyenrich")
            page.imported = True
            page.save(update_fields=["imported"])
        logger.info("CompanyEnrich: %d profiles returned, %d usable, %d new; no emails supplied",
                    len(page.response["items"]), len(leads), created_count)
        return True


def from_options(options) -> CompanyEnrichSource | None:
    path = options.get("companyenrich_query")
    request_id = options.get("companyenrich_request_id")
    budget = options.get("discovery_credit_budget")
    if not path and not request_id and budget is None:
        return None
    if not path or not request_id or budget is None:
        raise OpenOutFindError(ErrorType.BAD_CONFIG,
                              "CompanyEnrich requires --companyenrich-query, "
                              "--companyenrich-request-id and --discovery-credit-budget together")
    return CompanyEnrichSource(companyenrich.read_query(path), request_id, budget)
