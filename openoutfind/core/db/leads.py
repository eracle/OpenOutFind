import logging

import numpy as np
from django.db import transaction

from openoutfind.crm.models import DealState

logger = logging.getLogger(__name__)


@transaction.atomic
def promote_lead_to_deal(profile_url: str, reason: str = ""):
    """Create a QUALIFIED Deal for a Lead.

    Returns the Deal. **Says nothing** — the verdict is announced by the step that
    reached it (``pipeline.qualify``), which prints the person and the reason as one
    line. Announcing here too printed the acceptance twice, once as a URL and a state
    name and once as a judgement about a person.
    """
    from openoutfind.crm.models import Lead, Deal

    lead = Lead.objects.filter(profile_url=profile_url).first()
    if not lead:
        raise ValueError(f"No Lead for {profile_url}")

    deal = Deal.objects.create(
        lead=lead,
        state=DealState.QUALIFIED,
        reason=reason,
    )

    return deal


def store_rows(rows: list[dict], country_code: str = "") -> tuple[list, int]:
    """Persist Lead Finder rows as embedded Leads; return every row's Lead and how many
    were new.

    Keyed on ``profile_url`` (the provider's per-person URL): a person already in the
    store comes back as the row it is, unchanged, since a page that returned them is
    evidence about its search either way. A row with no URL is skipped. New rows are
    embedded in one call — a round brings in up to 2,000.

    Stamps the firmographic ``profile_text`` (the LLM qualifier's input) and the embedding
    from the same row, so qualification never re-fetches anything. ``country_code`` is the
    market's (Lead Finder rows carry no ISO code) — blank means unknown, which the
    contacts-store geo-gate treats conservatively. The row's person fields
    (``person_for``) land as named columns and its employer as a shared ``Company`` row
    (``company_for``) — apart from both the text and the vector, because the export
    hands them on as fields and a name would only add noise to the embedding.
    """
    from openoutfind.core.ml.embeddings import embed_texts
    from openoutfind.crm.models import Lead
    from openoutfind.discovery import (
        company_for, person_for, profile_text_for, source_fields_for,
    )

    by_url = {row["contact_linkedin_profile_url"]: row for row in rows
              if row.get("contact_linkedin_profile_url")}
    known = {lead.profile_url: lead for lead in Lead.objects.filter(profile_url__in=by_url)}
    fresh = [url for url in by_url if url not in known]

    texts = [profile_text_for(by_url[url]) for url in fresh]
    vectors = embed_texts(texts) if texts else []
    for url, text, vector in zip(fresh, texts, vectors):
        row = by_url[url]
        known[url] = Lead.objects.create(
            profile_url=url,
            embedding=np.asarray(vector, dtype=np.float32).tobytes(),
            profile_text=text,
            source_fields=source_fields_for(row),
            country_code=country_code,
            company=company_for(row),
            **person_for(row),
        )
    return [known[url] for url in by_url], len(fresh)


def disqualify_lead(profile_url: str):
    """Set Lead.disqualified = True (account-level, permanent)."""
    from openoutfind.crm.models import Lead

    lead = Lead.objects.filter(profile_url=profile_url).first()
    if not lead:
        logger.warning("disqualify_lead: no Lead for %s", profile_url)
        return
    lead.disqualified = True
    lead.save(update_fields=["disqualified"])


# ``suppress_email`` lived here — an opt-out from an address suppressed every lead
# holding it and moved their open deals to UNSUBSCRIBED. It is **gone with the
# sending leg**, and that is a legal position, not a cleanup.
#
# Every mechanism that fed it was a *sending* mechanism: the `List-Unsubscribe`
# header, the `+unsub` alias on the operator's own sending address, the mailbox scan
# that read a client-generated unsubscribe, and the agent recognising a worded request
# in a reply. A finder that never contacts anyone is not the sender under CAN-SPAM /
# GDPR / CASL, so the duty is not inherited — it moves to OpenOutSend with the
# code. Instantly and Smartlead both block a suppressed address at import, which is
# where an opt-out now lands.
#
# ``Lead.disqualified`` itself stays: it is the permanent, account-level exclusion,
# the export filters it, and eleven candidate queries already read it. What is gone is
# the *inbound* path that used to set it from mail we received.
#
# The one suppression duty that does not leave is the hub's, and it never involved a
# sequencer: a person objecting to the shared contacts store is removed via the hub's
# own endpoint, store-wide. That obligation runs between the data subject and the hub.
