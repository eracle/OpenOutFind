# openoutfind/core/pipeline/icp.py
"""ICP generators — the LLM writes the install's cold-start priors, in two shapes.

The same two inputs (``product_docs + campaign_target``) are the only prior available
before any lead has been judged, and the engine needs them expressed two ways:

- ``generate_seed`` — the ICP as a **query**: one value per family (a title, a
  seniority, a country, a size band), the single most precise conjunction the model can
  name. That conjunction is the whole starting **pool**, so the initial maximal set is
  exactly one query — the seed. Breadth is not seeded; it grows from the leads that
  qualify (``mint.py``), which add more values per family and so more maximals for the
  selector to rank.
- ``generate_anchors`` — the ICP as **profiles**: a few invented leads that would be
  ideal fits, written in the shape ``discovery.profile_text_for`` produces. Embedded and
  handed to the GP as synthetic positives (``BayesianQualifier.set_anchors``), they are
  what lets the model fit at all when every real verdict so far is a rejection — a
  single-class label set never produces a posterior, so without them BALD, P(f>0.5), and
  every piece of steering that reads them stay unavailable for the whole cold phase. They
  are permanent: once written they stand alongside whatever real positives arrive, for
  the install's whole life.

Profiles rather than the product text itself because the space they have to land in is
one of *lead* embeddings: marketing prose about the product embeds nowhere near a row of
firmographics, so it would anchor the model in a region no candidate occupies. They are
also embedded **without** query terms (unlike a discovered lead, whose retrieving query
rides its embedding) — an anchor is a claim about what a good lead looks like, not about
which query to run, and folding the seed's keywords in would have discovery score the
seed highly on the strength of our own guess.

One value per family, never headcount as a range to search: the size band is a single
ICP attribute that rides every maximal unchanged. See ``discovery.filters_for``.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jinja2
import numpy as np
from pydantic import BaseModel, Field
from termcolor import colored

from openoutfind.core.conf import PROMPTS_DIR
from openoutfind.discovery import LEAD_SENIORITIES, Seniority

logger = logging.getLogger(__name__)

# How many synthetic ideal profiles anchor a cold start. Several rather than one so the
# positive region is outlined rather than pinned to a single hallucination, but few
# enough that a handful of real labels outweighs them.
ANCHOR_COUNT = 3


class ICPSpec(BaseModel):
    """The LLM's provider-agnostic ICP output — the walk's opening **vocabulary**.

    Not a query. The seed used to be "one value per family, the single most precise
    conjunction", which is what the clause model needed; the walk now conjoins tokens
    itself against measured feedback, so what it wants from the LLM is *words worth
    trying*, and as many as the ICP genuinely implies.

    **``domain_keywords`` is the field that makes an ICP an ICP.** The old spec had
    nowhere to put "what the target company actually does" — no field for it — so a
    health-and-wellness ICP seeded on ``content``/``lead``/``united``/``states``
    and every query it could compose selected for *role* while being blind to
    *industry*. The obvious home would be ``lead_industry``, and that field is inert:
    a nonsense value returns the identical count to no filter at all (§8 of the roadmap
    card). But domain words are demonstrably alive in ``lead_job_title``, which matches
    title *and* headline text — ``saas`` counts 3,306, ``startup`` 6,223, ``llm`` 1,214,
    ``agentic`` 1,213, ``stealth`` 932 (§10). So they go there, alongside the role words,
    and the frontier conjoins the two.

    ``seniority`` is typed to Lead Finder's vocabulary, not ``str``: an unknown level
    returns an empty page rather than an error, wasting a fetch. Everything else is free
    text — a token the index doesn't carry is a normal empty page, one fetch spent, and
    the walk retires it.
    """

    role_keywords: list[str] = Field(
        default_factory=list,
        description="Single lowercase words from the job titles the buyer holds — "
                    "'founder', 'head', 'marketing', 'content'. Words, never phrases.",
    )
    domain_keywords: list[str] = Field(
        default_factory=list,
        description="Single lowercase words naming what the target company does or "
                    "sells — 'wellness', 'supplement', 'nutrition', 'saas'. Words, "
                    "never phrases.",
    )
    seniority: Seniority | None = None
    location: str = ""
    headcount_min: int = 1
    headcount_max: int = 10000
    country_code: str = ""


# Which ``ICPSpec`` attrs feed which search axis. Both keyword lists land in
# ``lead_job_title`` — the only free-text axis, and the one that matches headline text as
# well as titles, which is what makes a domain word like ``wellness`` reachable at all.
# Headcount is absent: numbers riding every query, not search terms.
_SEED_FIELDS = (
    ("lead_job_title", "role_keywords"),
    ("lead_job_title", "domain_keywords"),
    ("lead_seniority", "seniority"),
    ("lead_location", "location"),
)


def _seed_keywords(spec: ICPSpec) -> list[tuple[str, str]]:
    """The ICP as ``(field, token)`` keywords — the vocabulary the walk opens with.

    **A job title is split into words; the closed axes are not.** Lead Finder reads
    ``"Head of Growth"`` as three ANDed words, a query narrow enough to be empty before
    the walk has learned anything, so splitting hands the frontier three separate
    one-token nodes and lets *measurement* decide which pair is worth conjoining — which
    is how ``"founder cto"`` (9,027 rows, near-perfect precision) gets found and ``"head
    of growth"`` never gets fired. Stopwords go with them, so ``of`` never becomes a
    search term, and the model's own phrasing survives being sloppy.

    The same split applied to the other two axes was silently fatal, because they match a
    whole value: ``"United States"`` seeded ``united`` and ``states``, which count 0
    apiece and died at offset 0 as *nobody matches this*; ``c_suite`` came apart at the
    underscore and seeded ``suite``. A place is re-cased instead (``as_place``) and a
    seniority is passed through — it is already one of the twelve values the provider
    publishes, typed as such on ``ICPSpec``.
    """
    from openoutfind.core.pipeline.vocabulary import tokenize
    from openoutfind.discovery import as_place

    keywords = set()
    for field, attr in _SEED_FIELDS:
        value = getattr(spec, attr)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not item:
                continue
            if field == "lead_job_title":
                keywords |= {(field, token) for token in tokenize(str(item))}
            elif field == "lead_location":
                keywords.add((field, as_place(item)))
            else:
                keywords.add((field, str(item)))
    return sorted(keywords)


class Seed(NamedTuple):
    """The opening vocabulary, and what every node built from it carries into a query.

    Both extras are query attributes rather than configuration: the band rides the
    filters unchanged, and the country is what a lead surfaced by this walk is tagged
    with. They travel to ``select.seed_frontier`` and land on the nodes.
    """

    keywords: list[tuple[str, str]]
    headcount: tuple[int, int]
    country_code: str = ""


def generate_seed(site_config) -> Seed:
    """LLM-generate the opening vocabulary and size band.

    The cold start, and the **only** LLM call discovery makes about queries: with no
    qualified leads there are no profiles to count words from, so the ICP text is the one
    available source. Everything after this is counting (``vocabulary.refresh``).

    The band is **returned, not stored**: it rides every query unchanged and is never
    searched, so it belongs on the nodes this seed opens (``select.seed_frontier``),
    where it is part of the query that was actually fired.

    Returns empty keywords when the ICP is empty.
    """
    from pydantic_ai import Agent

    from openoutfind.core.llm import get_llm_model, run_agent_sync
    from openoutfind.core.models import Keyword
    from openoutfind.discovery import describe_node

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    prompt = env.get_template("icp_filters.j2").render(
        product_docs=site_config.product_docs,
        campaign_target=site_config.campaign_target,
        seniorities=LEAD_SENIORITIES,
    )

    agent = Agent(
        get_llm_model(),
        output_type=ICPSpec,
        model_settings={"temperature": 0.3, "timeout": 60},
    )
    spec = run_agent_sync(agent.run(prompt)).output

    band = (spec.headcount_min, spec.headcount_max)
    country_code = spec.country_code.lower()
    keywords = _seed_keywords(spec)
    if not keywords:
        return Seed([], band, country_code)

    Keyword.rows_for(keywords)

    # The seed is a *query*, not a description of a buyer — it says `founder cto` where
    # the operator asked for "engineering leaders at small SaaS firms". The operator's
    # echo is `log_icp_echo` below; this one stays for the maintainer reading the walk.
    logger.debug("%s: %s · headcount %d–%d",
                 colored("discovery seed", "cyan", attrs=["bold"]),
                 colored(describe_node(keywords), "cyan"),
                 spec.headcount_min, spec.headcount_max)
    return Seed(keywords, band, country_code)


# ── anchors: the ICP as synthetic profiles ───────────────────────────


class Anchor(NamedTuple):
    """One invented ideal lead: the line the GP embeds, and the row the walk counts.

    Two shapes of the same claim, because the two consumers read different things. The
    GP wants ``profile`` — one flat line in ``profile_text_for``'s shape, embedded whole.
    The vocabulary wants ``source_fields`` — the same person as a *lead row*, each value
    already under the field it is searchable in, exactly as ``discovery.source_fields_for``
    stores one for a real lead.
    """

    profile: str
    source_fields: dict


class _AnchorProfile(BaseModel):
    """One invented lead — written once as a line, and again as its queryable fields.

    The fields are asked for rather than parsed out. Splitting the flat line by guess is
    what made anchors unusable as vocabulary: a bag of words cannot say whether
    ``united states`` is a job title or a place, and filing it wrong poisons the axis for
    the install's life. The model already knows which is which — it just was never asked.
    """

    profile: str = Field(
        description="Lowercase one-line lead profile: headline, industry, job title, "
                    "company name, seniority, company industry, state, country — space "
                    "separated, no labels.",
    )
    job_title: str = Field(
        default="",
        description="This lead's job title alone, lowercase, no company and no location "
                    "— e.g. 'head of revenue'.",
    )
    location_state: str = Field(
        default="",
        description="State, province or region alone, or empty if the country has none "
                    "worth naming — e.g. 'california'.",
    )
    location_country: str = Field(
        default="",
        description="Country alone — e.g. 'united states'.",
    )


class _AnchorProfiles(BaseModel):
    """The LLM's invented ideal leads, each one line in ``profile_text_for``'s shape."""

    profiles: list[_AnchorProfile] = Field(default_factory=list)


def generate_anchors(site_config, count: int = ANCHOR_COUNT, existing=()) -> list[Anchor]:
    """LLM-invent ``count`` ideal-lead profiles. ``[]`` on an outage or empty ICP.

    ``existing`` are the profiles already written — shown to the model so a top-up round
    widens the positive region instead of restating it.

    Best-effort by design: an unanchored install still runs, it just spends its cold
    phase without a fitted GP, so failure must not propagate to the caller.
    """
    from pydantic_ai import Agent

    from openoutfind.core.llm import get_llm_model, run_agent_sync

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    prompt = env.get_template("anchor_profiles.j2").render(
        product_docs=site_config.product_docs,
        campaign_target=site_config.campaign_target,
        count=count,
        existing=list(existing),
    )

    try:
        agent = Agent(
            get_llm_model(),
            output_type=_AnchorProfiles,
            # Warmer than the seed: the seed wants the single most likely conjunction,
            # these want spread across the ideal region.
            model_settings={"temperature": 0.8, "timeout": 60},
        )
        result = run_agent_sync(agent.run(prompt)).output
    except Exception:
        logger.exception("anchor generation failed — install stays unanchored")
        return []

    return [anchor for anchor in map(_as_anchor, result.profiles) if anchor.profile]


def _as_anchor(written: _AnchorProfile) -> Anchor:
    """One LLM-written profile as the pair the two consumers need.

    ``source_fields_for`` is reused rather than reimplemented so an anchor row and a
    discovered lead row are built by the same function: it keeps only the keys
    ``KEYWORD_SOURCE_FIELDS`` reads and drops the empty ones, which is what makes an
    anchor with no state contribute its country alone instead of a blank place.
    """
    from openoutfind.discovery import source_fields_for

    return Anchor(
        profile=written.profile.strip().lower(),
        source_fields=source_fields_for({
            "contact_job_title": written.job_title.strip().lower(),
            "contact_location_state": written.location_state.strip().lower(),
            "contact_location_country": written.location_country.strip().lower(),
        }),
    )


def anchor_leads():
    """The invented ideal leads, oldest first — the anchors as the rows they are.

    One query, and the only definition of *anchor* the codebase has: a ``Lead`` this
    install wrote rather than discovered. The GP reads their embeddings, the label store
    their ``profile_text`` and the vocabulary their ``source_fields``, each with the same
    accessor it uses for a real lead.
    """
    from openoutfind.crm.models import Lead

    return list(Lead.objects.filter(synthetic=True).order_by("pk"))


def stored_anchors() -> np.ndarray | None:
    """The anchors' embeddings as ``(N, dim)``, or ``None`` when there are none."""
    embeddings = [lead.embedding_array for lead in anchor_leads()]
    present = [e for e in embeddings if e is not None]
    if not present:
        return None
    return np.array(present, dtype=np.float32)


def ensure_anchors(site_config) -> np.ndarray | None:
    """The anchor embeddings as ``(N, dim)``, filled up to ``ANCHOR_COUNT``.

    Generates on first use and fills the remainder on a later call if an earlier one came
    back short. Already-written profiles are shown to the model so the second round widens
    the ideal region rather than restating it, and the set is written as rows — a restart
    must not re-invent anchors (and re-anchor the GP somewhere slightly different).

    ``None`` when there is no ICP text to work from, or the LLM call failed and nothing is
    stored — callers treat that as "no anchors", never as an error. A failed fill-up keeps
    whatever is already there.

    Never called once a real lead has qualified: from that point the set is permanent,
    and callers restore it with ``stored_anchors`` instead of inventing more.
    """
    from openoutfind.crm.models import Lead
    from openoutfind.discovery import embed_profile

    written = anchor_leads()
    profiles = [lead.profile_text for lead in written]
    if len(written) >= ANCHOR_COUNT:
        return stored_anchors()

    if not (site_config.product_docs or site_config.campaign_target):
        return stored_anchors()

    fresh = [
        anchor for anchor in generate_anchors(site_config, count=ANCHOR_COUNT - len(written),
                                              existing=profiles)
        if anchor.profile not in profiles
    ]
    if not fresh:
        return stored_anchors()

    for anchor in fresh:
        lead = Lead(
            synthetic=True,
            profile_text=anchor.profile,
            # The model's own assignment of value to search field, never split back out
            # of the flat line — that guess is what the field exists to prevent.
            source_fields=anchor.source_fields,
            country_code=site_config.operator_country_code,
        )
        # Embedded **without** query terms, unlike a discovered lead: an anchor is a claim
        # about what a good lead looks like, not about which query to run.
        lead.embedding_array = embed_profile(anchor.profile)
        lead.save()

    logger.debug("%s: +%d synthetic ideal profile(s) (%d total)",
                 colored("anchors", "cyan", attrs=["bold"]), len(fresh),
                 len(written) + len(fresh))
    log_icp_echo()
    return stored_anchors()


def log_icp_echo() -> None:
    """Tell the operator who the system thinks this install sells to. No-op unanchored.

    **This is the earliest proof the product description was understood**, and therefore
    the earliest chance to correct it — the loop the README sells. The material costs
    nothing to print: the anchors are already computed, already one line each in
    ``profile_text``'s shape, and until now only their *count* was ever shown.

    Printed on the pass that writes them and again at the start of every later run, so
    the operator meets it before the first search rather than only on a cold start.
    """
    profiles = [lead.profile_text for lead in anchor_leads() if lead.profile_text]
    if not profiles:
        return

    logger.info("%s", colored("Looking for people like:", "cyan", attrs=["bold"]))
    for profile in profiles:
        logger.info("    · %s", profile)
