# openoutfind/core/pipeline/icp.py
"""ICP generators — the LLM writes the install's cold-start priors, in two shapes.

The same two inputs (``product_docs + campaign_target``) are the only prior available
before any lead has been judged, and the engine needs them expressed two ways:

- ``write_initial_population`` — the ICP as **searches**: the walk's whole first round,
  written by the model as whole searches rather than drawn at random. "A head
  of procurement at a cannabis extraction company in California" is ``{title: head,
  procurement; tag: cannabis extraction; place: California, United States}``: the model
  knows which words go together, a title loses nothing by being a set of words (order and
  stopwords are ignored), and a tag phrase is kept whole, as the provider matches it.
  The answer also carries the **shape** every search rides — the size band and the market.
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
firmographics, so it would anchor the model in a region no candidate occupies.

Under ``--agent-qualify`` the calling agent writes both, in one answer
(``PopulationAnswer``, handed in with ``find --population``).
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jinja2
import numpy as np
from pydantic import BaseModel, Field
from termcolor import colored

from openoutfind.core.conf import PROMPTS_DIR, WALK_CONFIG
from openoutfind.discovery import LEAD_SENIORITIES, Seniority

logger = logging.getLogger(__name__)

# How many synthetic ideal profiles anchor a cold start. Several rather than one so the
# positive region is outlined rather than pinned to a single hallucination, but few
# enough that a handful of real labels outweighs them.
ANCHOR_COUNT = 3

POPULATION_SIZE = WALK_CONFIG["population"]


# ── the initial population: the ICP as searches ──────────────────────


class WrittenSearch(BaseModel):
    """One whole search, as the model writes it — every part optional but the title."""

    title: str = Field(
        description="The words of the buyer's job title, as LinkedIn titles carry them — "
                    "'head procurement', 'founder', 'production manager'. Every word must "
                    "appear in the title; nothing is stemmed.",
    )
    tag: str = Field(
        default="",
        description="At most one phrase the company's own LinkedIn tags hold — "
                    "'cannabis', 'thc vape', 'cannabis extraction' — or empty for any "
                    "company. Its words must sit together, in order, inside one tag.",
    )
    state: str = Field(
        default="",
        description="A state or region inside the market country — 'California' — or "
                    "empty for the whole country.",
    )
    seniority: Seniority | None = None


class InitialPopulation(BaseModel):
    """The walk's first round, and the shape every search in the walk rides.

    ``seniority`` is typed to Lead Finder's vocabulary, not ``str``: an unknown level
    returns an empty page rather than an error, wasting a search.
    """

    searches: list[WrittenSearch] = Field(min_length=POPULATION_SIZE,
                                          max_length=POPULATION_SIZE)
    country: str = Field(description="The market country by its real name — "
                                     "'United States'. Never a region.")
    country_code: str = Field(default="", description="The market's ISO-3166-1 alpha-2 "
                                                      "code, lowercase — 'us'.")
    headcount_min: int = 1
    headcount_max: int = 10000


def as_terms(written: WrittenSearch, country: str):
    """One written search as ``query.Terms``, its state placed inside the market."""
    from openoutfind.core.pipeline.query import Terms, place_of

    return Terms.written(title=written.title, tag=written.tag,
                         place=place_of(written.state, country),
                         seniority=written.seniority or "")


def write_initial_population(site_config) -> int:
    """Write the initial population as ``Search`` rows. Returns how many were new.

    The cold start, and the **only** LLM call the walk makes: every later search is
    composed from the store's keywords (``compose``), with no model in the loop. Under ``--agent-qualify`` the
    population is the calling agent's ``--population`` answer instead, or the run stops
    with ``population_pending``.

    A search that is not legal (a tag with no title words left once stopwords go) is
    dropped; two that analyse to the same terms are one row.
    """
    from openoutfind.core.models import Search

    answer = agent_population_answer(site_config)
    population = answer if answer is not None else _population_from_llm(site_config)

    created = 0
    for written in population.searches:
        terms = as_terms(written, population.country)
        if not terms.is_legal:
            logger.debug("written search dropped — not legal: %s", written)
            continue
        _, new = Search.objects.get_or_create(key=terms.key, defaults={
            "terms": terms.as_json(),
            "origin": Search.Origin.WRITTEN,
            "headcount_min": population.headcount_min,
            "headcount_max": population.headcount_max,
            "country_code": population.country_code.lower(),
        })
        created += new
    logger.debug("%s: %d search(es) · %s · headcount %d–%d",
                 colored("initial population", "cyan", attrs=["bold"]), created,
                 population.country, population.headcount_min, population.headcount_max)
    return created


def _population_from_llm(site_config) -> InitialPopulation:
    """Ask ``AI_MODEL`` for the initial population."""
    from pydantic_ai import Agent

    from openoutfind.core.llm import get_llm_model, run_agent_sync
    from openoutfind.core.pipeline.brief import ensure_rules

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    prompt = env.get_template("initial_population.j2").render(
        product_docs=site_config.product_docs,
        campaign_target=site_config.campaign_target,
        rules=ensure_rules(site_config),
        seniorities=LEAD_SENIORITIES,
        size=POPULATION_SIZE,
    )

    # No temperature: the current Claude models reject one that isn't their default.
    agent = Agent(get_llm_model(), output_type=InitialPopulation,
                  model_settings={"timeout": 120})
    return run_agent_sync(agent.run(prompt)).output


# ── anchors: the ICP as synthetic profiles ───────────────────────────


class Anchor(NamedTuple):
    """One invented ideal lead: the line the GP embeds, and the row the walk counts.

    Two shapes of the same claim, because the two consumers read different things. The
    GP wants ``profile`` — one flat line in ``profile_text_for``'s shape, embedded whole.
    The term pools want ``source_fields`` — the same person as a *lead row*, each value
    already under the field it is searchable in, exactly as ``discovery.source_fields_for``
    stores one for a real lead.
    """

    profile: str
    source_fields: dict


class _AnchorProfile(BaseModel):
    """One invented lead — written once as a line, and again as its queryable fields.

    The fields are asked for rather than parsed out: a bag of words cannot say whether
    ``united states`` is a job title or a place, and the model already knows which is
    which.
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
    company_keywords: str = Field(
        default="",
        description="The tags LinkedIn would give this lead's company, lowercase and "
                    "comma separated — what it does, sells and makes, never the "
                    "company's name — e.g. 'cannabis, thc vape, cannabis extraction'.",
    )


class _AnchorProfiles(BaseModel):
    """The LLM's invented ideal leads, each one line in ``profile_text_for``'s shape."""

    profiles: list[_AnchorProfile] = Field(default_factory=list)


def generate_anchors(site_config, count: int = ANCHOR_COUNT, existing=()) -> list[Anchor]:
    """LLM-invent ``count`` ideal-lead profiles. ``[]`` on an outage or empty ICP.

    ``existing`` are the profiles already written — shown to the model so a top-up round
    widens the positive region instead of restating it.

    Best-effort by design: a failure must not propagate to the caller, and the walk,
    which does not start unanchored, checks what came back (``walk._anchor``). The one
    exception is ``population_pending`` under ``--agent-qualify``, which is not a failure
    but the question the calling agent answers with ``--population``.
    """
    answer = agent_population_answer(site_config, existing=existing)
    if answer is not None:
        return [anchor for anchor in map(_as_anchor, answer.anchors[:count]) if anchor.profile]

    from pydantic_ai import Agent

    from openoutfind.core.llm import get_llm_model, run_agent_sync
    from openoutfind.core.pipeline.brief import ensure_rules

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    prompt = env.get_template("anchor_profiles.j2").render(
        product_docs=site_config.product_docs,
        campaign_target=site_config.campaign_target,
        rules=ensure_rules(site_config),
        count=count,
        existing=list(existing),
    )

    try:
        # No temperature: the current Claude models reject one that isn't their default.
        # The spread across the ideal region is asked for in the prompt instead.
        agent = Agent(get_llm_model(), output_type=_AnchorProfiles,
                      model_settings={"timeout": 60})
        result = run_agent_sync(agent.run(prompt)).output
    except Exception:
        logger.exception("anchor generation failed — install stays unanchored")
        return []

    return [anchor for anchor in map(_as_anchor, result.profiles) if anchor.profile]


# ── the calling agent as the author (`find --agent-qualify --population`) ──


class PopulationAnswer(InitialPopulation):
    """Both cold-start priors in one answer — what ``--population`` carries.

    One answer rather than two questions because they come from the same two texts and
    an agent writes them in the same breath; asking twice would cost the caller a second
    round trip for nothing. The model path writes the same searches (``InitialPopulation``)
    and its anchors in a call of their own.
    """

    anchors: list[_AnchorProfile] = Field(min_length=ANCHOR_COUNT)


def agent_population_answer(site_config, existing=()) -> PopulationAnswer | None:
    """The ``--population`` answer when a calling agent is the author; ``None`` to ask
    the LLM.

    Under ``--agent-qualify`` with no answer given, stops the run with
    ``population_pending`` — the payload is everything the answer is written from, the
    field types it is written in, and its schema.
    """
    from openoutfind.core import agent_qualify
    from openoutfind.core.pipeline.brief import stored
    from openoutfind.core.pipeline.qualify import PopulationPending

    if not agent_qualify.active():
        return None
    answer = agent_qualify.population_answer()
    if answer is not None:
        return answer
    brief = stored()
    raise PopulationPending(
        f"this store has no cold start yet — write the {POPULATION_SIZE} searches of the "
        f"initial population and {ANCHOR_COUNT} ideal profiles",
        payload={
            "product_docs": site_config.product_docs,
            "campaign_target": site_config.campaign_target,
            "rules": brief.rules if brief is not None else "",
            "population_size": POPULATION_SIZE,
            "fields": FIELD_TYPES,
            "seniorities": list(LEAD_SENIORITIES),
            "anchor_count": ANCHOR_COUNT,
            "existing_anchors": list(existing),
            "schema": PopulationAnswer.model_json_schema(),
        },
    )


# How the provider matches each field a written search holds — handed to the calling
# agent with ``population_pending``, so it writes the forms that match.
FIELD_TYPES = {
    "title": "words, every one required in the job title, any order; stopwords ignored; "
             "nothing stemmed (manager ≠ managers)",
    "tag": "one phrase inside one of the company's LinkedIn tags: adjacent, in order, "
           "nothing stemmed (vape ≠ vapes); never without a title",
    "state": "a state or region inside the market country, or empty for all of it",
    "seniority": "one value from the closed list, or null",
}


def _as_anchor(written: _AnchorProfile) -> Anchor:
    """One LLM-written profile as the pair the two consumers need.

    ``source_fields_for`` is reused rather than reimplemented so an anchor row and a
    discovered lead row are built by the same function: it keeps only the keys
    ``SOURCE_FIELDS`` reads and drops the empty ones, which is what makes an anchor with
    no state contribute its country alone instead of a blank place.
    """
    from openoutfind.discovery import source_fields_for

    return Anchor(
        profile=written.profile.strip().lower(),
        source_fields=source_fields_for({
            "contact_job_title": written.job_title.strip().lower(),
            "contact_location_state": written.location_state.strip().lower(),
            "contact_location_country": written.location_country.strip().lower(),
            "company_keywords": written.company_keywords.strip().lower(),
        }),
    )


def anchor_leads():
    """The invented ideal leads, oldest first — the anchors as the rows they are.

    One query, and the only definition of *anchor* the codebase has: a ``Lead`` this
    install wrote rather than discovered. The GP reads their embeddings with the same
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

    Never called on a standing set once a real lead has qualified: from that point the set
    is permanent, and callers restore it with ``stored_anchors`` instead of inventing more.
    The walk calls it on a store with none at all — one seeded with marks never ran cold —
    so every store is anchored.
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
