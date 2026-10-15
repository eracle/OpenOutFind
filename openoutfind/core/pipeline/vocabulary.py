# openoutfind/core/pipeline/vocabulary.py
"""The keyword vocabulary — the tokens a discovery query can be built from.

Growth is **counting, not generation**. The old engine asked an LLM to invent clause
values ("propose new job titles adjacent to these leads"), and the LLM wrote prose:
``Head of Content Strategy``, ``Chief Science Officer``, ``Director of Education``.
Every word in a Lead Finder value is another AND, so a four-token title is near-empty
before it is conjoined with anything — a large, independent cause of the 60-consecutive-
empty-query run that started this rewrite. Here the vocabulary is simply *the words that
appear in profiles the LLM already accepted*, one word per keyword, and narrowing happens
by conjoining tokens at query time against measured feedback.

**Admission is document frequency, and it is nearly free.** Over the label store, a df≥2
floor drops 65% of the vocabulary (3,485 → 1,208 tokens) and loses **zero** good tokens —
none with support ≥ 8 and true precision > 0.8 disappears until df≥10. What it removes is
the singleton tail, which is mostly company names and typos (``cloneable.ai``,
``buildteam.ai.``, ``vendbot``) — and that tail is not harmless: it is 56% of the *top* of
any embedding-based ranking, so it would be fired first. See §13a/§13b of the roadmap card
``p1-e3-leadfinder-index-semantics-and-query-model-rethink``.

**A token's field is measured, not chosen.** ``cto`` is alive in ``lead_job_title`` and
dead everywhere else; ``belgium`` is the reverse; ``cannabis`` in a title finds regulators
and in ``company_keywords`` finds cannabis companies. Reading each axis from the lead-row
fields that *are* that axis (``discovery.KEYWORD_SOURCE_FIELDS``) keeps the per-field
vocabularies nearly disjoint at no cost, and the walk counts each axis where its words
live (``label_tokens``). ``lead_seniority`` is the exception and the easy case: the
provider publishes a closed 12-value list, so it is seeded whole and never grown.

The upgrade path, if the singleton tail ever needs a sharper cut than df≥2, is Dunning's
log-likelihood ratio — it kills df=1 junk *and* the high-frequency-uninformative tokens a
df floor waves through (``services`` reads 25/34 in the store and would score negative).
Not needed yet.
"""
from __future__ import annotations

import logging
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logger = logging.getLogger(__name__)

# A token is a word: a letter, then letters/digits and the punctuation real job titles
# carry (``ai-native``, ``ai/ml``, ``co-founder``, ``r&d``). Single characters are
# dropped — they are never a search term and they match everything.
_TOKEN = re.compile(r"[a-z][a-z0-9&/+.\-]{1,}")

# Rank abbreviations. ``Sr. Manager, Extraction`` grew ``sr.`` as a title word, and the
# rank is the seniority axis's job; as a title word it only narrows to whoever spelled it.
_STOPWORDS = ENGLISH_STOP_WORDS | {"sr", "jr", "snr"}

# Document-frequency floor over the qualified profiles. Two, not one: the singleton tail
# is 65% of the vocabulary and costs nothing to drop (§13a).
MIN_DOCUMENT_FREQUENCY = 2


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens of a text, stopwords stripped.

    A trailing ``.`` is punctuation, not part of the word (``sr.``, ``inc.``), so it goes
    before the stopword check. ``sklearn``'s stopword list rather than an NLP dependency:
    sklearn is already required for the qualifier, while ``nltk`` and ``spacy`` are not
    installed and would each pull a model download into a first run for no measured gain.
    """
    words = (t.rstrip(".") for t in _TOKEN.findall(text.lower()))
    return {t for t in words if len(t) > 1 and t not in _STOPWORDS}


# Separators inside a company tag. Tags are taxonomy paths and slugs as often as words
# (``food_and_drink/restaurants_and_delivery``, ``cloud-services``), and a title's
# ``ai/ml`` or ``co-founder`` has no counterpart there, so they split.
_TAG_SEPARATORS = re.compile(r"[/_\-]")

# Company words the vocabulary never *learns*. Every cannabis operator's tag list carries
# ``health``, ``medical`` and ``retail`` beside ``cannabis``, so two fits at two operators
# admitted them, and the walk fired company "health" + a title. They are still searchable
# when the ICP names one — a seed word is the operator's intent, a learned one is only
# co-occurrence. Measured on the 2026-10-15 vape order (GTI, Verano).
_GENERIC_COMPANY_WORDS = frozenset({
    "health", "healthcare", "care", "medical", "medicine", "alternative", "retail",
    "finance", "financial", "investing", "services", "service", "business", "technology",
    "consulting", "manufacturing", "products", "industry", "industrial", "portfolio",
})


def company_words(text: str) -> set[str]:
    """The single words of a company's tags (or of an ICP's domain words)."""
    return tokenize(_TAG_SEPARATORS.sub(" ", text))


def learnable_company_words(text: str) -> set[str]:
    """The words of a company's tags the vocabulary may grow from.

    A tag holding a ``/`` is the provider's category path (``finance/investing``,
    ``law_and_government/government``), filed on every company in a sector rather than
    written by the company, so it is skipped whole; so are ``_GENERIC_COMPANY_WORDS``.
    """
    own_tags = [tag for tag in text.split(",") if "/" not in tag]
    return company_words(",".join(own_tags)) - _GENERIC_COMPANY_WORDS


def keywords_for(fields: dict) -> set[tuple[str, str]]:
    """One lead row's stored per-field text → the ``(field, token)`` keywords it grows.

    **Each axis has its own unit of vocabulary, and the difference is the provider's,
    not a preference.** A job title is free text whose words are matched individually,
    so it contributes *words* and the walk conjoins them. A company's tags contribute
    words too, one per node (two read as a phrase). A location is matched as one whole
    place, so it contributes exactly one value.

    Splitting a location on spaces is what this replaces, and it was silently fatal:
    ``United states`` became ``united`` and ``states``, ``New york`` became ``new`` and
    ``york``, and every fragment counts **0** at the provider. The walk spent a
    round-trip apiece finding that out, then wrote each one off at offset 0 as *nobody
    matches this* — pruning its whole subtree over what was really a fact about our
    syntax. A live campaign put 19 such fragments in its vocabulary and killed Israel and
    Spain outright.

    A US state is only a place *with* its country (``California`` counts 0,
    ``California, United States`` counts 2,714,366), so a row carrying a state yields the
    pair and a row with only a country yields the country.
    """
    from openoutfind.discovery import KEYWORD_SOURCE_FIELDS, as_place

    (title_key,) = KEYWORD_SOURCE_FIELDS["lead_job_title"]
    (company_key,) = KEYWORD_SOURCE_FIELDS["company_keywords"]
    state_key, country_key = KEYWORD_SOURCE_FIELDS["lead_location"]

    keywords = {("lead_job_title", token)
                for token in tokenize(str(fields.get(title_key) or ""))}
    keywords |= {("company_keywords", token)
                 for token in learnable_company_words(str(fields.get(company_key) or ""))}
    country, state = as_place(fields.get(country_key)), as_place(fields.get(state_key))
    if country:
        keywords.add(("lead_location", f"{state}, {country}" if state else country))
    return keywords


def _counted_apart() -> dict[str, tuple[str, ...]]:
    """Axes whose source is not part of ``profile_text`` → the source fields to count.

    Derived, not listed: an axis is counted in ``profile_text`` when every one of its
    source fields is in ``discovery.TEXT_FIELDS``, and against its own stored text when
    any is not. Today that is ``company_keywords`` alone.
    """
    from openoutfind.discovery import KEYWORD_SOURCE_FIELDS, TEXT_FIELDS

    return {axis: keys for axis, keys in KEYWORD_SOURCE_FIELDS.items()
            if not set(keys) <= set(TEXT_FIELDS)}


def _scoped(field: str, word: str) -> str:
    """A word as it is stored for an axis counted apart — scoped so it can't collide."""
    return f"{field}:{word}"


def pair_words(field: str, token: str) -> frozenset[str]:
    """The words a ``(field, token)`` keyword is matched by in a label token set.

    A word on an axis counted in ``profile_text`` is the bare word (``keyword_words``).
    A word on an axis counted apart is scoped to that axis, so ``cannabis`` the company
    tag is only ever found among a lead's company tags — never in a title or headline
    that happens to say it, which is the coincidence the company axis exists to escape.
    """
    words = keyword_words(token)
    if field in _counted_apart():
        return frozenset(_scoped(field, word) for word in words)
    return words


def label_tokens(profile_text: str, source_fields: dict | None) -> frozenset[str]:
    """One labelled lead as the token set the walk counts nodes against.

    ``profile_text``'s words (``profile_tokens``), plus the scoped words of every axis
    counted apart, read from the lead's stored ``source_fields``. A lead stored before
    an axis had a source contributes nothing on it, which is the truth: we do not know.
    """
    tokens = set(profile_tokens(profile_text))
    fields = source_fields or {}
    for axis, keys in _counted_apart().items():
        for key in keys:
            text = str(fields.get(key) or "")
            words = company_words(text) if axis == "company_keywords" else tokenize(text)
            tokens |= {_scoped(axis, word) for word in words}
    return frozenset(tokens)


def keyword_words(token: str) -> frozenset[str]:
    """A keyword's words, as a profile's own tokens would spell them.

    A keyword is not always one word any more — ``California, United States`` is a single
    ``lead_location`` value — while a profile is still counted as a bag of words. So a
    node's tokens are matched by *word containment*: every word of every keyword has to
    appear in the profile. For a one-word job title that is the old behaviour exactly.

    Stopwords are stripped for the same reason they are stripped from profiles, and by
    the same tokenizer, so ``District of Columbia, United States`` still matches a profile
    whose ``of`` was dropped. A keyword that is *nothing but* stopwords keeps its raw
    words instead: an empty word set is a subset of every profile, which would make such a
    keyword match everybody rather than nobody.
    """
    words = tokenize(token)
    return frozenset(words or token.lower().split())


def profile_tokens(profile_text: str) -> frozenset[str]:
    """The token set of one stored profile's ``profile_text``.

    **Field-agnostic on purpose** for the axes it carries (title, place, seniority): a
    node's ``a``/``b`` are counted by asking whether a labelled profile contains its
    tokens *anywhere* in the text, not in the field the query will put them in. That is
    how the estimator was measured (§13c), and it lets the whole existing label store
    count for nodes built from fields those older rows never recorded. Company tags are
    not in the text; ``label_tokens`` adds them.
    """
    return frozenset(tokenize(profile_text))


# ── growing the vocabulary ───────────────────────────────────────────

# Count of accepted leads as of the last refresh, so a pass that changed nothing costs
# one `COUNT` instead of a tokenize over the whole accepted set. Process-local by design,
# exactly like `cycle._scored_at`: one process per job, and a second job simply counts
# once more than it had to.
_refreshed_at: tuple[int, int] | None = None


def _accepted_count() -> int:
    """How many leads have been accepted — what the vocabulary is derived from.

    The whole of the refresh's input state: the admission rule is a document frequency
    over accepted profiles, so nothing but their number can change the answer.
    """
    from openoutfind.crm.models import Deal, DealState, Outcome

    return (
        Deal.objects.filter(lead_id__isnull=False)
        .exclude(state=DealState.FAILED, outcome=Outcome.WRONG_FIT)
        .count()
    )


def _anchor_source_fields() -> list[dict]:
    """The synthetic ideal leads as rows the vocabulary can count.

    Empty for an install anchored before the fields were asked for — its flat profiles
    stay GP observations only, because recovering the fields from the line is exactly the
    guess this avoids.
    """
    from openoutfind.crm.models import Lead

    return [
        fields for fields in
        Lead.objects.filter(synthetic=True).values_list("source_fields", flat=True)
        if fields
    ]


def _qualified_source_fields() -> list[dict]:
    """Per-field raw text of the leads the LLM accepted — the vocabulary's only source.

    Qualified mirrors the GP's own labelling: any deal that is not an LLM rejection
    (``FAILED`` + ``wrong_fit``). Leads discovered before ``Lead.source_fields`` existed
    carry ``{}`` and contribute nothing here — they still count toward every node's
    ``a``/``b`` (that reads ``profile_text``), they just cannot say which *field* one of
    their words belongs in, and guessing would put ``united states`` into job titles.
    """
    from openoutfind.crm.models import Deal, DealState, Lead, Outcome

    lead_ids = (
        Deal.objects.filter(lead_id__isnull=False)
        .exclude(state=DealState.FAILED, outcome=Outcome.WRONG_FIT)
        .values_list("lead_id", flat=True)
    )
    return [
        fields for fields in
        Lead.objects.filter(pk__in=lead_ids).values_list("source_fields", flat=True)
        if fields
    ]


def _admit(profiles: list[dict]) -> tuple[list[tuple[str, str]], dict]:
    """The ``(field, token)`` pairs clearing the df floor, and every pair's frequency.

    Document frequency is how many qualified profiles carry a pair — except on the company
    axis, where the document is the **company**: two fits from Rove share one tag list,
    and counting it twice would admit ``rove``, ``melted`` and ``diamond`` on the strength
    of one employer. A company is told apart by its tag text, the field being counted.
    """
    from openoutfind.discovery import KEYWORD_SOURCE_FIELDS

    (company_key,) = KEYWORD_SOURCE_FIELDS["company_keywords"]
    documents: dict[tuple[str, str], set] = {}
    for index, fields in enumerate(profiles):
        for pair in keywords_for(fields):
            document = fields.get(company_key) if pair[0] == "company_keywords" else index
            documents.setdefault(pair, set()).add(document)

    frequency = {pair: len(seen) for pair, seen in documents.items()}
    admitted = [pair for pair, df in frequency.items() if df >= MIN_DOCUMENT_FREQUENCY]
    return admitted, frequency


def refresh() -> int:
    """Fold the qualified leads' words into the vocabulary. Returns tokens added.

    Cheap enough to run on every pass — it is a tokenize and a count over a few hundred
    profiles, with no LLM call and no provider call, which is the whole point of replacing
    minting with counting.

    **Its trigger is an acceptance, not a query.** It used to be called from exactly one
    place, ``discover._ensure_frontier``, so the vocabulary only grew when the walk
    happened to widen. That was invisible while the exploit state fell through to
    discovery on every pass — the two always happened together — and becomes a stale
    frontier the moment a run spends its time labelling instead: acceptances accumulate,
    none of them reach the vocabulary, and the walk keeps firing queries built on evidence
    it has already superseded. Callers may therefore call this whenever they like; the
    signature below makes a redundant call one indexed ``COUNT``.
    """
    global _refreshed_at
    from openoutfind.core.models import Keyword

    anchors = _anchor_source_fields()
    signature = (_accepted_count(), len(anchors))
    if _refreshed_at == signature:
        return 0
    _refreshed_at = signature

    # **Anchors count as documents, not as a special case.** They are invented ideal leads
    # written in a lead row's own shape, so the df floor applies to them exactly as it does
    # to real acceptances, and their influence dilutes on its own as real profiles arrive —
    # 3 anchors among 122 acceptances decide nothing. Without them an install with no
    # acceptances has no vocabulary at all beyond the seed, which is how one live install
    # came to fire 63 queries off a corpus of 3 profiles: at df>=2 a word had to appear in
    # two of three, so what survived was whatever generic token they happened to share.
    profiles = anchors + _qualified_source_fields()
    if not profiles:
        # The cold-phase reality, and worth naming: with nothing qualified yet there are
        # no words to count, so the vocabulary is whatever the ICP seed put there and the
        # frontier cannot grow past its depth-1 nodes until the first lead is accepted.
        logger.debug("vocabulary: no qualified lead carries per-field text yet — "
                     "nothing to grow from (seed vocabulary stands)")
        return 0

    admitted, frequency = _admit(profiles)
    logger.debug("vocabulary: %d qualified profile(s) → %d distinct token(s) → "
                 "%d admitted at df>=%d", len(profiles), len(frequency),
                 len(admitted), MIN_DOCUMENT_FREQUENCY)
    if not admitted:
        return 0

    known = set(Keyword.objects.values_list("field", "token"))
    fresh = [pair for pair in admitted if pair not in known]
    if fresh:
        Keyword.rows_for(fresh)
        logger.info("vocabulary: +%d keyword(s) from %d qualified profile(s)",
                    len(fresh), len(profiles))
    return len(fresh)


def seed_seniorities() -> int:
    """Ensure the closed ``lead_seniority`` vocabulary exists. Returns rows added.

    The one axis whose vocabulary is published rather than discovered, so it is seeded
    whole instead of grown: a value outside the list is not an error but an empty page,
    and there are only twelve. ``c_suite`` is included despite appearing zero times in
    13,827 stored profiles — the store says nothing about what the *input* side accepts,
    and one empty page settles it far more cheaply than a special case would.
    """
    from openoutfind.core.models import Keyword
    from openoutfind.discovery import LEAD_SENIORITIES

    pairs = [("lead_seniority", s) for s in LEAD_SENIORITIES]
    known = set(Keyword.objects.filter(field="lead_seniority").values_list("field", "token"))
    fresh = [p for p in pairs if p not in known]
    if fresh:
        Keyword.rows_for(fresh)
    return len(fresh)


def admitted_keywords() -> list[tuple[str, str]]:
    """Every ``(field, token)`` a query node may be built from, sorted."""
    from openoutfind.core.models import Keyword

    return sorted(Keyword.objects.values_list("field", "token"))
