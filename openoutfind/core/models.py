# openoutfind/core/models.py
from __future__ import annotations

from django.db import models

from openoutfind.discovery import SEARCH_FIELDS, describe_node


class Keyword(models.Model):
    """One ``(field, token)`` pair — the unit a discovery query is built from.

    A **single word**, not a phrase. Multi-word values were the old model's silent
    killer: every extra word is another AND (``Manager`` → ``Content Manager`` is a
    ~300× narrowing), so the LLM-written four-token titles the pool used to hold —
    ``Head of Content Strategy``, ``Chief Science Officer`` — were near-empty before
    they were conjoined with anything. Joining tokens is still how the walk narrows,
    but it happens at query time (``discovery.filters_for``), one token per move,
    against measured feedback rather than an LLM's guess at a job title.

    **Globally unique on ``(field, token)``, with no campaign of its own** — a token
    is not campaign-specific, ``lead_location = belgium`` is the same search term
    whoever runs it. A campaign reaches its vocabulary through its query nodes.

    ``field`` is constrained to ``discovery.SEARCH_FIELDS``: the field names are the
    provider contract, and an unknown one is silently *dropped* (you get the
    unfiltered page, with rows, reading as success). ``token`` is deliberately
    **not** constrained — except for ``lead_seniority`` these are free-text search
    terms, and a token the index doesn't carry simply returns an empty page.
    """

    field = models.CharField(
        max_length=32, choices=[(f, f) for f in SEARCH_FIELDS],
    )
    token = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["field", "token"], name="uniq_keyword"),
        ]

    def __str__(self):
        return f"{self.field.removeprefix('lead_')} {self.token}"

    @property
    def pair(self) -> tuple[str, str]:
        return (self.field, self.token)

    @classmethod
    def rows_for(cls, keywords) -> list["Keyword"]:
        """Get-or-create the rows for ``(field, token)`` pairs, in order. Idempotent."""
        return [
            cls.objects.get_or_create(field=field, token=token)[0]
            for field, token in keywords
        ]


class QueryNode(models.Model):
    """One node in a campaign's discovery walk — a keyword set, and where it has been paged to.

    The walk is a **greedy add-only descent over keyword sets**, valued by counting and
    sampled by Thompson. A node is a set of ``(field, token)`` keywords; its children
    are itself plus one more token. There is no remove move — the frontier is global,
    so a shallow node's untried siblings stay reachable without one.

    **This model replaces a whole lattice.** The old walk stored clauses, fetched
    ``DiscoveryQuery`` rows, and an ``EmptyClauseSet`` blacklist, and computed a
    Cartesian product of *maximal* conjunctions on every call — 63,000 candidates for a
    live pool, essentially all of them genuinely empty, because the clause model assumed
    orthogonal facets over what is really a keyword index. See the roadmap card
    ``p1-e3-leadfinder-index-semantics-and-query-model-rethink``.

    **The node carries no value column.** Its estimate is derived from the label store
    every time it is needed — ``a``/``b`` are the qualified/rejected leads whose profile
    text contains all of this node's tokens — so there is no counter to drift, nothing to
    migrate, and nothing to reconcile after a crash. It is also the *same* estimator
    before and after the node is fired, which is what makes a bad page self-correcting:
    a node that looks good from the store and returns nobody useful has its own misses
    land in the counters that made it look good.

    **``parent`` is the level, not provenance.** A child inherits its parent's measured
    rate as the prior its own counts move off (``select.estimate``), because that
    predicts a child's true precision better than the child's raw counts alone (0.661 vs
    0.653, §13c). A node reachable by several paths — and with add-only over three
    fields most are — is created once on its canonical key and keeps the parent giving
    the **highest** estimate.

    **State is a corpus fact, never a model fact.** A node is retired only for
    emptiness: nothing is ever retired for scoring badly, because the qualifier refits
    constantly and a barren yield is a verdict about a view. Which *kind* of emptiness
    depends on the offset it appeared at, because the provider reports ``0`` both for a
    query that matches nobody and for one paged past its end (§7).
    """

    class State(models.TextChoices):
        # Never fetched. Its estimate is its parent's, moved by its own store counts.
        FRONTIER = "frontier", "frontier"
        # Fetched at least once and still has pages left.
        FIRED = "fired", "fired"
        # Paged until a page came back empty. Retired; children pruned when this
        # happened below the 10k reach cap, since every superset's population is
        # then already in our DB.
        DRAINED = "drained", "drained"
        # Returned nothing at offset 0 — the index matches nobody. Retired, and its
        # whole subtree with it: a superset of these tokens matches a subset of people.
        DEAD = "dead", "dead"

    keywords = models.ManyToManyField(Keyword, related_name="nodes")
    # sha256 of the canonicalized keyword set — the node-identity key, a column
    # because the set lives across an M2M and no unique constraint can span one.
    token_key = models.CharField(max_length=64)
    parent = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="children",
    )
    # Where the next page starts. Advances by the page size on every fetch; the
    # provider stops answering past 10,000 whatever the query counts.
    next_offset = models.IntegerField(default=0)
    state = models.CharField(max_length=16, choices=State.choices, default=State.FRONTIER)
    # The provider's exact corpus count at offset 0, when we have asked for it. Free,
    # in the same call, and read only as a diagnostic — the walk fires nodes rather
    # than counting them first, since a dead node generates no children either way.
    leads_found = models.IntegerField(null=True, blank=True)

    # The ICP's company-size band, as this node queries it. A fixed constraint riding the
    # query unchanged, never a search axis: loosening a size bound queries off-ICP rather
    # than widening usefully, and the provider fills a half-open band with any-size
    # companies rather than returning nothing.
    #
    # Columns here rather than one band on a config singleton because the band is *part of
    # the query this node fired*. A seed writes it (``icp.generate_seed``), a child
    # inherits its parent's, and a later re-seed opens its own nodes with its own band
    # instead of retroactively rewriting what an already-fired node meant.
    headcount_min = models.IntegerField(default=1)
    headcount_max = models.IntegerField(default=10000)

    # ISO-3166 alpha-2 of the country this node searches, stamped onto every lead it
    # surfaces — Lead Finder rows carry no ISO code, so the query is what we know. Blank
    # is unknown, which the contacts-store geo-gate treats conservatively.
    #
    # **Not the operator's own country.** That is jurisdiction, it is answered rather
    # than inferred, and it lives in the environment; an operator in Berlin searching
    # Texas is one campaign with two different countries in it, and conflating them once
    # let an LLM's guess at a target market decide whether an EEA operator contributes.
    country_code = models.CharField(max_length=2, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Query Node"
        verbose_name_plural = "Query Nodes"
        constraints = [
            models.UniqueConstraint(
                fields=["token_key"], name="uniq_query_node",
            ),
        ]
        indexes = [
            models.Index(fields=["state"], name="query_node_state_idx"),
        ]

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """This node's keywords as sorted ``(field, token)`` pairs."""
        return sorted(self.keywords.values_list("field", "token"))

    def to_filters(self) -> dict:
        """This node as a Lead Finder filter dict — the only thing the provider sees."""
        from openoutfind.discovery import filters_for

        return filters_for(self.pairs, (self.headcount_min, self.headcount_max))

    def __str__(self):
        """The query itself, not its row id — a node *is* its keyword set."""
        suffix = "" if self.state == self.State.FRONTIER else f" [{self.state}]"
        offset = f" @{self.next_offset}" if self.next_offset else ""
        return f"{describe_node(self.pairs)}{offset}{suffix}"


class CalibrationDraw(models.Model):
    """One lead the calibrate arm drew at random from a score band, to be judged.

    Its verdict is a **calibration verdict** once the lead has a deal: a verdict on a
    lead the model did not choose, which is what lets the bar be measured rather than
    picked (``core/ml/calibration.py``). The row keeps what the draw saw — the band, the
    lead's score and the band's size in the pool at that moment — because the weight a
    verdict carries is the share of the pool it stands for, and the pool moves on.
    """

    lead = models.OneToOneField("outfind_crm.Lead", on_delete=models.CASCADE)
    band = models.PositiveSmallIntegerField()
    p_at_draw = models.FloatField()
    band_size = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"draw band {self.band} · P={self.p_at_draw:.2f} of {self.band_size}"


class CalibrationBar(models.Model):
    """One measurement of the bar, from the calibration verdicts there were at the time.

    The latest row is the bar ``calibration.current_threshold`` returns. ``break_even`` is
    the fit rate below which a verdict was worth less than searching; ``threshold`` is the
    lowest score whose measured fit rate is above it, NULL when no score's is — a result,
    kept so the same verdicts are not measured again, and read as "use the fallback".
    """

    threshold = models.FloatField(null=True)
    break_even = models.FloatField()
    n_verdicts = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        if self.threshold is None:
            return f"no bar above {self.break_even:.1%} from {self.n_verdicts} verdicts"
        return (f"bar {self.threshold:.2f} (break-even {self.break_even:.1%}) "
                f"from {self.n_verdicts} verdicts")


class RunningMean(models.Model):
    """One measured quantity, kept as its sum and count: the mean is ``total / count``.

    What a search and a verdict cost, and how often the pool's verdicts fit, are measured
    on the store's own run rather than taken from settings (``core/meter.py``). A row is a
    plain running average — each event adds its value and one to the count — so the next
    mean follows from the last without keeping the events. It lives in the store because
    an ``--agent-qualify`` run is many processes, and each one must read what the last
    one measured.
    """

    name = models.CharField(max_length=40, unique=True)
    total = models.FloatField(default=0.0)
    count = models.PositiveIntegerField(default=0)

    def __str__(self):
        if not self.count:
            return f"{self.name}: no measurement"
        return f"{self.name}: {self.total / self.count:.3f} over {self.count}"


class Brief(models.Model):
    """Our reading of the customer's two texts: the rules every prompt is written from.

    The customer's words stay in the environment, untouched — they are answers a person
    gave. This is what we concluded from them (who fits, who does not, which companies
    look like buyers and are not), so it is produced, and lives in the store beside the
    rest of what the pipeline produced. One row; ``core/pipeline/brief.py`` owns it.

    ``written_by`` is the model id that wrote it, or ``--brief`` when the caller handed
    it in.
    """

    rules = models.TextField()
    written_by = models.CharField(max_length=100)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"brief by {self.written_by}, {self.updated_at:%Y-%m-%d %H:%M}"
