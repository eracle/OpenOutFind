# openoutfind/core/models.py
from __future__ import annotations

from django.db import models

# How many people one page holds: the provider's largest ``limit`` (250 is refused).
PAGE_SIZE = 200
# Elasticsearch's ``max_result_window``: no query reaches past 10,000 rows, so page 50
# is the last a search has however many millions it counts.
LAST_PAGE = 10_000 // PAGE_SIZE


class Search(models.Model):
    """One search the walk has written or reached — its terms and the shape it rides.

    ``terms`` is ``query.Terms`` as JSON: title words, one tag phrase, one place, one
    seniority. ``key`` is the terms' identity, so a search two moves reach is one row.
    The size band and the country code are the **shape** the customer fixed;
    every search carries the same, copied from the initial population, because they are
    part of the query that was fired rather than a setting.

    **Dead is a corpus fact**: page 1 came back empty with a count of 0, so nobody in
    the index matches, and neither does any search holding all its terms
    (``Terms.holds_all_of``).
    """

    class Origin(models.TextChoices):
        # Written whole by the LLM (or the calling agent) — the initial population.
        WRITTEN = "written", "written"
        # A neighbour of a fired search on the lattice, fired from the frontier.
        NEIGHBOUR = "neighbour", "neighbour"

    terms = models.JSONField()
    key = models.CharField(max_length=64, unique=True)
    origin = models.CharField(max_length=9, choices=Origin.choices,
                              default=Origin.NEIGHBOUR)
    # The move that reached it (``lattice.MOVES``), blank for a written search.
    move = models.CharField(max_length=16, blank=True, default="")
    # The provider's exact count, read from page 1 — the only offset whose count is an
    # answer. ``None`` until page 1 has been fetched.
    leads_found = models.IntegerField(null=True, blank=True)
    dead = models.BooleanField(default=False)

    # The customer's company-size band, riding the query unchanged — never a search
    # axis, since loosening it queries off-ICP rather than widening usefully.
    headcount_min = models.IntegerField(default=1)
    headcount_max = models.IntegerField(default=10000)
    # ISO-3166 alpha-2 of the market searched, stamped onto every lead a page brings in
    # — Lead Finder rows carry no ISO code. **Not the operator's own country**, which is
    # jurisdiction and lives in the environment.
    country_code = models.CharField(max_length=2, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Searches"

    @property
    def query(self):
        """The terms as a ``query.Terms``."""
        from openoutfind.core.pipeline.query import Terms

        return Terms.from_json(self.terms)

    @property
    def headcount(self) -> tuple[int, int]:
        return (self.headcount_min, self.headcount_max)

    def to_filters(self) -> dict:
        """This search as a Lead Finder filter dict — the only thing the provider sees."""
        return self.query.to_filters(self.headcount)

    def __str__(self):
        return self.query.as_text() + (" [dead]" if self.dead else "")


class Page(models.Model):
    """One fired page of a search — the frontier's arm is a search at a page.

    ``number`` is 1-based, at ``PAGE_SIZE`` people a page. ``round`` is the batch it was
    fired in (``walk.run_round``), 0 for the initial population.
    """

    search = models.ForeignKey(Search, on_delete=models.CASCADE, related_name="pages")
    number = models.PositiveSmallIntegerField()
    round = models.PositiveIntegerField()
    # Rows the provider returned, before any were dropped.
    rows = models.PositiveIntegerField(default=0)
    # What the walk expected when it chose this page, N(q) and θ(q) at the mean belief, so
    # both can be read against what came back. None on a written search, chosen unvalued.
    expected_count = models.FloatField(null=True, blank=True)
    expected_rate = models.FloatField(null=True, blank=True)
    leads = models.ManyToManyField("outfind_crm.Lead", through="PageLead",
                                   related_name="pages")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["search", "number"], name="uniq_search_page"),
        ]

    @property
    def at_bottom(self) -> bool:
        """Whether no page lies past this one: it came back empty, reached the count,
        or is the last page the provider's 10,000-row window reaches."""
        found = self.search.leads_found
        return (not self.rows or self.number >= LAST_PAGE
                or (found is not None and self.number * PAGE_SIZE >= found))

    def __str__(self):
        return f"{self.search} @ page {self.number}"


class PageLead(models.Model):
    """One person a page returned — written for every row the page brought back.

    Whether the person was new to the store or not: a person on several pages is
    evidence about each, and the first page that holds them is the one that brought them
    in (``frontier`` reads a page's novelty from that). Rows dropped for no headline or a company at its cap are not
    here, and neither is a marked lead or an anchor.
    """

    page = models.ForeignKey(Page, on_delete=models.CASCADE)
    lead = models.ForeignKey("outfind_crm.Lead", on_delete=models.CASCADE)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["page", "lead"], name="uniq_page_lead"),
        ]


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
