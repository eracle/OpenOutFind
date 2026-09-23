import re

from django.db import models
from django.utils.translation import gettext_lazy as _

# Employer strings that mean "no company", lowercased with punctuation collapsed to
# spaces. Each one was seen in a live Lead Finder page stapled to an unrelated domain.
PLACEHOLDER_NAMES = frozenset({
    "self employed",
    "freelance",
    "freelancer",
    "independent",
    "none",
    "none at this time",
    "n a",
    "stealth",
    "stealth startup",
    "stealth mode",
    "stealth mode startup",
    "stealth mode startup company",
    "confidential",
    "confidential careers",
    "private company",
})


class Company(models.Model):
    """The organisation a Lead works for, stored once.

    Normalised out of ``Lead`` because the company record is worth keeping in its own
    right, and because many leads in one ICP share an employer — a page of discovery
    typically carries only a handful of distinct companies.

    **Identity is ``key``, not the domain.** A unique constraint cannot cover "the
    domain when there is one, the name otherwise", so the natural key is computed and
    stored: the lowercased domain, or ``name:<lowercased name>`` when the provider gave
    no domain.

    **A caution the provider forces.** Lead Finder fuzzy-matches the company it staples
    to a lead row — see the ``TEXT_FIELDS`` note in ``discovery.py``, where a boutique
    law firm's founder comes back as Meta. A Company row is *what the provider said*,
    not verified truth, and anything treating it as an account (per-company send caps,
    account-level suppression) inherits that error.

    **A placeholder employer is no employer.** "Self employed", "Stealth startup" and "None
    at this time" are what a profile says when it names no company, and the provider
    fuzzy-matches them to a real domain like any other name — hundreds of unrelated leads
    then share one row carrying somebody else's website. ``from_row`` returns ``None`` for
    them, exactly as for a row that named nothing.
    """

    class Meta:
        verbose_name = _("Company")
        verbose_name_plural = _("Companies")

    # The computed natural key — see ``key_for``.
    key = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=200, null=True, blank=True, default=None)
    domain = models.CharField(max_length=200, null=True, blank=True, default=None)
    creation_date = models.DateTimeField(auto_now_add=True)
    update_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name or self.domain or self.key

    @staticmethod
    def key_for(name: str | None, domain: str | None) -> str:
        """The natural key for a name/domain pair, or ``""`` when there is neither."""
        if domain:
            return domain.lower()
        return f"name:{name.lower()}" if name else ""

    @staticmethod
    def is_placeholder(name: str | None) -> bool:
        """Whether ``name`` is a stand-in for "no company" rather than a company."""
        normalised = " ".join(re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).split())
        return normalised in PLACEHOLDER_NAMES

    @classmethod
    def from_row(cls, name: str | None, domain: str | None) -> "Company | None":
        """Get or create the Company for one discovery row. ``None`` when it named none."""
        if cls.is_placeholder(name):
            return None
        key = cls.key_for(name, domain)
        if not key:
            return None
        company, _created = cls.objects.get_or_create(
            key=key, defaults={"name": name, "domain": domain})
        return company
