# openoutfind/core/admin.py
from django.contrib import admin

from openoutfind.core.models import Campaign, Keyword, QueryNode, SiteConfig
from openoutfind.crm.models import DealState
from openoutfind.crm.models.deal import Deal
from openoutfind.discovery import describe_filters


@admin.register(SiteConfig)
class SiteConfigAdmin(admin.ModelAdmin):
    list_display = ("__str__", "ai_model", "llm_api_base")

    def has_add_permission(self, request):
        return not SiteConfig.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Campaign)
class CampaignAdmin(admin.ModelAdmin):
    list_display = ("name", "phase")
    filter_horizontal = ("users",)

    @admin.display(description="phase")
    def phase(self, obj):
        """Cold (still part-steering on invented profiles) vs learning (real evidence rules).

        The anchors are permanent, so this can no longer read their count — it mirrors
        ``BayesianQualifier.is_cold`` instead: cold until real acceptances reach
        ``ANCHOR_COUNT``, learning from there, with the (also permanent) anchor count
        shown alongside so the two never look conflated.
        """
        from openoutfind.core.pipeline.icp import ANCHOR_COUNT

        n_anchors = len(obj.anchor_profiles or [])
        n_real = Deal.objects.filter(
            campaign=obj, lead_id__isnull=False,
        ).exclude(state=DealState.FAILED).count()
        if not n_anchors:
            return "learning (unanchored)"
        if n_real < ANCHOR_COUNT:
            return f"cold ({n_real}/{ANCHOR_COUNT} real, {n_anchors} anchor{'' if n_anchors == 1 else 's'})"
        return f"learning ({n_anchors} anchor{'' if n_anchors == 1 else 's'} + {n_real} real)"


@admin.register(QueryNode)
class QueryNodeAdmin(admin.ModelAdmin):
    """The discovery walk, node by node — what was searched, how deep, and what it found.

    There is no value column to display: a node's estimate is counted from the label
    store every time it is needed (``select.estimate``), so showing a stored number here
    would only show one that had gone stale.
    """

    list_display = (
        "id", "query", "campaign", "state", "next_offset", "leads_found",
        "lead_yield", "updated_at",
    )
    list_filter = ("state", "campaign")
    readonly_fields = (
        "campaign", "query", "token_key", "parent", "next_offset", "state",
        "leads_found", "lead_yield", "created_at", "updated_at",
    )
    date_hierarchy = "created_at"

    @admin.display(description="query")
    def query(self, obj):
        """The node's keyword set, rendered as the region it searches."""
        return describe_filters(obj.to_filters())

    @admin.display(description="leads")
    def lead_yield(self, obj):
        """First-touch leads this node surfaced."""
        return obj.leads.count()


@admin.register(Keyword)
class KeywordAdmin(admin.ModelAdmin):
    """The vocabulary — every ``(field, token)`` a query node can be built from."""

    list_display = ("__str__", "field", "token", "node_count", "created_at")
    list_filter = ("field",)
    search_fields = ("token",)

    @admin.display(description="nodes")
    def node_count(self, obj):
        """How many query nodes carry this keyword."""
        return obj.nodes.count()
