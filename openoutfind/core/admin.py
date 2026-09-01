# openoutfind/core/admin.py
from django.contrib import admin

from openoutfind.core.models import Keyword, QueryNode
from openoutfind.discovery import describe_filters


@admin.register(QueryNode)
class QueryNodeAdmin(admin.ModelAdmin):
    """The discovery walk, node by node — what was searched, how deep, and what it found.

    There is no value column to display: a node's estimate is counted from the label
    store every time it is needed (``select.estimate``), so showing a stored number here
    would only show one that had gone stale.
    """

    list_display = (
        "id", "query", "state", "next_offset", "leads_found",
        "lead_yield", "updated_at",
    )
    list_filter = ("state",)
    readonly_fields = (
        "query", "token_key", "parent", "next_offset", "state",
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
