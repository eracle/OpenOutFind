# openoutfind/core/admin.py
from django.contrib import admin

from openoutfind.core.models import Page, Search
from openoutfind.discovery import describe_filters


@admin.register(Search)
class SearchAdmin(admin.ModelAdmin):
    """The walk's searches — what each one asks the provider, and how far it paged.

    There is no value column: the frontier values a search from the store's people on the
    GP as it stands (``frontier``), so a stored number would only show a stale one.
    """

    list_display = ("id", "query_filters", "origin", "move", "leads_found", "dead",
                    "page_count", "created_at")
    list_filter = ("origin", "move", "dead")
    readonly_fields = ("query_filters", "key", "terms", "origin", "move", "leads_found",
                       "dead", "headcount_min", "headcount_max", "country_code",
                       "created_at")
    date_hierarchy = "created_at"

    @admin.display(description="query")
    def query_filters(self, obj):
        """The search rendered as the filters the provider sees."""
        return describe_filters(obj.to_filters())

    @admin.display(description="pages")
    def page_count(self, obj):
        return obj.pages.count()


@admin.register(Page)
class PageAdmin(admin.ModelAdmin):
    """Every fired page, and the round that fired it."""

    list_display = ("__str__", "round", "rows", "people", "created_at")
    list_filter = ("round",)
    readonly_fields = ("search", "number", "round", "rows", "created_at")

    @admin.display(description="people")
    def people(self, obj):
        """People on the page–lead table for this page."""
        return obj.leads.count()
