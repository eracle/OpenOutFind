# openoutfind/core/admin.py
from django.contrib import admin

from openoutfind.core.models import Page, Search
from openoutfind.discovery import describe_filters


@admin.register(Search)
class SearchAdmin(admin.ModelAdmin):
    """The walk's searches — what each one asks the provider, and how far it paged.

    There is no fitness column: a page's fitness is recomputed from its people on the GP
    as it stands (``fitness.page_fitness``), so a stored number would only show a stale one.
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
    """Every fired page — the walk's individuals — and which the population holds."""

    list_display = ("__str__", "generation", "rows", "people", "in_population",
                    "created_at")
    list_filter = ("in_population", "generation")
    readonly_fields = ("search", "number", "generation", "rows", "in_population",
                       "created_at")

    @admin.display(description="people")
    def people(self, obj):
        """People on the page–lead table for this page."""
        return obj.leads.count()
