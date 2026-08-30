#!/usr/bin/env python
"""
Bootstrap script for initial CRM data.

Ensures the default Site exists.
Idempotent — safe to run multiple times.
"""
import logging

logger = logging.getLogger(__name__)


def setup_crm():
    from django.contrib.sites.models import Site

    # Ensure default Site exists
    Site.objects.get_or_create(id=1, defaults={"domain": "localhost", "name": "localhost"})

    logger.debug("CRM setup complete.")
