"""Who is running this daemon.

Self-hosted means one operator, so identity is a lookup, not a parameter. This
replaces the ``OperatorSession`` object that used to be threaded through every
call: it was the browser era's session handle, and once the browser went there was
nothing session-like left in it — just the Django ``User`` and the campaign the
handler happened to be working on. The campaign now rides on the deal (a real FK),
and the operator is looked up here.

Nothing is cached across calls. Both reads are a single indexed row and happen at
most once per cycle; a cache would only add a way for a renamed operator to keep
signing emails with their old name until the daemon restarts.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_active_user():
    """The Django ``User`` running the daemon (the onboarded operator)."""
    from django.contrib.auth.models import User

    return User.objects.filter(is_active=True, is_staff=True).order_by("pk").first()


def campaigns():
    """Every campaign this operator runs, oldest first — the cycle's rotation."""
    from openoutfind.core.models import Campaign

    return list(Campaign.objects.filter(users=get_active_user()).order_by("pk"))


def self_profile() -> dict:
    """The operator's own identity, synthesized (not scraped).

    Name comes from the Django user (the agents read ``first_name`` for the seller
    binding, falling back to the username), country from ``SiteConfig``. The contacts
    store uses ``public_identifier`` (the operator email) as the stable operator key.
    """
    from openoutfind.core.models import SiteConfig

    user = get_active_user()
    return {
        "public_identifier": user.email or user.username,
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "country_code": SiteConfig.load().country_code or "",
    }


def seller_name() -> str:
    """The seller's first name as the LLM knows it, with a username fallback."""
    profile = self_profile()
    return (profile.get("first_name") or "").strip() or get_active_user().username


def seller_full_name() -> str:
    """The seller's full name for the prompt's identity binding."""
    profile = self_profile()
    full = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
    return full or get_active_user().username
