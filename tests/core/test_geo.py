# tests/core/test_geo.py
"""Two country sets: the broad email opt-in one (``is_gdpr_protected``, read by the
OpenOutreach wizard) and the EEA/UK/CH data-collection one (``is_eea_located``, which gates
the hub give-back). An unknown country is treated as protected by both."""
import pytest

from openoutfind.core.geo import GDPR_COUNTRY_CODES, is_eea_located, is_gdpr_protected

EU = {"at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr", "hu", "ie",
      "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro", "sk", "si", "es", "se"}


def test_every_eu_member_is_in_both_sets():
    assert EU <= GDPR_COUNTRY_CODES
    assert all(is_eea_located(code) for code in EU)


@pytest.mark.parametrize("code,opt_in,eea", [
    ("DE", True, True),     # case-insensitive
    ("no", True, True),     # EEA, not EU
    ("li", True, True),
    ("gb", True, True),
    ("ch", True, True),
    ("Br", True, False),    # email opt-in, but collectable: not EEA/UK/CH
    ("ca", True, False),
    ("au", True, False),
    ("jp", True, False),
    ("kr", True, False),
    ("nz", True, False),
    ("us", False, False),
    ("in", False, False),
    ("sg", False, False),
    (None, True, True),     # unknown is protected
    ("", True, True),
])
def test_each_country_falls_in_its_sets(code, opt_in, eea):
    assert is_gdpr_protected(code) is opt_in
    assert is_eea_located(code) is eea


def test_a_blank_country_is_eea_located():
    assert is_eea_located("   ") is True
