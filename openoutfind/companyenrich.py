"""CompanyEnrich's bounded, paid people search; no email discovery or hub calls.

Request fields are validated against a checked-in subset of the official OpenAPI.
Compound companyFilter lookups and expansions are deliberately excluded from this
fixed 2-credit/person contract. See docs/companyenrich.md for scope and pricing.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

import requests
from jsonschema import Draft202012Validator, FormatChecker

from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.enrichment.provider import ProviderUnavailable

API_URL = "https://api.companyenrich.com/people/search"
CREDITS_PER_PERSON = 2
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_SCHEMA = json.loads(files("openoutfind").joinpath("companyenrich_query.schema.json").read_text())
_VALIDATOR = Draft202012Validator(_SCHEMA, format_checker=FormatChecker())


def read_query(path: str) -> dict:
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise OpenOutFindError(ErrorType.BAD_CONFIG, "CompanyEnrich query exceeds 64 KiB")
        query = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise OpenOutFindError(ErrorType.BAD_CONFIG, "Cannot read CompanyEnrich query JSON") from None
    return validate_query(query)


def validate_query(query: dict) -> dict:
    if not isinstance(query, dict):
        raise OpenOutFindError(ErrorType.BAD_CONFIG, "CompanyEnrich query must be an object")
    query = {"page": 1, "pageSize": 10, **query}
    error = next(_VALIDATOR.iter_errors(query), None)
    if error is not None:
        # The validator's full message can echo user input. Report the field only.
        field = ".".join(str(part) for part in error.absolute_path) or "query"
        raise OpenOutFindError(ErrorType.BAD_CONFIG,
                              f"Invalid CompanyEnrich {field}; see docs/companyenrich.md")
    for key in ("atCurrentCompanyAfter", "atCurrentCompanyBefore",
                "atCurrentPositionAfter", "atCurrentPositionBefore"):
        value = query.get(key)
        if value is not None:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                valid = "T" in value and parsed.utcoffset() == timedelta(0)
            except ValueError:
                valid = False
            if not valid:
                raise OpenOutFindError(ErrorType.BAD_CONFIG,
                                      f"CompanyEnrich {key} needs an ISO 8601 UTC timestamp")
    if not any(value for key, value in query.items() if key not in ("page", "pageSize")):
        raise OpenOutFindError(ErrorType.BAD_CONFIG, "CompanyEnrich requires a targeting filter")
    return query


def search(api_key: str, query: dict) -> dict:
    """Exactly one HTTP request; errors never become empty results or automatic retries."""
    query = validate_query(query)
    if not api_key:
        raise ProviderUnavailable("Set OPENOUTFIND_COMPANYENRICH_API_KEY", ErrorType.NO_CREDENTIAL)
    try:
        with requests.Session() as session:
            # requests' default adapter has retries disabled. A dedicated session also
            # avoids inheriting the BetterContact polling/retry transport.
            with session.post(API_URL, json=query,
                              headers={"Authorization": f"Bearer {api_key}",
                                       "Accept": "application/json"},
                              timeout=(5, 30), allow_redirects=False, stream=True) as response:
                if response.status_code != 200:
                    error_type = {401: ErrorType.PROVIDER_AUTH, 403: ErrorType.PROVIDER_AUTH,
                                  402: ErrorType.PROVIDER_OUT_OF_CREDITS,
                                  429: ErrorType.PROVIDER_RATE_LIMITED}.get(
                                      response.status_code, ErrorType.PROVIDER_UNAVAILABLE)
                    raise ProviderUnavailable(
                        f"CompanyEnrich HTTP {response.status_code}; not retried", error_type)
                body = bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ProviderUnavailable("CompanyEnrich response exceeded 4 MiB; not retried")
    except requests.RequestException:
        raise ProviderUnavailable("CompanyEnrich request failed; outcome unknown, not retried") from None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise ProviderUnavailable("Invalid CompanyEnrich JSON; not retried") from None
    if (not isinstance(data, dict) or not isinstance(data.get("items"), list)
            or len(data["items"]) > query["pageSize"]
            or any(not isinstance(item, dict) or type(item.get("id")) is not int
                   for item in data["items"])):
        raise ProviderUnavailable("Invalid CompanyEnrich people page; not retried")
    return data


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value, limit: int = 200) -> str:
    return value[:limit].strip() if isinstance(value, str) else ""


def rows_for(data: dict) -> list[dict]:
    """Adapt current professional profiles to the existing local qualification fields.

    Current employment is never inferred from the first historical job. Missing emails,
    employee counts, headlines and names are not invented. No URL is fetched.
    """
    rows = []
    for person in data["items"]:
        url = _text(_object(person.get("socials")).get("linkedin_url"), 2048)
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError:
            continue
        if (len(url) > 200 or parsed.scheme != "https"
                or parsed.hostname not in ("linkedin.com", "www.linkedin.com")
                or parsed.username or port is not None
                or not parsed.path.startswith("/in/") or not parsed.path[4:].strip("/")):
            continue
        url = "https://www.linkedin.com" + parsed.path.rstrip("/")
        experiences = person.get("experiences")
        current = next((job for job in experiences if isinstance(job, dict)
                        and job.get("isCurrent") is True), {}) if isinstance(experiences, list) else {}
        company = _object(current.get("company")) if current.get("type") == "company" else {}
        location = _object(person.get("location"))
        country = _object(location.get("country"))
        code = _text(country.get("code")).upper()
        row = {
            "contact_linkedin_profile_url": url,
            "contact_full_name": _text(person.get("name")),
            "contact_job_title": _text(person.get("position")),
            "contact_seniority": _text(person.get("seniority")),
            "contact_location_country": _text(country.get("name")),
            "company_name": _text(company.get("name") or (current.get("companyName")
                                  if current.get("type") == "unknown" else "")),
            "company_domain": _text(company.get("domain")),
            "company_industry": _text(company.get("industry")),
            "_discovery_source": "companyenrich",
            "_country_code": code if len(code) == 2 and code.isascii() and code.isalpha() else "",
            "_first_name": _text(person.get("first_name"), 100) or None,
            "_last_name": _text(person.get("last_name"), 100) or None,
        }
        # People search has no headline field. The BetterContact-specific headline
        # completeness heuristic does not apply; a position is useful qualification text.
        if row["contact_job_title"]:
            rows.append(row)
    return rows
