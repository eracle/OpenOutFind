# openoutfind/settings.py
"""
Minimal Django settings for the OpenOutFind ORM + Django Admin.
"""
import os
import sys
from pathlib import Path

# The agents drive async pydantic-ai from a sync boundary (core/llm.py), so an
# event loop can be live on the thread when the ORM is touched. We only use the
# ORM synchronously, so Django's async-safety guard is safe to relax.
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

ROOT_DIR = Path(__file__).resolve().parent.parent

BASE_DIR = ROOT_DIR

SECRET_KEY = "openoutfind-local-dev-key-change-in-production"

DEBUG = True

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.sites",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "openoutfind.crm.apps.CrmConfig",
    "openoutfind.core.apps.CoreConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "openoutfind.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Installed from a wheel, ROOT_DIR is inside site-packages — which is no place for an
# operator's CRM or a model cache, and may not be writable. Both therefore live under the
# home directory, and a checkout keeps its own `data/` and `.cache/` only because it
# already has them.
def state_dir(root: Path) -> Path:
    """Where the operator's own files live: the checkout, or `~/.openoutfind` installed."""
    return root if (root / "manage.py").exists() else Path.home() / ".openoutfind"


STATE_DIR = state_dir(ROOT_DIR)

# `--db PATH` sets OPENOUTFIND_DB; otherwise the operator's data dir.
DEFAULT_DATA_DIR = STATE_DIR / "data"

# Deliberately *not* derived from DATABASE_PATH: `--db /tmp/scratch.sqlite3` must not send
# fastembed off to re-download its weights beside a throwaway database.
FASTEMBED_CACHE_DIR = STATE_DIR / ".cache" / "fastembed"

DATABASE_PATH = Path(os.environ.get("OPENOUTFIND_DB") or DEFAULT_DATA_DIR / "db.sqlite3").expanduser()
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(DATABASE_PATH),
        # WAL lets `outfind status` read while the daemon holds a write lock;
        # without it a concurrent read fails with "database is locked".
        "OPTIONS": {
            "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
            "transaction_mode": "IMMEDIATE",
        },
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SITE_ID = 1

STATIC_URL = "/static/"
STATIC_ROOT = ROOT_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = ROOT_DIR / "media"

LOGIN_URL = "/admin/login/"

DEFAULT_FROM_EMAIL = "noreply@localhost"
EMAIL_SUBJECT_PREFIX = "CRM: "

LANGUAGE_CODE = "en"
LANGUAGES = [("en", "English")]
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

TESTING = sys.argv[1:2] == ["test"]
