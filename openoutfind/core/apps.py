# openoutfind/core/apps.py
from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = "openoutfind.core"
    # Namespaced: OpenOutreach hosts this app alongside OpenOutSend's own `core`
    # in one registry, and two apps cannot share a label.
    label = "outfind_core"
    default_auto_field = "django.db.models.BigAutoField"
