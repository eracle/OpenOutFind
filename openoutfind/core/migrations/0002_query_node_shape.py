"""Configuration leaves the database; what the pipeline produced stays, in its own shape.

``SiteConfig`` was one table holding two unlike things. The answers a human gave — the
model and its key, the finder keys, the country, the ICP text — are read from
``OPENOUTFIND_*`` on every run now (``core/config.py``) and are not carried forward: an
install re-states them in its environment, which is the only place a child is configured.

Everything the install *produced* is carried, because none of it can be re-derived:

  * the invented ideal leads become ``Lead`` rows with ``synthetic=True``, profile, fields
    and embedding each landing in the column that already exists for it
  * the ICP size band and target country land on every query node, which is what they
    were always attributes of

``model_blob`` is dropped rather than moved. Nothing ever read it back — the fit is
reproduced from the label rows whenever the evidence changes — so it was a cache of a
derived value, and the ranking model is unaffected by its going.
"""
import numpy as np
from django.db import migrations, models


def carry_produced_state(apps, schema_editor):
    """Move the anchors onto ``Lead`` and the query shape onto ``QueryNode``."""
    SiteConfig = apps.get_model("outfind_core", "SiteConfig")
    QueryNode = apps.get_model("outfind_core", "QueryNode")
    Lead = apps.get_model("outfind_crm", "Lead")

    config = SiteConfig.objects.first()
    if config is None:
        return

    QueryNode.objects.update(
        headcount_min=config.headcount_min,
        headcount_max=config.headcount_max,
        country_code=config.country_code,
    )

    profiles = list(config.anchor_profiles or [])
    source_fields = list(config.anchor_source_fields or [])
    embeddings = _anchor_embeddings(config, len(profiles))
    for index, profile in enumerate(profiles):
        Lead.objects.create(
            synthetic=True,
            profile_url=None,
            profile_text=profile,
            source_fields=source_fields[index] if index < len(source_fields) else {},
            country_code=config.country_code,
            embedding=embeddings[index].tobytes() if embeddings is not None else None,
        )


def _anchor_embeddings(config, count: int):
    """The stored anchor vectors as ``(N, dim)``, or ``None`` if there are none.

    An install anchored before the embeddings were kept, or one whose blob does not
    divide by its profile count, gets ``None`` — the anchors are still worth their text
    and their fields, and a lead with no vector is a case the qualifier already handles.
    """
    blob = config.anchor_embeddings
    if not blob or not count:
        return None
    flat = np.frombuffer(bytes(blob), dtype=np.float32)
    if flat.size % count:
        return None
    return flat.reshape(count, -1)


class Migration(migrations.Migration):

    dependencies = [
        ("outfind_core", "0001_initial"),
        # The anchors land as Lead rows, so the column that marks one has to exist first.
        ("outfind_crm", "0002_lead_synthetic"),
    ]

    operations = [
        migrations.AddField(
            model_name="querynode",
            name="country_code",
            field=models.CharField(blank=True, default="", max_length=2),
        ),
        migrations.AddField(
            model_name="querynode",
            name="headcount_max",
            field=models.IntegerField(default=10000),
        ),
        migrations.AddField(
            model_name="querynode",
            name="headcount_min",
            field=models.IntegerField(default=1),
        ),
        # Nothing is restored on the way down: the answers are in the environment and the
        # produced state is now in the tables above, so a reverse would have to invent a
        # config row to put nothing in.
        migrations.RunPython(carry_produced_state, migrations.RunPython.noop),
        migrations.DeleteModel(
            name="SiteConfig",
        ),
    ]
