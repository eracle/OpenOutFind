# openoutfind/core/ml/embeddings.py
"""Fastembed text embedding utilities — thin wrapper over ``openoutlearn.embeddings``,
supplying this install's model name and cache dir."""
from __future__ import annotations

import numpy as np

from django.conf import settings

from openoutlearn import embeddings as _shared

from openoutfind.core.conf import CAMPAIGN_CONFIG


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string → 384-dim numpy array."""
    return _shared.embed_text(
        text,
        model_name=CAMPAIGN_CONFIG["embedding_model"],
        cache_dir=settings.FASTEMBED_CACHE_DIR,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed multiple texts → (N, 384) numpy array."""
    return _shared.embed_texts(
        texts,
        model_name=CAMPAIGN_CONFIG["embedding_model"],
        cache_dir=settings.FASTEMBED_CACHE_DIR,
    )
