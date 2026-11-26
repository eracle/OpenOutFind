# tests/core/ml/test_embeddings.py
"""The embedding wrapper: ``openoutlearn.embeddings`` with this install's model and cache.

The suite stubs these functions globally so no test needs the ONNX model; the names are
bound here, at import, to the real ones. The embedding itself is ``openoutlearn``'s and
tested there.
"""
from __future__ import annotations

from unittest.mock import patch

from django.conf import settings

from openoutfind.core.conf import CAMPAIGN_CONFIG
from openoutfind.core.ml.embeddings import embed_text, embed_texts


def test_both_hand_openoutlearn_this_installs_model_and_cache_dir():
    """The cache is the state dir's, so the 65 MB model is never downloaded beside a
    throwaway ``--db``."""
    expected = {"model_name": CAMPAIGN_CONFIG["embedding_model"],
                "cache_dir": settings.FASTEMBED_CACHE_DIR}

    with patch("openoutlearn.embeddings.embed_text") as one, \
            patch("openoutlearn.embeddings.embed_texts") as many:
        embed_text("hello")
        embed_texts(["hello", "world"])

    assert one.call_args.kwargs == expected
    assert many.call_args.kwargs == expected
