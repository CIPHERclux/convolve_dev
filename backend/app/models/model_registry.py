"""
Model Registry — Singleton loader for ML models.

Fixes the duplicate-model-loading problem: SentenceTransformer was loaded
twice (feature_engine + colbert_encoder), wasting ~500MB.  Now loaded once.
"""

import threading
from typing import Optional, Any

from app.utils.logger import get_logger

log = get_logger("model_registry")


class ModelRegistry:
    """Thread-safe singleton that loads each ML model exactly once."""

    _lock = threading.Lock()
    _models: dict = {}

    # ── Semantic Encoder ─────────────────────────────────────────────────
    @classmethod
    def get_semantic_encoder(cls):
        """Load SentenceTransformer ONCE, share everywhere."""
        if "semantic" not in cls._models:
            with cls._lock:
                if "semantic" not in cls._models:
                    log.info("Loading SentenceTransformer (all-MiniLM-L6-v2)…")
                    from sentence_transformers import SentenceTransformer
                    cls._models["semantic"] = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    log.info("SentenceTransformer loaded ✓")
        return cls._models["semantic"]

    # ── NLTK / VADER ─────────────────────────────────────────────────────
    @classmethod
    def get_vader(cls):
        """Lazy-load VADER sentiment analyzer."""
        if "vader" not in cls._models:
            with cls._lock:
                if "vader" not in cls._models:
                    try:
                        import nltk
                        for res in ["punkt", "punkt_tab", "vader_lexicon"]:
                            try:
                                nltk.download(res, quiet=True)
                            except Exception:
                                pass
                        from nltk.sentiment.vader import SentimentIntensityAnalyzer
                        cls._models["vader"] = SentimentIntensityAnalyzer()
                        log.info("VADER loaded ✓")
                    except Exception as e:
                        log.warning(f"VADER unavailable: {e}")
                        cls._models["vader"] = None
        return cls._models.get("vader")

    # ── Status ───────────────────────────────────────────────────────────
    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._models.keys())

    @classmethod
    def is_loaded(cls, name: str) -> bool:
        return name in cls._models
