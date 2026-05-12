"""
Convolve MAS — Centralized Configuration

All settings flow from environment variables with sensible defaults.
Supports: LLM_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY (auto-detected).
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "Convolve MAS"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM ──────────────────────────────────────────────────────────────
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_BASE_URL: Optional[str] = None
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.7

    # Extra keys (read from .env but not used directly as fields)
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── Qdrant ───────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "convolve_memories"

    # ── Audio ────────────────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16000
    MIN_AUDIO_DURATION: float = 0.5  # seconds
    MAX_AUDIO_DURATION: float = 300.0  # 5 minutes

    # ── Feature Dimensions ───────────────────────────────────────────────
    SEMANTIC_DIM: int = 384  # all-MiniLM-L6-v2 output
    BIOMARKER_DIM: int = 32  # 8 acoustic + 8 visual + 8 linguistic + 8 special
    ACOUSTIC_DIM: int = 8
    LINGUISTIC_DIM: int = 8
    SPECIAL_DIM: int = 8

    # ── Reliability Thresholds ───────────────────────────────────────────
    MIN_SNR_THRESHOLD: float = 5.0
    MIN_FACE_CONFIDENCE: float = 0.5

    # ── Static Modality Weights (for reliability gating) ─────────────────
    WEIGHT_ACOUSTIC: float = 0.85
    WEIGHT_VISUAL: float = 0.80
    WEIGHT_LINGUISTIC: float = 0.95

    # ── Masking Detection ────────────────────────────────────────────────
    MASKING_THRESHOLD: float = 0.6
    CRISIS_THRESHOLD: float = 0.8

    # ── Episodic Summarization ───────────────────────────────────────────
    EPISODIC_ROLLUP_INTERVAL: int = 4  # turns between episodic summaries

    # ── Storage Paths ────────────────────────────────────────────────────
    DATA_DIR: str = "./data"
    PROFILES_DIR: str = "./data/profiles"
    BASELINES_DIR: str = "./data/baselines"
    UPLOAD_DIR: str = "./data/uploads"

    # ── Linguistic Feature Constants ─────────────────────────────────────
    ABSOLUTIST_WORDS: list = [
        "always", "never", "nothing", "everything", "everyone",
        "nobody", "completely", "totally", "absolutely", "entirely",
        "impossible", "definitely", "certainly", "forever", "constantly",
    ]
    FILLER_WORDS: list = [
        "um", "uh", "like", "you know", "i mean", "sort of",
        "kind of", "basically", "actually", "literally",
    ]

    @model_validator(mode="after")
    def resolve_llm_config(self):
        """Auto-detect API key source and configure provider after .env is loaded."""
        # Priority: LLM_API_KEY > OPENAI_API_KEY > GROQ_API_KEY
        if not self.LLM_API_KEY:
            if self.OPENAI_API_KEY:
                self.LLM_API_KEY = self.OPENAI_API_KEY
            elif self.GROQ_API_KEY:
                self.LLM_API_KEY = self.GROQ_API_KEY
                # Auto-configure Groq provider
                if not self.LLM_BASE_URL:
                    self.LLM_BASE_URL = "https://api.groq.com/openai/v1"
                if self.LLM_MODEL == "gpt-4o-mini":
                    self.LLM_MODEL = "llama-3.3-70b-versatile"
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Singleton instance
settings = Settings()

# Ensure data directories exist
for dir_path in [settings.DATA_DIR, settings.PROFILES_DIR, settings.BASELINES_DIR, settings.UPLOAD_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
