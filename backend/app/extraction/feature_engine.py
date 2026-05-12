"""
Feature Engine — Coordinator for all extraction modules.

KEY ARCHITECTURAL CHANGE (Phase 1 — Decoupled Extraction):
  • Semantic embedding is generated via ModelRegistry (single model load)
  • Biomarkers are extracted as METADATA — they are NOT used for vector search
  • The semantic embedding is the ONLY vector passed to Qdrant for similarity
  • Biomarkers are attached as payload for LLM context
"""

import numpy as np
from typing import Dict, Any, Optional

from app.config import settings
from app.models.model_registry import ModelRegistry
from app.models.schemas import ExtractionResult, BiomarkerPayload
from app.extraction.acoustic_engine import AcousticEngine
from app.extraction.linguistic_engine import LinguisticEngine
from app.extraction.special_signals import SpecialSignalsEngine
from app.utils.media_processor import MediaProcessor
from app.utils.logger import get_logger

log = get_logger("extraction.engine")


class FeatureEngine:
    """
    Orchestrates feature extraction across all modalities.

    Produces:
      1. Semantic embedding (384-dim) → for Qdrant vector search
      2. Biomarker payload (metadata) → for LLM context, NOT for search
    """

    def __init__(self):
        self.acoustic = AcousticEngine()
        self.linguistic = LinguisticEngine()
        self.special = SpecialSignalsEngine()
        self.media = MediaProcessor()
        log.info("FeatureEngine initialized")

    def extract(
        self,
        text: str = "",
        audio_array: Optional[np.ndarray] = None,
        modality: str = "text",
        last_system_end_time: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract all features from input.

        Args:
            text: User's text (transcribed or typed)
            audio_array: Audio numpy array (float32, mono, 16kHz) or None
            modality: "text" or "audio"
            last_system_end_time: ISO timestamp of last system response

        Returns:
            ExtractionResult with semantic embedding + biomarker payload
        """
        log.info(f"Extracting features — modality={modality}")

        has_audio = (
            audio_array is not None
            and len(audio_array) > 0
            and modality in ("audio", "video")
        )

        # ── Acoustic features (indices 0-7) ──────────────────────────────
        if has_audio and self.media.validate_audio(audio_array):
            acoustic_features = self.acoustic.extract(audio_array)
        else:
            acoustic_features = np.zeros(8, dtype=np.float32)

        # ── Visual features (indices 8-15) — zeroed for MVP ─────────────
        visual_features = np.zeros(8, dtype=np.float32)

        # ── Linguistic features (indices 16-23) ─────────────────────────
        linguistic_features = self.linguistic.extract(text, last_system_end_time)

        # ── Special signals (indices 24-31) ──────────────────────────────
        special_features = self.special.extract(
            acoustic_features=acoustic_features,
            audio_array=audio_array,
        )

        # ── Combine into raw 32-D biomarker vector ──────────────────────
        raw_vector = np.concatenate([
            acoustic_features,
            visual_features,
            linguistic_features,
            special_features,
        ]).astype(np.float32)

        # ── Semantic embedding (for Qdrant search) ──────────────────────
        semantic_embedding = self._encode_semantic(text)

        # ── Build payload ────────────────────────────────────────────────
        snr = self.media.compute_snr(audio_array) if has_audio else 0.0
        biomarkers = BiomarkerPayload(
            # Acoustic
            jitter=float(raw_vector[0]),
            shimmer=float(raw_vector[1]),
            f0_variance=float(raw_vector[2]),
            loudness_range=float(raw_vector[3]),
            teo=float(raw_vector[4]),
            hnr=float(raw_vector[5]),
            speech_rate=float(raw_vector[6]),
            pause_rate=float(raw_vector[7]),
            # Linguistic
            absolutist_index=float(raw_vector[16]),
            i_ratio=float(raw_vector[17]),
            response_latency=float(raw_vector[18]),
            lexical_density=float(raw_vector[19]),
            past_tense_ratio=float(raw_vector[20]),
            filler_density=float(raw_vector[21]),
            sentiment=float(raw_vector[22]),
            rumination=float(raw_vector[23]),
            # Special
            laughter=float(raw_vector[24]),
            crying=float(raw_vector[25]),
            sigh=float(raw_vector[26]),
            strain=float(raw_vector[27]),
            # Meta
            modality=modality,
            reliability_scores={"audio_snr": snr},
        )

        return ExtractionResult(
            semantic_embedding=semantic_embedding.tolist(),
            biomarkers=biomarkers,
            raw_vector=raw_vector.tolist(),
            text=text,
        )

    def _encode_semantic(self, text: str) -> np.ndarray:
        """Generate semantic embedding using shared model."""
        if not text or not text.strip():
            return np.zeros(settings.SEMANTIC_DIM, dtype=np.float32)
        try:
            encoder = ModelRegistry.get_semantic_encoder()
            embedding = encoder.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            log.warning(f"Semantic encoding failed: {e}")
            return np.zeros(settings.SEMANTIC_DIM, dtype=np.float32)
