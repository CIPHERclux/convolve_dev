"""
Pydantic Schemas — request/response models for the API.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SessionStartRequest(BaseModel):
    user_id: str = Field(default="default_user", description="User identifier")


class TextChatRequest(BaseModel):
    session_id: str
    text: str = Field(..., min_length=1, description="User message text")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal Models (passed between services)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProcessedMedia(BaseModel):
    """Output of the media processor."""
    audio_array: Optional[Any] = None  # np.ndarray (not serializable)
    sample_rate: int = 16000
    duration: float = 0.0
    modality: str = "text"  # "text" | "audio"
    text: str = ""


class BiomarkerPayload(BaseModel):
    """Biomarker telemetry — attached to Qdrant payload, NOT used for search."""
    # Acoustic (indices 0-7)
    jitter: float = 0.0
    shimmer: float = 0.0
    f0_variance: float = 0.0
    loudness_range: float = 0.0
    teo: float = 0.0
    hnr: float = 0.0
    speech_rate: float = 0.0
    pause_rate: float = 0.0
    # Linguistic (indices 16-23)
    absolutist_index: float = 0.0
    i_ratio: float = 0.0
    response_latency: float = 0.0
    lexical_density: float = 0.0
    past_tense_ratio: float = 0.0
    filler_density: float = 0.0
    sentiment: float = 0.0
    rumination: float = 0.0
    # Special (indices 24-27)
    laughter: float = 0.0
    crying: float = 0.0
    sigh: float = 0.0
    strain: float = 0.0
    # Meta
    modality: str = "text"
    reliability_scores: dict[str, float] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Full output from the feature extraction pipeline."""
    semantic_embedding: list[float] = Field(default_factory=list)
    biomarkers: BiomarkerPayload = Field(default_factory=BiomarkerPayload)
    raw_vector: list[float] = Field(default_factory=list)  # Full 32-D for tracker
    text: str = ""


class SafetyCheck(BaseModel):
    """Output of safety service."""
    is_crisis: bool = False
    risk_level: str = "none"  # "none" | "low" | "moderate" | "high" | "critical"
    flags: list[str] = Field(default_factory=list)
    recommended_action: str = ""


class MemoryContext(BaseModel):
    """Retrieved context from memory for LLM."""
    relevant_memories: list[dict[str, Any]] = Field(default_factory=list)
    user_profile_summary: str = ""
    biomarker_summary: str = ""
    episodic_summaries: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """A tool call made by the LLM during response generation."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Response Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SessionStartResponse(BaseModel):
    session_id: str
    user_id: str
    message: str = "Session started"


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    turn_number: int
    user_transcription: Optional[str] = None
    biomarkers: Optional[BiomarkerPayload] = None
    masking_detected: bool = False
    masking_details: Optional[dict[str, Any]] = None
    safety: Optional[SafetyCheck] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    emotion_tag: str = "neutral"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ProfileResponse(BaseModel):
    user_id: str
    facts: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    qdrant_connected: bool = False
    models_loaded: list[str] = Field(default_factory=list)
