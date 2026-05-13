"""
Tests for SafetyService — crisis detection and risk classification.

SafetyService is deterministic (no LLM, no external calls), making it
the most important module to test: its correctness directly affects
user safety.
"""

import pytest

from app.models.schemas import BiomarkerPayload
from app.services.safety_service import SafetyService


@pytest.fixture
def safety():
    return SafetyService()


# ─── Critical Risk Detection ────────────────────────────────────────────


class TestCriticalPatterns:
    """Test that critical crisis language is always flagged."""

    @pytest.mark.parametrize("text", [
        "I want to kill myself",
        "I'm going to end my life",
        "I wish I was dead",
        "I wish I were never born",
        "No point in living anymore",
        "Everyone would be better off without me",
        "I don't want to be alive anymore",
        "Nobody would care if I disappeared",
        "I want to die",
    ])
    def test_critical_patterns_detected(self, safety, text):
        result = safety.check(text)
        assert result.is_crisis is True, f"Failed to detect crisis: '{text}'"
        assert result.risk_level == "critical"
        assert len(result.flags) > 0
        assert result.recommended_action != ""

    def test_critical_case_insensitive(self, safety):
        result = safety.check("I WANT TO KILL MYSELF")
        assert result.is_crisis is True
        assert result.risk_level == "critical"


# ─── High Risk Detection ────────────────────────────────────────────────


class TestHighRiskPatterns:
    """Test that high-risk language is flagged appropriately."""

    @pytest.mark.parametrize("text", [
        "I've been hurting myself lately",
        "I have a plan to end it",
        "I gave away my stuff to friends",
        "This is my goodbye letter",
    ])
    def test_high_risk_detected(self, safety, text):
        result = safety.check(text)
        assert result.is_crisis is True
        assert result.risk_level == "high"
        assert "high_risk_language" in result.flags


# ─── Moderate Risk Detection ────────────────────────────────────────────


class TestModeratePatterns:
    """Test moderate distress detection — should NOT be flagged as crisis."""

    @pytest.mark.parametrize("text", [
        "I feel completely alone",
        "Nothing matters anymore",
        "I'm giving up",
        "I feel trapped with no way out",
        "I am totally isolated",
    ])
    def test_moderate_detected_not_crisis(self, safety, text):
        result = safety.check(text)
        assert result.is_crisis is False
        assert result.risk_level == "moderate"
        assert "distress_language" in result.flags


# ─── Safe Input ──────────────────────────────────────────────────────────


class TestSafeInput:
    """Test that normal conversation is not flagged."""

    @pytest.mark.parametrize("text", [
        "I had a great day today!",
        "The weather is nice outside",
        "I'm working on a project for school",
        "My friend came over for dinner",
        "I feel happy and grateful",
        "",
    ])
    def test_safe_input_no_flags(self, safety, text):
        result = safety.check(text)
        assert result.is_crisis is False
        assert result.risk_level == "none"
        assert len(result.flags) == 0


# ─── Biomarker Escalation ───────────────────────────────────────────────


class TestBiomarkerEscalation:
    """Test that biomarker signals can escalate risk."""

    def test_crying_adds_low_risk(self, safety):
        biomarkers = BiomarkerPayload(crying=0.7)
        result = safety.check("I don't know what to say", biomarkers)
        assert "crying_detected" in result.flags

    def test_severe_negative_sentiment_adds_moderate(self, safety):
        biomarkers = BiomarkerPayload(sentiment=-0.8)
        result = safety.check("I don't know", biomarkers)
        assert "severe_negative_sentiment" in result.flags
        assert result.risk_level in ("moderate", "high", "critical")

    def test_biomarkers_dont_override_critical(self, safety):
        """Biomarkers shouldn't downgrade a critical text result."""
        biomarkers = BiomarkerPayload(sentiment=0.5)  # positive biomarker
        result = safety.check("I want to kill myself", biomarkers)
        assert result.risk_level == "critical"


# ─── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_text(self, safety):
        result = safety.check("")
        assert result.is_crisis is False
        assert result.risk_level == "none"

    def test_none_biomarkers(self, safety):
        result = safety.check("hello", None)
        assert result.is_crisis is False

    def test_whitespace_only(self, safety):
        result = safety.check("   ")
        assert result.is_crisis is False
