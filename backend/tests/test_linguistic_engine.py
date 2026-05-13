"""
Tests for LinguisticEngine — text-based feature extraction.

Tests the 8 linguistic features: absolutist index, I-ratio, response
latency, lexical density, past tense ratio, filler density, sentiment,
and rumination. All are pure functions with no external dependencies.
"""

import numpy as np
import pytest

from app.extraction.linguistic_engine import LinguisticEngine


@pytest.fixture
def engine():
    return LinguisticEngine()


# ─── Absolutist Index ────────────────────────────────────────────────────


class TestAbsolutistIndex:

    def test_high_absolutist_language(self, engine):
        text = "I always fail at everything, nothing ever goes right, never ever"
        features = engine.extract(text)
        assert features[0] > 0.0, "Should detect high absolutist language"

    def test_no_absolutist_language(self, engine):
        text = "Sometimes things go well and other times they don't"
        features = engine.extract(text)
        assert features[0] <= 0.0, "Should not detect absolutist language"


# ─── I-Ratio (Self-Focused Language) ────────────────────────────────────


class TestIRatio:

    def test_high_self_focus(self, engine):
        text = "I feel like I'm always messing up my life and I can't fix myself"
        features = engine.extract(text)
        assert features[1] > 0.0, "Should detect high self-focused language"

    def test_low_self_focus(self, engine):
        text = "The team worked together to solve the problem efficiently"
        features = engine.extract(text)
        assert features[1] < 0.0, "Should not detect self-focused language"


# ─── Lexical Density ────────────────────────────────────────────────────


class TestLexicalDensity:

    def test_low_density_repetitive(self, engine):
        text = "bad bad bad bad bad bad bad bad"
        features = engine.extract(text)
        assert features[3] < 0.0, "Repetitive text should have low density"

    def test_high_density_varied(self, engine):
        text = "The curious fox investigated every mysterious shadow beneath the ancient oak tree"
        features = engine.extract(text)
        assert features[3] > 0.0, "Varied vocabulary should have high density"


# ─── Past Tense Ratio ───────────────────────────────────────────────────


class TestPastTenseRatio:

    def test_past_tense_heavy(self, engine):
        text = "I went to the store and saw my friend who told me she had moved away"
        features = engine.extract(text)
        assert features[4] > 0.0, "Past-tense text should score high"

    def test_present_tense(self, engine):
        text = "I am going to the store and I feel great about life right now"
        features = engine.extract(text)
        assert features[4] <= 0.0, "Present-tense text should score low"


# ─── Sentiment ───────────────────────────────────────────────────────────


class TestSentiment:

    def test_very_negative_crisis(self, engine):
        text = "I don't want to live anymore"
        features = engine.extract(text)
        assert features[6] < -0.5, f"Crisis text should be very negative, got {features[6]}"

    def test_positive_sentiment(self, engine):
        text = "I'm feeling so much better, things are getting better"
        features = engine.extract(text)
        assert features[6] > 0.0, "Positive text should have positive sentiment"

    def test_neutral_sentiment(self, engine):
        text = "I went to the store today"
        features = engine.extract(text)
        assert -0.5 < features[6] < 0.5, "Neutral text should have near-zero sentiment"


# ─── Rumination ──────────────────────────────────────────────────────────


class TestRumination:

    def test_repetitive_negative(self, engine):
        text = "Why does this always happen? Why can't I ever get it right? Why why why"
        features = engine.extract(text)
        assert features[7] > 0.0, "Repetitive 'why' should trigger rumination"

    def test_self_blame_patterns(self, engine):
        text = "It's my fault, I always mess things up, I never do anything right"
        features = engine.extract(text)
        assert features[7] > 0.0, "Self-blame patterns should trigger rumination"


# ─── Output Shape ────────────────────────────────────────────────────────


class TestOutputShape:

    def test_returns_8_features(self, engine):
        features = engine.extract("Hello there")
        assert len(features) == 8
        assert features.dtype == np.float32

    def test_empty_input(self, engine):
        features = engine.extract("")
        assert np.all(features == 0.0)

    def test_whitespace_input(self, engine):
        features = engine.extract("   ")
        assert np.all(features == 0.0)

    def test_features_bounded(self, engine):
        """All features should be in [-1, 1] range."""
        texts = [
            "I always always always never never never everything nothing",
            "I me my mine myself I'm I've I'll I'd me me me",
            "The quick brown fox jumps over the lazy dog",
        ]
        for text in texts:
            features = engine.extract(text)
            assert np.all(features >= -1.0), f"Feature below -1: {features}"
            assert np.all(features <= 1.0), f"Feature above 1: {features}"
