"""
Biomarker Tracker — tracks biomarker signals across turns for trend analysis.

Simplified: removed trajectory_matcher dependency (trajectory vectors killed).
Masking detection preserved — this is the core value of the system.
"""

from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from app.config import settings
from app.utils.logger import get_logger

log = get_logger("memory.tracker")


# Feature name mapping
FEATURE_NAMES = {
    0: "jitter",
    1: "shimmer",
    2: "f0_variance",
    3: "loudness_range",
    4: "teo",
    5: "hnr",
    6: "speech_rate",
    7: "pause_rate",
    8: "masking_score",
    9: "brow_tension",
    10: "gaze_aversion",
    11: "facial_dynamism",
    12: "stare_duration",
    13: "blink_rate",
    14: "head_nodding",
    15: "head_tilt",
    16: "absolutist_index",
    17: "i_ratio",
    18: "response_latency",
    19: "lexical_density",
    20: "past_tense_ratio",
    21: "filler_density",
    22: "sentiment",
    23: "rumination",
    24: "laughter",
    25: "crying",
    26: "sigh",
    27: "strain",
}


class BiomarkerTracker:
    """
    Rolling-window biomarker tracker with delta/trend analysis
    and cross-modal masking detection.
    """

    def __init__(self, history_size: int = 15):
        self.history: deque = deque(maxlen=history_size)
        self.modalities: deque = deque(maxlen=history_size)
        self.timestamps: deque = deque(maxlen=history_size)
        self.turn_numbers: deque = deque(maxlen=history_size)

        self.sig_threshold = 0.15  # significant change threshold

    def add_turn(self, biomarker: list[float], modality: str = "text", turn: int = 0):
        """Add a new turn's biomarker to history."""
        vec = list(biomarker)
        while len(vec) < settings.BIOMARKER_DIM:
            vec.append(0.0)
        vec = vec[: settings.BIOMARKER_DIM]

        self.history.append(vec)
        self.modalities.append(modality)
        self.timestamps.append(datetime.now())
        self.turn_numbers.append(turn)

    def get_delta(self) -> dict[str, Any]:
        """Get change from previous turn."""
        if len(self.history) < 2:
            return {"has_delta": False}

        curr = np.array(self.history[-1])
        prev = np.array(self.history[-2])
        delta = curr - prev

        changes = []
        for i, change in enumerate(delta):
            if abs(change) >= self.sig_threshold:
                changes.append(
                    {
                        "feature": FEATURE_NAMES.get(i, f"feature_{i}"),
                        "change": float(change),
                        "direction": "increased" if change > 0 else "decreased",
                        "current": float(curr[i]),
                    }
                )
        changes.sort(key=lambda x: abs(x["change"]), reverse=True)

        return {
            "has_delta": True,
            "significant_changes": changes[:5],
            "magnitude": float(np.linalg.norm(delta)),
        }

    def get_trend(self, n: int = 5) -> dict[str, Any]:
        """Analyze trend over recent turns."""
        recent = list(self.history)[-n:]
        if len(recent) < 2:
            return {"has_trend": False}

        # Distress score per turn
        scores = []
        for b in recent:
            d = 0.0
            if len(b) > 0:
                d += max(0, b[0]) * 0.2  # jitter
            if len(b) > 1:
                d += max(0, b[1]) * 0.15  # shimmer
            if len(b) > 22:
                d += max(0, -b[22]) * 0.3  # negative sentiment
            if len(b) > 23:
                d += max(0, b[23]) * 0.1  # rumination
            if len(b) > 25:
                d += max(0, b[25]) * 0.25  # crying
            scores.append(min(1.0, d))

        change = scores[-1] - scores[0]
        trend = "increasing" if change > 0.1 else ("decreasing" if change < -0.1 else "stable")

        return {
            "has_trend": True,
            "distress_trend": trend,
            "distress_scores": scores,
            "current_distress": scores[-1],
            "turns_analyzed": len(recent),
        }

    def detect_masking(self) -> dict[str, Any]:
        """
        Detect contradictions between voice biomarkers and linguistic sentiment.
        THIS IS THE CORE OF THE SYSTEM — where masking detection lives.
        """
        if not self.history:
            return {"detected": False, "reason": "No data"}

        curr = self.history[-1]
        contradictions = []

        # High arousal + positive words = nervous positivity
        if len(curr) > 22 and len(curr) > 3 and curr[3] > 0.5 and curr[22] > 0.3:
            contradictions.append(
                {
                    "type": "nervous_positivity",
                    "detail": "High vocal energy with positive words — possible anxiety masking",
                }
            )

        # Voice tremor + positive sentiment = hidden anxiety
        if len(curr) > 22 and len(curr) > 0 and curr[0] > 0.4 and curr[22] > 0:
            contradictions.append(
                {
                    "type": "hidden_anxiety",
                    "detail": "Voice tremor despite positive words — possible hidden anxiety",
                }
            )

        # Flat pitch + high volume = suppressed emotions
        if len(curr) > 3 and len(curr) > 2 and curr[2] < -0.3 and curr[3] > 0.4:
            contradictions.append(
                {
                    "type": "suppressed_emotions",
                    "detail": "Flat pitch despite high volume — possibly suppressing emotions",
                }
            )

        # Crying + laughter = mixed signals
        if len(curr) > 25 and len(curr) > 24 and curr[24] > 0.3 and curr[25] > 0.3:
            contradictions.append(
                {
                    "type": "mixed_signals",
                    "detail": "Both laughter and crying detected — complex emotional state",
                }
            )

        # High jitter + shimmer but positive sentiment
        if len(curr) > 22 and len(curr) > 1 and curr[0] > 0.3 and curr[1] > 0.3 and curr[22] > 0.2:
            contradictions.append(
                {
                    "type": "voice_text_mismatch",
                    "detail": "Unstable voice paired with positive text — possible masking",
                }
            )

        return {
            "detected": len(contradictions) > 0,
            "contradictions": contradictions,
            "masking_likely": len(contradictions) >= 2,
            "masking_score": min(1.0, len(contradictions) * 0.35),
        }

    def get_summary_for_llm(self) -> str:
        """Generate human-readable biomarker summary for LLM context."""
        if not self.history:
            return ""

        modality = self.modalities[-1] if self.modalities else "text"
        if modality == "text":
            return ""

        curr = self.history[-1]
        lines = [f"=== BIOMARKER ANALYSIS ({modality.upper()}) ==="]

        # Key signals
        signals = []
        if len(curr) > 0 and curr[0] > 0.3:
            signals.append(f"Voice tremor (jitter={curr[0]:.2f}) — anxiety indicator")
        if len(curr) > 1 and curr[1] > 0.3:
            signals.append(f"Voice instability (shimmer={curr[1]:.2f}) — emotional distress")
        if len(curr) > 2 and curr[2] < -0.3:
            signals.append(f"Flat pitch (f0_var={curr[2]:.2f}) — possible depression")
        if len(curr) > 6 and curr[6] > 0.3:
            signals.append(f"Fast speech (rate={curr[6]:.2f}) — anxiety")
        if len(curr) > 6 and curr[6] < -0.3:
            signals.append(f"Slow speech (rate={curr[6]:.2f}) — fatigue/sadness")
        if len(curr) > 22 and curr[22] < -0.3:
            signals.append(f"Negative sentiment ({curr[22]:.2f})")
        if len(curr) > 25 and curr[25] > 0.3:
            signals.append(f"Crying detected ({curr[25]:.2f})")

        if signals:
            lines.append("Current signals:")
            for s in signals[:5]:
                lines.append(f"  • {s}")
        else:
            lines.append("No notable signals detected")

        # Masking
        masking = self.detect_masking()
        if masking["detected"]:
            lines.append("\n⚠️ MASKING DETECTED:")
            for c in masking["contradictions"][:3]:
                lines.append(f"  • {c['detail']}")

        # Trend
        trend = self.get_trend()
        if trend.get("has_trend"):
            lines.append(f"\nDistress trend: {trend['distress_trend'].upper()}")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        return {
            "history_length": len(self.history),
            "modalities_seen": list(set(self.modalities)),
        }
