"""
Safety Service — crisis detection and safety checks.

Runs BEFORE LLM call. Fast, deterministic, no LLM dependency.
"""

import re

from app.models.schemas import BiomarkerPayload, SafetyCheck
from app.utils.logger import get_logger

log = get_logger("services.safety")

# Crisis patterns (compiled once)
_CRISIS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(want to|going to|gonna|should) (kill|hurt|end|unalive) (myself|me|my life)",
        r"(wish i was|wish i were) (dead|never born|gone)",
        r"(no point|no reason) (in|to|for) (living|life|being here|going on)",
        r"(better off|be better) (dead|gone|not here|without me)",
        r"(can't|cannot) (take|handle|do) (it|this) (anymore|any more)",
        r"(want to|going to) (die|disappear|vanish)",
        r"(suicide|suicidal|kill myself|end it all|end my life)",
        r"(don't want to|dont want to) (live|be alive|be here|exist|wake up)",
        r"(nobody|no one) (would|will) (care|miss|notice) (if i|when i)",
    ]
]

_HIGH_RISK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(hurting|harming|cutting) (myself|me)",
        r"(have a|made a) (plan|method|way)",
        r"(gave away|giving away) (my|all) (stuff|things|belongings)",
        r"(goodbye|farewell|final|last) (letter|message|note|words)",
        r"(pills|bridge|gun|rope|blade)",
    ]
]

_MODERATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(feel|feeling) (hopeless|worthless|useless|empty|numb)",
        r"(completely|totally|utterly) (alone|isolated|abandoned)",
        r"(nothing|no one) (matters|cares|helps)",
        r"(give|giving|given) up",
        r"(trapped|stuck|no way out|no escape)",
    ]
]


class SafetyService:
    """Fast, deterministic safety checking — no LLM calls."""

    def check(self, text: str, biomarkers: BiomarkerPayload = None) -> SafetyCheck:
        """Run safety checks on user input."""
        flags: list[str] = []
        risk_level = "none"

        if not text:
            return SafetyCheck()

        text_lower = text.lower().strip()

        # Critical patterns
        for pat in _CRISIS_PATTERNS:
            if pat.search(text_lower):
                flags.append("crisis_language_detected")
                risk_level = "critical"
                break

        # High-risk patterns
        if risk_level != "critical":
            for pat in _HIGH_RISK_PATTERNS:
                if pat.search(text_lower):
                    flags.append("high_risk_language")
                    risk_level = "high"
                    break

        # Moderate patterns
        if risk_level == "none":
            for pat in _MODERATE_PATTERNS:
                if pat.search(text_lower):
                    flags.append("distress_language")
                    risk_level = "moderate"
                    break

        # Biomarker escalation check
        if biomarkers:
            if biomarkers.crying > 0.5:
                flags.append("crying_detected")
                if risk_level == "none":
                    risk_level = "low"
            if biomarkers.sentiment < -0.7:
                flags.append("severe_negative_sentiment")
                if risk_level in ("none", "low"):
                    risk_level = "moderate"

        # Recommended action
        action = ""
        if risk_level == "critical":
            action = "Provide crisis resources immediately. Do not attempt therapy."
        elif risk_level == "high":
            action = "Acknowledge distress. Gently assess safety. Provide resources."
        elif risk_level == "moderate":
            action = "Acknowledge feelings. Monitor closely."

        result = SafetyCheck(
            is_crisis=risk_level in ("critical", "high"),
            risk_level=risk_level,
            flags=flags,
            recommended_action=action,
        )

        if result.is_crisis:
            log.warning(f"SAFETY ALERT: {risk_level} — flags={flags}")

        return result
