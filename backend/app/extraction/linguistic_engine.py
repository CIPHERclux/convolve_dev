"""
Linguistic Engine — 8 text-based features.

Refactored: uses ModelRegistry for VADER, structured logging.
"""

import contextlib
import re
from datetime import datetime

import numpy as np

from app.config import settings
from app.models.model_registry import ModelRegistry
from app.utils.logger import get_logger

log = get_logger("extraction.linguistic")


class LinguisticEngine:
    """
    Extracts 8 linguistic features from text.

    Features (normalized to [-1, 1]):
        [0] Absolutist Index  — "always", "never" usage
        [1] I-ratio           — self-focused language
        [2] Response Latency  — delay before responding
        [3] Lexical Density   — vocabulary richness
        [4] Past Tense Ratio  — stuck in the past
        [5] Filler Density    — cognitive load
        [6] Sentiment         — semantic sentiment (pattern + VADER)
        [7] Rumination        — repetitive negative thought
    """

    def __init__(self):
        self.absolutist_words = set(settings.ABSOLUTIST_WORDS)
        self.filler_words = set(settings.FILLER_WORDS)
        self._init_patterns()

    def _init_patterns(self):
        """Compile sentiment patterns for semantic understanding."""
        very_neg = [
            (r"(world|everyone|they).*(better|fine).*(without me|if i)", -0.95),
            (r"(no|dont|don't) (want to|wanna) (live|be here|exist|be alive)", -0.95),
            (r"(should|want to|gonna|going to) (unalive|kill|end|hurt) (myself|me)", -1.0),
            (r"(wish i was|wish i were) (dead|never born|gone)", -0.95),
            (r"no (point|reason|purpose) (in|to|for) (living|life|being here)", -0.9),
            (r"(better off|be better) (dead|gone|not here)", -0.95),
            (r"(can't|cannot) (take|handle|do) (it|this) (anymore|any more)", -0.85),
            (r"(nothing|no one|nobody) (will ever|ever) (care|love|help|change)", -0.8),
            (r"(completely|totally|utterly) (alone|worthless|hopeless|useless)", -0.85),
            (r"(give|giving|given) up (on|with) (life|everything|myself)", -0.85),
        ]
        neg = [
            (r"(really|so|very|extremely) (sad|depressed|down|upset|hurt)", -0.6),
            (r"(hate|hating) (myself|my life|everything|this)", -0.65),
            (r"(scared|terrified|afraid|anxious|worried) (of|about|that)", -0.5),
            (r"(lonely|alone|isolated|abandoned)", -0.5),
            (r"(crying|cried|cry) (all|every|so much)", -0.55),
            (r"(feel|feeling) (so )?(trapped|stuck|lost|empty|numb)", -0.6),
        ]
        pos = [
            (r"(feel|feeling) (much |so |a lot )?(better|good|great|happy|hopeful)", 0.7),
            (r"(things are|it's|everything is) (getting|looking) better", 0.6),
            (r"(really|so|very) (happy|excited|grateful|thankful)", 0.7),
            (r"(good|great|wonderful|amazing) (day|news|thing|time)", 0.55),
            (r"(love|appreciate) (you|this|that)", 0.5),
        ]
        self._p_very_neg = [(re.compile(p, re.IGNORECASE), s) for p, s in very_neg]
        self._p_neg = [(re.compile(p, re.IGNORECASE), s) for p, s in neg]
        self._p_pos = [(re.compile(p, re.IGNORECASE), s) for p, s in pos]

    def extract(self, text: str, last_system_end_time: str | None = None) -> np.ndarray:
        """Extract all 8 linguistic features."""
        features = np.zeros(8, dtype=np.float32)
        if not text or not text.strip():
            return features

        words = self._tokenize(text)
        sentences = self._sent_tokenize(text)
        if not words:
            return features

        try:
            features[0] = self._absolutist_index(words)
            features[1] = self._i_ratio(words)
            features[2] = self._response_latency(last_system_end_time)
            features[3] = self._lexical_density(words)
            features[4] = self._past_tense_ratio(words)
            features[5] = self._filler_density(text, words)
            features[6] = self._sentiment(text)
            features[7] = self._rumination(text, words, sentences)
        except Exception as e:
            log.warning(f"Linguistic extraction error: {e}")

        return features

    # ── Tokenization ─────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r"[^\w\s']", " ", text.lower())
        return [w for w in text.split() if w]

    def _sent_tokenize(self, text: str) -> list[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    # ── Feature Extractors ───────────────────────────────────────────────

    def _absolutist_index(self, words: list[str]) -> float:
        if not words:
            return 0.0
        count = sum(1 for w in words if w in self.absolutist_words)
        ratio = count / len(words)
        return float(np.clip((ratio - 0.02) / 0.03, -1, 1))

    def _i_ratio(self, words: list[str]) -> float:
        if not words:
            return 0.0
        i_words = {"i", "i'm", "i've", "i'll", "i'd", "me", "my", "mine", "myself"}
        count = sum(1 for w in words if w in i_words)
        ratio = count / len(words)
        return float(np.clip((ratio - 0.08) / 0.07, -1, 1))

    def _response_latency(self, last_time_str: str | None) -> float:
        if not last_time_str:
            return 0.0
        try:
            if isinstance(last_time_str, str):
                last = datetime.fromisoformat(last_time_str.replace("Z", "+00:00"))
            else:
                last = last_time_str
            now = datetime.now()
            if last.tzinfo:
                now = now.replace(tzinfo=last.tzinfo)
            latency = (now - last).total_seconds()
            return float(np.clip((latency - 5) / 25, -1, 1))
        except Exception:
            return 0.0

    def _lexical_density(self, words: list[str]) -> float:
        if not words or len(words) < 3:
            return 0.0
        ratio = len(set(words)) / len(words)
        return float(np.clip((ratio - 0.7) / 0.15, -1, 1))

    def _past_tense_ratio(self, words: list[str]) -> float:
        if not words:
            return 0.0
        past = {"was", "were", "had", "did", "went", "said", "got", "made",
                "came", "thought", "felt", "knew", "took", "saw", "found",
                "gave", "told", "left", "called"}
        count = sum(1 for w in words if w.endswith("ed") or w in past)
        ratio = count / len(words)
        return float(np.clip((ratio - 0.07) / 0.05, -1, 1))

    def _filler_density(self, text: str, words: list[str]) -> float:
        if not words:
            return 0.0
        text_lower = text.lower()
        count = sum(text_lower.count(f) for f in self.filler_words)
        ratio = count / len(words)
        return float(np.clip((ratio - 0.03) / 0.05, -1, 1))

    def _sentiment(self, text: str) -> float:
        """Pattern-based + VADER sentiment (semantic understanding)."""
        score = 0.0
        weight = 0.0

        # Priority 1: Very negative patterns
        for pat, s in self._p_very_neg:
            if pat.search(text):
                score = min(score, s)
                weight = 1.0

        # Priority 2: Negative patterns
        for pat, s in self._p_neg:
            if pat.search(text):
                score += s * 0.5
                weight = max(weight, 0.7)

        # Priority 3: Positive patterns
        for pat, s in self._p_pos:
            if pat.search(text):
                score += s * 0.5
                weight = max(weight, 0.7)

        if weight > 0.5:
            return float(np.clip(score, -1.0, 1.0))

        # Fallback: VADER
        vader = ModelRegistry.get_vader()
        vader_score = 0.0
        if vader:
            with contextlib.suppress(Exception):
                vader_score = vader.polarity_scores(text)["compound"]

        # Simple word backup
        pos_words = {"good", "great", "happy", "love", "wonderful", "amazing", "better", "best"}
        neg_words = {"bad", "sad", "hate", "terrible", "awful", "angry", "upset", "hurt", "pain", "alone"}
        words = self._tokenize(text)
        pc = sum(1 for w in words if w in pos_words)
        nc = sum(1 for w in words if w in neg_words)
        word_score = (pc - nc) / (pc + nc) if (pc + nc) > 0 else 0.0

        final = 0.6 * vader_score + 0.4 * word_score
        return float(np.clip(final, -1.0, 1.0))

    def _rumination(self, text: str, words: list[str], sentences: list[str]) -> float:
        if not words or len(words) < 5:
            return 0.0
        score = 0.0

        # Word repetition
        counts = {}
        for w in words:
            if len(w) > 3:
                counts[w] = counts.get(w, 0) + 1
        repeated = sum(1 for c in counts.values() if c >= 2)
        if counts and repeated / len(counts) > 0.2:
            score += 0.3

        # Negative word repetition
        neg = ["sad", "bad", "hate", "hurt", "pain", "alone", "never", "always", "why", "cant", "can't"]
        if any(counts.get(w, 0) >= 2 for w in neg):
            score += 0.3

        # "Why" repetition
        if text.lower().count("why") >= 2:
            score += 0.2

        # Self-blame
        for pat in ["my fault", "i always", "i never", "i can't", "i cant"]:
            if pat in text.lower():
                score += 0.1

        return float(np.clip(score, 0, 1))
