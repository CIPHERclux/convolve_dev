"""
Special Signals Engine — laughter, crying, sigh, strain detection.

Conservative thresholds — requires multiple indicators for confident detection.
"""

import librosa
import numpy as np

from app.config import settings
from app.utils.logger import get_logger

log = get_logger("extraction.special")


class SpecialSignalsEngine:
    """
    Extracts 4 special audio signals (indices 24-27 in biomarker vector).

    Features (0.0-1.0 scale):
        [0] Laughter  - rhythmic energy bursts + pitch jumps
        [1] Crying    - voice breaks + vocal instability
        [2] Sigh      - sustained declining energy
        [3] Strain    - jitter + shimmer + low HNR
    """

    def __init__(self):
        self.sample_rate = settings.AUDIO_SAMPLE_RATE

    def extract(
        self,
        acoustic_features: np.ndarray = None,
        audio_array: np.ndarray = None,
    ) -> np.ndarray:
        features = np.zeros(8, dtype=np.float32)

        # Conservative defaults
        features[0:4] = 0.05

        if audio_array is None or len(audio_array) < self.sample_rate:
            return features

        try:
            features[0] = self._detect_laughter(audio_array, acoustic_features)
            features[1] = self._detect_crying(audio_array, acoustic_features)
            features[2] = self._detect_sigh(audio_array)
            features[3] = self._detect_strain(acoustic_features)

            log.debug(
                f"laughter={features[0]:.2f} crying={features[1]:.2f} "
                f"sigh={features[2]:.2f} strain={features[3]:.2f}"
            )
        except Exception as e:
            log.warning(f"Special signals error: {e}")

        return features

    def _detect_laughter(self, audio: np.ndarray, af: np.ndarray = None) -> float:
        try:
            duration = len(audio) / self.sample_rate
            if duration < 1.0:
                return 0.05

            evidence = 0.0
            indicators = 0

            fl = int(0.025 * self.sample_rate)
            hl = int(0.010 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0]

            if len(rms) > 20:
                rms_n = (rms - np.mean(rms)) / (np.std(rms) + 1e-8)
                peaks = [i for i in range(1, len(rms_n) - 1)
                         if rms_n[i] > rms_n[i - 1] and rms_n[i] > rms_n[i + 1] and rms_n[i] > 0.5]
                if len(peaks) >= 5:
                    intervals = np.diff(peaks)
                    cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
                    if cv < 0.3:
                        evidence += 0.25
                        indicators += 1

            if af is not None and len(af) > 2 and af[2] > 0.5:
                evidence += 0.2
                indicators += 1

            spec = np.abs(librosa.stft(audio))
            hf = spec[spec.shape[0] * 2 // 3 :, :]
            if np.var(np.mean(hf, axis=0)) > 0.01:
                evidence += 0.15
                indicators += 1

            if indicators >= 2:
                return min(0.7, 0.1 + evidence)
            elif indicators == 1:
                return min(0.3, 0.1 + evidence * 0.5)
            return 0.05
        except Exception:
            return 0.05

    def _detect_crying(self, audio: np.ndarray, af: np.ndarray = None) -> float:
        try:
            duration = len(audio) / self.sample_rate
            if duration < 1.5:
                return 0.05

            evidence = 0.0
            indicators = 0

            fl = int(0.025 * self.sample_rate)
            hl = int(0.010 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0]

            if len(rms) > 10:
                diffs = np.diff(rms)
                threshold = np.std(diffs) * 2.5
                drops = np.sum(diffs < -threshold)
                dps = drops / duration
                if dps > 2.0:
                    evidence += 0.3
                    indicators += 1
                elif dps > 1.0:
                    evidence += 0.15
                    indicators += 1

            if af is not None and len(af) >= 2 and af[0] > 0.4 and af[1] > 0.4:
                evidence += 0.25
                indicators += 1

            if indicators >= 2:
                return min(0.8, 0.1 + evidence)
            elif indicators == 1:
                return min(0.35, 0.1 + evidence * 0.5)
            return 0.05
        except Exception:
            return 0.05

    def _detect_sigh(self, audio: np.ndarray) -> float:
        try:
            if len(audio) / self.sample_rate < 0.5:
                return 0.05
            fl = int(0.05 * self.sample_rate)
            hl = int(0.025 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0]
            score = 0.05
            for i in range(len(rms) - 8):
                seg = rms[i : i + 8]
                if all(seg[j] >= seg[j + 1] * 0.9 for j in range(7)) and np.mean(seg) > 0.15 * np.max(rms):
                    score = min(0.5, score + 0.15)
            return score
        except Exception:
            return 0.05

    def _detect_strain(self, af: np.ndarray = None) -> float:
        if af is None or len(af) < 6:
            return 0.05
        try:
            score = 0.05
            if af[0] > 0.3:
                score += 0.15
            if af[1] > 0.3:
                score += 0.15
            if af[5] < -0.3:
                score += 0.2
            return min(0.7, score)
        except Exception:
            return 0.05
