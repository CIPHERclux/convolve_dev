from typing import Optional
"""
Acoustic Engine — 8-feature voice biomarker extraction.

Refactored from the original kairos/extraction/acoustic_engine.py:
- Same core DSP algorithms (jitter, shimmer, F0, TEO, HNR, etc.)
- Structured logging instead of print()
- Clean error handling
"""


import librosa
import numpy as np

from app.config import settings
from app.utils.logger import get_logger

log = get_logger("extraction.acoustic")


class AcousticEngine:
    """
    Extracts 8 acoustic biomarker features from speech audio.

    Features (normalized to [-1, 1], where 0 = typical baseline):
        [0] Jitter       — cycle-to-cycle pitch variation (anxiety)
        [1] Shimmer      — cycle-to-cycle amplitude variation (distress)
        [2] F0 Variance  — pitch variation (flat affect vs animated)
        [3] Loudness     — dynamic range (withdrawn vs agitated)
        [4] TEO          — Teager energy (vocal stress)
        [5] HNR          — harmonics-to-noise (voice clarity)
        [6] Speech Rate  — syllables/sec (anxious vs depressed)
        [7] Pause Rate   — pauses/min (hesitation, cognitive load)
    """

    def __init__(self, sample_rate: Optional[int] = None):
        self.sample_rate = sample_rate or settings.AUDIO_SAMPLE_RATE

        # Calibration baselines (neutral speech reference values)
        self.baselines = {
            "jitter":         {"mean": 0.015, "std": 0.01},
            "shimmer":        {"mean": 0.04,  "std": 0.025},
            "f0_cv":          {"mean": 0.2,   "std": 0.1},
            "loudness_range": {"mean": 20,    "std": 8},
            "teo":            {"mean": 1.0,   "std": 0.3},
            "hnr":            {"mean": 12,    "std": 5},
            "speech_rate":    {"mean": 4.5,   "std": 1.2},
            "pause_rate":     {"mean": 2.5,   "std": 1.5},
        }

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract all 8 acoustic features from audio array."""
        features = np.zeros(8, dtype=np.float32)

        if audio is None or len(audio) < self.sample_rate // 2:
            log.debug("Audio too short or None — returning zeros")
            return features

        # Normalize
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        else:
            return features

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            log.debug(f"Audio near-silent (RMS={rms:.4f})")
            return features

        try:
            # Extract F0 (pitch)
            f0, _voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=60, fmax=400,
                sr=self.sample_rate, frame_length=2048, hop_length=512,
            )
            f0_valid = f0[~np.isnan(f0)]

            if len(f0_valid) < 5:
                f0_valid = self._fallback_f0(audio)

            features[0] = self._extract_jitter(f0_valid)
            features[1] = self._extract_shimmer(audio)
            features[2] = self._extract_f0_variance(f0_valid)
            features[3] = self._extract_loudness_dynamics(audio)
            features[4] = self._extract_teo(audio)
            features[5] = self._extract_hnr(audio, f0_valid)
            features[6] = self._extract_speech_rate(audio, voiced_probs)
            features[7] = self._extract_pause_frequency(audio, len(audio) / self.sample_rate)

            names = ["jitter", "shimmer", "f0_var", "loudness", "teo", "hnr", "speech_rate", "pause"]
            for name, val in zip(names, features, strict=False):
                log.debug(f"{name}={val:.3f}")

        except Exception as e:
            log.error(f"Acoustic extraction failed: {e}", exc_info=True)

        return features

    # ── Normalization ────────────────────────────────────────────────────

    def _normalize(self, raw: float, feature: str) -> float:
        b = self.baselines.get(feature, {"mean": 0, "std": 1})
        z = (raw - b["mean"]) / (b["std"] + 1e-8)
        return float(np.tanh(z * 0.5))

    # ── Feature Extractors ───────────────────────────────────────────────

    def _fallback_f0(self, audio: np.ndarray) -> np.ndarray:
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, fmin=60, fmax=400)
            vals = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if 60 < p < 400:
                    vals.append(p)
            if len(vals) >= 3:
                return np.array(vals)
        except Exception:
            pass
        return np.array([150.0, 150.0, 150.0])

    def _extract_jitter(self, f0: np.ndarray) -> float:
        if len(f0) < 3:
            return 0.0
        try:
            periods = 1.0 / np.maximum(f0, 50)
            diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)
            if mean_period <= 0:
                return 0.0
            return self._normalize(np.mean(diffs) / mean_period, "jitter")
        except Exception:
            return 0.0

    def _extract_shimmer(self, audio: np.ndarray) -> float:
        try:
            fl = int(0.025 * self.sample_rate)
            hl = int(0.010 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0]
            if len(rms) < 3:
                return 0.0
            diffs = np.abs(np.diff(rms))
            mean_rms = np.mean(rms)
            if mean_rms <= 0:
                return 0.0
            return self._normalize(np.mean(diffs) / mean_rms, "shimmer")
        except Exception:
            return 0.0

    def _extract_f0_variance(self, f0: np.ndarray) -> float:
        if len(f0) < 3:
            return 0.0
        try:
            mean_f0 = np.mean(f0)
            if mean_f0 < 50:
                return 0.0
            cv = np.std(f0) / mean_f0
            return self._normalize(cv, "f0_cv")
        except Exception:
            return 0.0

    def _extract_loudness_dynamics(self, audio: np.ndarray) -> float:
        try:
            fl = int(0.025 * self.sample_rate)
            hl = int(0.010 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0]
            if len(rms) < 3:
                return 0.0
            rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=1.0)
            dynamic_range = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)
            return self._normalize(dynamic_range, "loudness_range")
        except Exception:
            return 0.0

    def _extract_teo(self, audio: np.ndarray) -> float:
        if len(audio) < 3:
            return 0.0
        try:
            teo = audio[1:-1] ** 2 - audio[:-2] * audio[2:]
            sig_e = np.mean(audio ** 2)
            if sig_e < 1e-8:
                return 0.0
            return self._normalize(np.mean(np.abs(teo)) / sig_e, "teo")
        except Exception:
            return 0.0

    def _extract_hnr(self, audio: np.ndarray, f0: np.ndarray) -> float:
        if len(f0) < 1 or len(audio) < 2048:
            return 0.0
        try:
            avg_f0 = np.mean(f0)
            if avg_f0 < 60:
                return 0.0
            period_samples = int(self.sample_rate / avg_f0)
            frame_size = min(4096, len(audio))
            frame = audio[:frame_size]
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            if len(autocorr) <= period_samples:
                return 0.0
            search_start = max(1, period_samples - 10)
            search_end = min(len(autocorr) - 1, period_samples + 10)
            if search_end <= search_start:
                return 0.0
            r_max = np.max(autocorr[search_start:search_end])
            r_0 = autocorr[0]
            if r_0 <= 0:
                return 0.0
            r_ratio = np.clip(r_max / r_0, 0.01, 0.99)
            hnr_db = 10 * np.log10(r_ratio / (1 - r_ratio + 1e-8))
            return self._normalize(hnr_db, "hnr")
        except Exception:
            return 0.0

    def _extract_speech_rate(self, audio: np.ndarray, voiced_probs=None) -> float:
        try:
            duration = len(audio) / self.sample_rate
            if duration < 0.5:
                return 0.0
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate, backtrack=False)
            rate = len(onsets) / duration
            return self._normalize(rate, "speech_rate")
        except Exception:
            return 0.0

    def _extract_pause_frequency(self, audio: np.ndarray, duration: float) -> float:
        try:
            if duration < 2:
                return 0.0
            fl = int(0.025 * self.sample_rate)
            hl = int(0.010 * self.sample_rate)
            rms = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0]
            if len(rms) == 0:
                return 0.0
            threshold = np.percentile(rms, 25)
            min_pause_frames = int(0.25 * self.sample_rate / hl)
            is_pause = rms < threshold
            pause_count = 0
            current = 0
            for p in is_pause:
                if p:
                    current += 1
                else:
                    if current >= min_pause_frames:
                        pause_count += 1
                    current = 0
            if current >= min_pause_frames:
                pause_count += 1
            ppm = pause_count / (duration / 60)
            return self._normalize(ppm, "pause_rate")
        except Exception:
            return 0.0
