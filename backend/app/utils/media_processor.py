"""
Unified Media Processor — replaces 3 separate loaders
(audio_processor.py, video_processor.py, file_loader.py).

Single entry point for all media.  Uses ffmpeg + librosa only.
Eliminates moviepy dependency (~200MB saved).
"""

import contextlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from app.config import settings
from app.utils.logger import get_logger

log = get_logger("media_processor")


class MediaProcessor:
    """Unified audio ingestion — single code path, no fallback chains."""

    def __init__(self, target_sr: Optional[int] = None):
        self.target_sr = target_sr or settings.AUDIO_SAMPLE_RATE

    # ── Public API ───────────────────────────────────────────────────────

    def load_audio_file(self, filepath: str) -> Optional[tuple[np.ndarray, int]]:
        """
        Load any audio file → mono float32 numpy array.

        Supports: wav, mp3, m4a, ogg, flac, webm
        Returns: (audio_array, sample_rate) or (None, sr) on failure.
        """
        if not filepath or not os.path.exists(filepath):
            log.warning(f"File not found: {filepath}")
            return None, self.target_sr

        file_size = os.path.getsize(filepath)
        log.info(f"Loading audio: {filepath} ({file_size:,} bytes)")

        ext = Path(filepath).suffix.lower()

        # For formats librosa handles natively
        if ext in (".wav", ".flac", ".ogg"):
            audio, sr = self._load_librosa(filepath)
        else:
            # Convert to WAV via ffmpeg first (handles m4a, mp3, webm, etc.)
            audio, sr = self._load_via_ffmpeg(filepath)

        if audio is None or len(audio) == 0:
            log.warning("All loading methods failed — returning silence")
            return None, self.target_sr

        # Resample if needed
        if sr != self.target_sr:
            log.debug(f"Resampling {sr}Hz → {self.target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # Normalize
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        duration = len(audio) / sr
        log.info(f"Audio loaded: {len(audio):,} samples, {duration:.2f}s")
        return audio, sr

    def extract_audio_from_video(self, video_path: str) -> Optional[tuple[np.ndarray, int]]:
        """Extract audio track from video file via ffmpeg."""
        return self._load_via_ffmpeg(video_path)

    def validate_audio(self, audio: np.ndarray) -> bool:
        """Check if audio has actual speech content."""
        if audio is None or len(audio) == 0:
            return False
        duration = len(audio) / self.target_sr
        if duration < settings.MIN_AUDIO_DURATION:
            return False
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.005:
            log.debug(f"Audio appears silent (RMS={rms:.4f})")
            return False
        return True

    def compute_snr(self, audio: np.ndarray) -> float:
        """Compute signal-to-noise ratio in dB."""
        if len(audio) == 0:
            return 0.0

        signal_power = np.mean(audio**2)
        frame_size = min(1024, len(audio) // 10)
        if frame_size < 100:
            return 10.0

        hop = frame_size // 2
        frame_powers = []
        for i in range(0, len(audio) - frame_size, hop):
            frame = audio[i : i + frame_size]
            frame_powers.append(np.mean(frame**2))

        if not frame_powers:
            return 10.0

        sorted_powers = sorted(frame_powers)
        noise_frames = sorted_powers[: max(1, len(sorted_powers) // 10)]
        noise_power = np.mean(noise_frames)

        if noise_power <= 1e-10:
            return 40.0

        snr = 10 * np.log10(signal_power / noise_power)
        return float(max(0, snr))

    # ── Private Loading Methods ──────────────────────────────────────────

    def _load_librosa(self, filepath: str) -> Optional[tuple[np.ndarray, int]]:
        """Direct librosa load for natively supported formats."""
        try:
            audio, sr = librosa.load(filepath, sr=None, mono=True)
            audio = audio.astype(np.float32)
            log.debug(f"Loaded via librosa: {len(audio)} samples at {sr}Hz")
            return audio, sr
        except Exception as e:
            log.warning(f"librosa failed: {e}")
            return self._load_via_ffmpeg(filepath)

    def _load_via_ffmpeg(self, filepath: str) -> Optional[tuple[np.ndarray, int]]:
        """Convert any audio/video to WAV via ffmpeg, then load."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                filepath,
                "-vn",  # strip video
                "-acodec",
                "pcm_s16le",  # PCM 16-bit
                "-ar",
                str(self.target_sr),  # target sample rate
                "-ac",
                "1",  # mono
                "-loglevel",
                "error",
                tmp_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                log.warning(f"ffmpeg error: {result.stderr[:200]}")
                return None, self.target_sr

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                return None, self.target_sr

            audio, sr = librosa.load(tmp_path, sr=self.target_sr, mono=True)
            log.debug(f"Loaded via ffmpeg: {len(audio)} samples")
            return audio.astype(np.float32), sr

        except subprocess.TimeoutExpired:
            log.warning("ffmpeg timed out")
            return None, self.target_sr
        except Exception as e:
            log.warning(f"ffmpeg load failed: {e}")
            return None, self.target_sr
        finally:
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
