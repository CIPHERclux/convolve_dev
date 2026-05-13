from typing import Optional
"""
Baseline Manager — user-specific baseline calibration via Welford's algorithm.

Preserved from original with minimal changes (structured logging + config import).
"""

import json
import os

import numpy as np

from app.config import settings
from app.utils.logger import get_logger

log = get_logger("memory.baseline")


class BaselineManager:
    """
    Manages user-specific baseline statistics.
    Prevents natural voice traits from being misclassified as pathology.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        os.makedirs(settings.BASELINES_DIR, exist_ok=True)
        self.stats_file = os.path.join(settings.BASELINES_DIR, f"{user_id}_stats.json")

        self.count = 0
        self.mean: Optional[np.ndarray] = None
        self.m2: Optional[np.ndarray] = None

        self._load_stats()

    def _load_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file) as f:
                    data = json.load(f)
                self.count = data.get("count", 0)
                self.mean = np.array(data["mean"], dtype=np.float32) if data.get("mean") else None
                self.m2 = np.array(data["m2"], dtype=np.float32) if data.get("m2") else None
                log.info(f"Loaded baseline for {self.user_id} (n={self.count})")
            except Exception as e:
                log.warning(f"Could not load baseline: {e}")
                self._init()
        else:
            self._init()

    def _init(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def _save(self):
        try:
            data = {
                "user_id": self.user_id,
                "count": self.count,
                "mean": self.mean.tolist() if self.mean is not None else None,
                "m2": self.m2.tolist() if self.m2 is not None else None,
            }
            with open(self.stats_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.warning(f"Could not save baseline: {e}")

    def get_baseline(self) -> Optional[np.ndarray]:
        return self.mean

    def get_std(self) -> Optional[np.ndarray]:
        if self.count < 2 or self.m2 is None:
            return None
        return np.sqrt(self.m2 / (self.count - 1))

    def update(self, raw_biomarker: np.ndarray, reliability_mask: np.ndarray = None):
        """Update baseline via Welford's algorithm (only trusted features)."""
        dim = settings.BIOMARKER_DIM
        if self.mean is None:
            self.mean = np.zeros(dim, dtype=np.float32)
            self.m2 = np.zeros(dim, dtype=np.float32)

        if reliability_mask is not None:
            trusted = reliability_mask > 0.5
        else:
            trusted = np.ones(dim, dtype=bool)

        if not np.any(trusted):
            return

        self.count += 1
        delta = raw_biomarker - self.mean
        self.mean = np.where(trusted, self.mean + delta / self.count, self.mean)
        delta2 = raw_biomarker - self.mean
        self.m2 = np.where(trusted, self.m2 + delta * delta2, self.m2)

        if self.count % 10 == 0:
            self._save()

    def force_save(self):
        self._save()

    def is_bootstrap(self) -> bool:
        return self.count == 0
