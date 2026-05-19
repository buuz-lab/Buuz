import os

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

_MIN_SAMPLES = 500


class Calibrator:
    """
    Isotonic-regression probability calibrator.

    Pass-through when n_samples < 500 (not enough data to fit reliably).
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self._passthrough: bool = True

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "Calibrator":
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        if len(raw_probs) < _MIN_SAMPLES:
            self._passthrough = True
            return self
        self._passthrough = False
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw_probs, outcomes)
        return self

    def transform(self, raw_prob: float) -> float:
        if self._passthrough or self._iso is None:
            return float(raw_prob)
        return float(self._iso.predict([raw_prob])[0])

    def brier_score(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> float:
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        calibrated = np.array([self.transform(float(p)) for p in raw_probs])
        return float(np.mean((calibrated - outcomes) ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"iso": self._iso, "passthrough": self._passthrough}, path)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibrator model not found: {path}")
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj._iso = state["iso"]
        obj._passthrough = state["passthrough"]
        return obj
