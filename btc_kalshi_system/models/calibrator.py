import os

import joblib
import numpy as np
from loguru import logger
from sklearn.isotonic import IsotonicRegression

_MIN_SAMPLES = 300


class Calibrator:
    """
    Isotonic-regression probability calibrator.

    Pass-through when n_samples < _MIN_SAMPLES (not enough data to fit reliably).
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self._passthrough: bool = True
        self._n_samples: int = 0
        self._prev_brier: float | None = None

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "Calibrator":
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        self._n_samples = len(raw_probs)
        if len(raw_probs) < _MIN_SAMPLES:
            self._passthrough = True
            return self

        # Snapshot current state in case we need to revert (monotonicity guard)
        prev_iso = self._iso
        prev_passthrough = self._passthrough

        self._passthrough = False
        new_iso = IsotonicRegression(out_of_bounds="clip")
        new_iso.fit(raw_probs, outcomes)
        self._iso = new_iso

        # Monotonicity guard: revert if new Brier is worse than previous
        new_brier = self.brier_score(raw_probs, outcomes)
        if self._prev_brier is not None and new_brier > self._prev_brier:
            logger.warning(
                f"Calibrator: new Brier {new_brier:.4f} > previous {self._prev_brier:.4f} — reverting"
            )
            self._iso = prev_iso
            self._passthrough = prev_passthrough
        else:
            self._prev_brier = new_brier

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
        joblib.dump({
            "iso": self._iso,
            "passthrough": self._passthrough,
            "n_samples": self._n_samples,
            "prev_brier": self._prev_brier,
        }, path)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibrator model not found: {path}")
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj._iso = state["iso"]
        obj._passthrough = state["passthrough"]
        obj._n_samples = state.get("n_samples", 0)
        obj._prev_brier = state.get("prev_brier", None)
        return obj
