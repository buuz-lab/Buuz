import os

import joblib
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

_MIN_SAMPLES = 300


class Calibrator:
    """
    Quadratic-logistic probability calibrator.

    Fits LogisticRegression on [raw, raw²] features so it can learn both
    monotone and inverted-U relationships (e.g. k15_raw > 0.8 → P(up) < 0.5).

    Pass-through when n_samples < _MIN_SAMPLES (not enough data to fit reliably).
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
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

        prev_model = self._model
        prev_passthrough = self._passthrough

        self._passthrough = False
        X = np.column_stack([raw_probs, raw_probs ** 2])
        new_model = LogisticRegression(max_iter=1000)
        new_model.fit(X, outcomes)
        self._model = new_model

        new_brier = self.brier_score(raw_probs, outcomes)
        if self._prev_brier is not None and new_brier > self._prev_brier:
            logger.warning(
                f"Calibrator: new Brier {new_brier:.4f} > previous {self._prev_brier:.4f} — reverting"
            )
            self._model = prev_model
            self._passthrough = prev_passthrough
        else:
            self._prev_brier = new_brier

        return self

    def transform(self, raw_prob: float) -> float:
        if self._passthrough or self._model is None:
            return float(raw_prob)
        X = np.array([[raw_prob, raw_prob ** 2]])
        return float(np.clip(self._model.predict_proba(X)[0, 1], 0.0, 1.0))

    def brier_score(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> float:
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        calibrated = np.array([self.transform(float(p)) for p in raw_probs])
        return float(np.mean((calibrated - outcomes) ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "model": self._model,
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
        obj._model = state.get("model", state.get("iso"))  # backward compat with old isotonic saves
        obj._passthrough = state["passthrough"]
        obj._n_samples = state.get("n_samples", 0)
        obj._prev_brier = state.get("prev_brier", None)
        return obj
