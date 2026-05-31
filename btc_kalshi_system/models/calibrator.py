import os

import joblib
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

_MIN_SAMPLES = 100

_REGIME_ENCODING: dict[str, float] = {
    "trending_up":      1.0,
    "trending_down":   -1.0,
    "ranging":          0.3,   # 53.7% historical WR — mildly trustworthy
    "high_uncertainty": -0.3,  # 45.2% historical WR — mildly penalised
}


def _encode_regime(regime: str | None) -> float:
    return _REGIME_ENCODING.get(regime or "", 0.0)


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
        self._regime_aware: bool = False

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def fit(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        regimes: np.ndarray | None = None,
    ) -> "Calibrator":
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(raw_probs)
        self._n_samples = n

        use_regime = regimes is not None and len(regimes) == n
        if use_regime:
            regime_scores = np.array([_encode_regime(r) for r in regimes], dtype=float)
        else:
            regime_scores = None

        # Holdout split: newest 20% (min 20 rows) as unseen evaluation set.
        # Data is expected ordered newest-first (ORDER BY timestamp DESC).
        # Train on older rows; gate deployment on holdout Brier vs passthrough.
        n_holdout = max(20, n // 5)
        n_train = n - n_holdout
        if n_train < _MIN_SAMPLES:
            self._passthrough = True
            return self

        raw_train, y_train = raw_probs[n_holdout:], outcomes[n_holdout:]
        raw_holdout, y_holdout = raw_probs[:n_holdout], outcomes[:n_holdout]

        if use_regime:
            reg_train = regime_scores[n_holdout:]
            reg_holdout = regime_scores[:n_holdout]
            X_train = np.column_stack([raw_train, raw_train ** 2, reg_train])
            X_holdout = np.column_stack([raw_holdout, raw_holdout ** 2, reg_holdout])
        else:
            X_train = np.column_stack([raw_train, raw_train ** 2])
            X_holdout = np.column_stack([raw_holdout, raw_holdout ** 2])

        new_model = LogisticRegression(max_iter=1000)
        new_model.fit(X_train, y_train)
        holdout_preds = np.clip(new_model.predict_proba(X_holdout)[:, 1], 0.0, 1.0)
        holdout_brier = float(np.mean((holdout_preds - y_holdout) ** 2))
        passthrough_holdout_brier = float(np.mean((raw_holdout - y_holdout) ** 2))

        prev_model = self._model
        prev_passthrough = self._passthrough
        beats_passthrough = holdout_brier < passthrough_holdout_brier
        beats_prev = self._prev_brier is None or holdout_brier < self._prev_brier

        if beats_passthrough and beats_prev:
            self._model = new_model
            self._passthrough = False
            self._prev_brier = holdout_brier
            self._regime_aware = use_regime
        else:
            logger.warning(
                f"Calibrator: holdout Brier {holdout_brier:.4f} vs passthrough "
                f"{passthrough_holdout_brier:.4f}"
                + (f", prev {self._prev_brier:.4f}" if self._prev_brier is not None else "")
                + " — reverting"
            )
            self._model = prev_model
            self._passthrough = prev_passthrough
            if prev_passthrough:
                self._prev_brier = passthrough_holdout_brier

        return self

    def transform(self, raw_prob: float, regime: str | None = None) -> float:
        if self._passthrough or self._model is None:
            return float(raw_prob)
        if self._regime_aware:
            X = np.array([[raw_prob, raw_prob ** 2, _encode_regime(regime)]])
        else:
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
            "regime_aware": self._regime_aware,
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
        obj._regime_aware = state.get("regime_aware", False)
        return obj
