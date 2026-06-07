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

    Fits LogisticRegression on [raw, raw², regime_score?, edge?] features.
    edge = abs(raw_prob - market_price) — how far our signal is from Kalshi
    pricing at entry. Two trades with the same raw_prob but different edges
    have very different Kelly implications; this feature captures that.

    Pass-through when n_samples < _MIN_SAMPLES (not enough data to fit reliably).
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._passthrough: bool = True
        self._n_samples: int = 0
        self._prev_brier: float | None = None
        self._regime_aware: bool = False
        self._edge_aware: bool = False

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def _build_X(
        self,
        raw: np.ndarray,
        regime_scores: np.ndarray | None,
        edges: np.ndarray | None,
    ) -> np.ndarray:
        cols = [raw, raw ** 2]
        if regime_scores is not None:
            cols.append(regime_scores)
        if edges is not None:
            cols.append(edges)
        return np.column_stack(cols)

    def fit(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        regimes: np.ndarray | None = None,
        edges: np.ndarray | None = None,
    ) -> "Calibrator":
        """
        raw_probs : model output probabilities (regime_prob for Phase 3c)
        outcomes  : binary win/loss labels
        regimes   : DeepSeek regime strings — enables regime-conditional calibration
        edges     : abs(raw_prob - market_price) at trade time — enables edge-aware
                    Kelly sizing. Two trades at the same raw_prob but different
                    edges will receive different calibrated outputs.
        """
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(raw_probs)
        self._n_samples = n

        use_regime = regimes is not None and len(regimes) == n
        use_edge   = edges   is not None and len(edges)   == n

        regime_scores = np.array([_encode_regime(r) for r in regimes], dtype=float) if use_regime else None
        edge_arr      = np.asarray(edges, dtype=float) if use_edge else None

        n_holdout = max(20, n // 5)
        n_train = n - n_holdout
        if n_train < _MIN_SAMPLES:
            self._passthrough = True
            return self

        raw_train,    y_train    = raw_probs[n_holdout:], outcomes[n_holdout:]
        raw_holdout,  y_holdout  = raw_probs[:n_holdout], outcomes[:n_holdout]
        reg_train   = regime_scores[n_holdout:] if use_regime else None
        reg_holdout = regime_scores[:n_holdout] if use_regime else None
        edg_train   = edge_arr[n_holdout:] if use_edge else None
        edg_holdout = edge_arr[:n_holdout] if use_edge else None

        X_train   = self._build_X(raw_train,   reg_train,   edg_train)
        X_holdout = self._build_X(raw_holdout, reg_holdout, edg_holdout)

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
            # Direction sanity guard: at a typical edge (0.15) in the most trusted
            # regime, calibrated output must not invert direction.
            guard_raw = np.array([0.75])
            guard_reg = np.array([_encode_regime("trending_up")]) if use_regime else None
            guard_edg = np.array([0.15]) if use_edge else None
            guard_X = self._build_X(guard_raw, guard_reg, guard_edg)
            guard_val = float(np.clip(new_model.predict_proba(guard_X)[:, 1], 0.0, 1.0)[0])
            direction_ok = guard_val >= 0.5
            if not direction_ok:
                logger.warning(
                    f"Calibrator direction guard: transform(0.75, trending_up, edge=0.15)"
                    f"={guard_val:.4f} < 0.5 — inverted signal detected. Forcing passthrough."
                )
            if direction_ok:
                self._model = new_model
                self._passthrough = False
                self._prev_brier = holdout_brier
                self._regime_aware = use_regime
                self._edge_aware   = use_edge
            else:
                self._model = prev_model
                self._passthrough = prev_passthrough
                if prev_passthrough:
                    self._prev_brier = passthrough_holdout_brier
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

    def transform(self, raw_prob: float, regime: str | None = None,
                  edge: float | None = None) -> float:
        """
        edge = abs(raw_prob - market_price) at inference time.
        When edge_aware and edge is None, defaults to 0.0 (no edge context).
        """
        if self._passthrough or self._model is None:
            return float(raw_prob)
        raw = np.array([raw_prob])
        reg = np.array([_encode_regime(regime)]) if self._regime_aware else None
        edg = np.array([edge if edge is not None else 0.0]) if self._edge_aware else None
        X = self._build_X(raw, reg, edg)
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
            "edge_aware": self._edge_aware,
        }, path)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibrator model not found: {path}")
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj._model = state.get("model", state.get("iso"))
        obj._passthrough = state["passthrough"]
        obj._n_samples = state.get("n_samples", 0)
        obj._prev_brier = state.get("prev_brier", None)
        obj._regime_aware = state.get("regime_aware", False)
        obj._edge_aware   = state.get("edge_aware",   False)
        return obj
