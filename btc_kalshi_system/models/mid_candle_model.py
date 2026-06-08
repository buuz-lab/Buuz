import os

import joblib
import numpy as np
import xgboost as xgb

# Features captured at 40-60% candle progress and written to candle_features.
# k5_at_midcandle  — most-recent cached k5 (computed at 33% or 66% 5-min mark).
# k15_at_midcandle — prior-close k15 (regime anchor; only computed once per candle open).
# Their delta encodes in-candle momentum divergence from the prior trend.
_MID_CANDLE_FEATURES = [
    "cvd_since_open",           # CVD of in-candle ticks only (not full 15-min window)
    "cvd_rate",                 # cvd_since_open / elapsed_fraction — detects stalling
    "tick_count_since_open",    # trade activity since candle open
    "brti_drift_since_open",    # BRTI price change from candle open to snapshot
    "brti_velocity",            # brti_drift / elapsed_fraction
    "kalshi_drift_cents",       # Kalshi mid vs open price in cents
    "kalshi_velocity",          # kalshi_drift_cents / elapsed_fraction
    "cvd_brti_divergence",      # +1 aligned, -1 diverging (CVD vs BRTI direction)
    "kalshi_brti_alignment",    # +1 aligned, -1 diverging (Kalshi vs BRTI direction)
    "k5_at_midcandle",          # k5 Kronos prob at snapshot time
    "k15_at_midcandle",         # k15 Kronos prob at snapshot time (prior-close anchor)
    "k5_k15_delta_at_midcandle",# k5 - k15: positive = bullish divergence from prior trend
    "spread_change",            # Kalshi spread widening vs open (positive = illiquid)
    "oi_delta_at_midcandle",    # OI expanding/contracting at snapshot time
    "kalshi_mid_candle_spread", # Absolute spread at snapshot (liquidity context)
    "kalshi_mid_candle_progress", # Actual progress when snapshot taken (40-60%)
]


class NotTrainedError(RuntimeError):
    """Raised when predict() is called before a model has been trained or loaded."""


class MidCandleModel:
    """XGBoost binary classifier trained on mid-candle (40-60% progress) features.

    Predicts prob_up for the SAME candle — no 1-candle lag. At 40-60% progress
    the label (btc_direction) is not yet resolved, so features are genuinely
    predictive rather than leaky.

    Key difference from RegimeModel: trained on in-flight microstructure
    (CVD rate, BRTI velocity, k5/k15 divergence) rather than prior-close
    macrostructure. Catches signals the Kalshi market hasn't priced yet.
    """

    def __init__(self) -> None:
        self._clf: xgb.XGBClassifier | None = None

    def predict(self, snapshot: dict) -> dict:
        """Score a mid-candle snapshot dict.

        Returns {"prob_up": float, "direction": int, "confidence": float}.
        snapshot keys must include _MID_CANDLE_FEATURES; missing/None → NaN
        (XGBoost handles NaN natively as missing).
        """
        if self._clf is None:
            raise NotTrainedError("MidCandleModel not trained. Call train() or load() first.")
        X = np.array([[
            snapshot.get(k) if snapshot.get(k) is not None else float("nan")
            for k in _MID_CANDLE_FEATURES
        ]])
        prob_up = float(self._clf.predict_proba(X)[0, 1])
        direction = int(prob_up >= 0.5)
        confidence = float(abs(prob_up - 0.5) * 2)
        return {"prob_up": prob_up, "direction": direction, "confidence": confidence}

    def train(self, X: np.ndarray, y: np.ndarray, **xgb_kwargs) -> "MidCandleModel":
        """Fit XGBoost. Pass scale_pos_weight when classes are imbalanced."""
        defaults = {
            "n_estimators": 100,
            "max_depth": 3,        # shallower than regime model — fewer features, less data
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        defaults.update(xgb_kwargs)
        self._clf = xgb.XGBClassifier(**defaults)
        self._clf.fit(X, y)
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str) -> "MidCandleModel":
        if not os.path.exists(path):
            raise FileNotFoundError(f"MidCandleModel file not found: {path}")
        obj = cls.__new__(cls)
        obj._clf = joblib.load(path)
        return obj
