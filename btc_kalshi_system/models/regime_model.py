import os

import joblib
import numpy as np
import xgboost as xgb

_FEATURE_ORDER = [
    "funding_rate",
    "funding_rate_trend",
    "oi_delta_pct",
    "cvd_normalized",
    "basis_spread_pct",
    "brti_volatility_1h",
    "cvd_velocity",
    "cvd_acceleration",
    "brti_momentum_5min",
    "brti_momentum_15min",
    "candle_progress",
    "hour_sin",
    "hour_cos",
    "kalshi_implied_prob",
    "funding_window_proximity",
    "trend_slope_1h",
    "trend_r2_1h",
    "hourly_sr_proximity",
    "range_breakout_flag",
    "tape_speed_tpm",
    "large_print_direction",
    # Features 22-27: Deribit options + Kalshi spread (session 6)
    "atm_iv",
    "iv_rv_spread",
    "pcr_oi",
    "term_structure_slope",
    "skew_25d",
    "kalshi_spread_normalized",
]


class NotTrainedError(RuntimeError):
    """Raised when get_regime() is called before a model has been trained or loaded."""


class RegimeModel:
    """
    XGBoost binary classifier for BTC market regime.

    Returns prob_up, direction (0/1), and confidence (distance from 0.5).
    Training is stubbed — no labels exist yet. Load a saved model or train
    before calling get_regime().
    """

    def __init__(self) -> None:
        self._clf: xgb.XGBClassifier | None = None

    def get_regime(self, features: dict) -> dict:
        if self._clf is None:
            raise NotTrainedError(
                "RegimeModel has not been trained. Call train() or load() first."
            )
        X = np.array([[features[k] for k in _FEATURE_ORDER]])
        prob_up = float(self._clf.predict_proba(X)[0, 1])
        direction = int(prob_up >= 0.5)
        confidence = float(abs(prob_up - 0.5) * 2)  # 0 at boundary, 1 at extremes
        return {"prob_up": prob_up, "direction": direction, "confidence": confidence}

    def train(self, X: np.ndarray, y: np.ndarray, **xgb_kwargs) -> "RegimeModel":
        """
        Fit the XGBoost classifier. Extra keyword arguments are forwarded to
        XGBClassifier — most usefully `scale_pos_weight` for imbalanced labels
        (use `(y==0).sum() / (y==1).sum()`). Caller-supplied kwargs override the
        defaults below.
        """
        defaults = {
            "n_estimators": 100,
            "max_depth": 4,
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
    def load(cls, path: str) -> "RegimeModel":
        if not os.path.exists(path):
            raise FileNotFoundError(f"RegimeModel file not found: {path}")
        obj = cls.__new__(cls)
        obj._clf = joblib.load(path)
        return obj
