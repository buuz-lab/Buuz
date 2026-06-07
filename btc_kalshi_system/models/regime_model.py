import os

import joblib
import numpy as np
import xgboost as xgb

_FEATURE_ORDER = [
    # Market microstructure — no Kalshi features so regime model is independent
    # of the same Kalshi signal used in Gates 5 and 8. Circularity was: Kalshi →
    # regime_prob → combined signal → edge vs Kalshi price → Kalshi consensus block.
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
    "funding_window_proximity",
    "trend_slope_1h",
    "trend_r2_1h",
    "hourly_sr_proximity",
    "range_breakout_flag",
    "tape_speed_tpm",
    "large_print_direction",
    # Liquidity context (session 31) — current hourly volume vs 30-day average.
    # 1.0 = normal, <0.5 = thin (noisy), >2.0 = active. XGBoost handles NaN on
    # pre-session rows where the column was not yet logged.
    "volume_ratio_1h",
    # Deribit options (session 6) — independent of Kalshi
    "atm_iv",
    "iv_rv_spread",
    "pcr_oi",
    "term_structure_slope",
    "skew_25d",
    # 24h BTC price return context (session 11) — XGBoost handles NaN rows
    "btc_24h_return",
    # Kronos momentum meta-features (session 26) — logged from _cached_kronos at candle close;
    # NULL (→ NaN) when bootstrap loop hasn't fired yet; XGBoost treats NaN as missing.
    "kronos_raw_15min",
    "kronos_raw_5min",
    # Kalshi imbalance snapshot (session 35) — captured from WS at trade entry time;
    # None (→ NaN) when REST fallback is used. Independent of Kalshi implied_prob (Gate 5).
    "kalshi_open_imbalance",
    # Macro correlation features (session 35) — 8-day rolling BTC correlation with SPX/QQQ;
    # sourced from Redis via DerivativesFeed; default 0.0 when unavailable.
    "btc_spx_corr_8d",
    "btc_qqq_corr_8d",
    # T+30s market reaction (session 38) — Kalshi mid price drift in first ~30s of candle.
    # Positive = market repriced upward in the entry window; negative = downward.
    # Only populated for candles where kalshi_early_mid AND kalshi_open_mid are both logged.
    # Historical rows default to NaN (XGBoost missing-value handling).
    "kalshi_early_drift",
    # Session 39 — cascade momentum, cross-asset, order flow, options delta, LLM direction
    "liq_net_norm",
    "eth_direction_15min",
    "okx_spot_imbalance",
    "pcr_delta",
    "skew_delta",
    "deepseek_dir_prob",
    # Session 40 — microstructure divergence and directional trend context
    "cvd_price_divergence",
    "recent_up_fraction",
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
        # None entries (e.g. kronos_raw_15min before bootstrap loop fires) → NaN;
        # XGBoost treats NaN as missing values natively.
        X = np.array([[features[k] if features[k] is not None else float("nan") for k in _FEATURE_ORDER]])
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
