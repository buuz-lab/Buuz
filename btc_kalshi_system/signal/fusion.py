"""
SignalFusionEngine — combines Kronos MC forecast, regime model, and DeepSeek
context into a single gated TradingSignal.

Gate 1 (DeepSeek): suppress_trading=True  → return None
Gate 2 (direction): Kronos ≠ regime       → return None (skipped if regime not trained)

Combined probability formula (when both models available):
    combined = 0.6 * kronos_calibrated + 0.4 * regime_prob
    if deepseek_regime == "high_uncertainty":
        combined = 0.5 + (combined - 0.5) * 0.5

When RegimeModel raises NotTrainedError (regime model not yet trained):
    combined = 0.5 + (kronos_calibrated - 0.5) * 0.5
    Gate 2 is bypassed — trading is allowed with conservative Kronos-only signal.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from btc_kalshi_system.data.feature_store import FeatureStore
from btc_kalshi_system.models.calibrator import Calibrator
from btc_kalshi_system.models.deepseek_parser import DeepSeekContextParser
from btc_kalshi_system.models.kronos_engine import KronosEngine
from btc_kalshi_system.models.regime_model import NotTrainedError, RegimeModel

_KRONOS_WEIGHT = 0.6
_REGIME_WEIGHT = 0.4
_UNCERTAINTY_SHRINK = 0.5


@dataclass
class TradingSignal:
    direction: int           # 1 = up/long, 0 = down/short
    calibrated_prob: float   # final combined probability
    kronos_raw: float        # raw Kronos MC P(close > strike)
    kronos_calibrated: float # after isotonic calibration
    regime_prob: float       # prob_up from RegimeModel (nan if not trained)
    regime_direction: int    # 0/1 from RegimeModel (-1 if not trained)
    deepseek_regime: str     # regime label from DeepSeek
    timeframe: str
    strike: float
    timestamp: datetime


class SignalFusionEngine:
    def __init__(
        self,
        feature_store: FeatureStore,
        kronos_engine: KronosEngine,
        calibrator: Calibrator,
        regime_model: RegimeModel,
        deepseek_parser: DeepSeekContextParser,
        market_context: Optional[dict] = None,
    ) -> None:
        self._store = feature_store
        self._kronos = kronos_engine
        self._calibrator = calibrator
        self._regime = regime_model
        self._deepseek = deepseek_parser
        self._market_context: dict = market_context or {}

    def update_market_context(self, ctx: dict) -> None:
        self._market_context = ctx

    def get_signal(self, timeframe: str, strike: float) -> Optional[TradingSignal]:
        ds = self._deepseek.get_current_context(self._market_context)

        # Gate 1: DeepSeek says suppress
        if ds["suppress_trading"]:
            return None

        deepseek_regime = ds["regime"]

        # For up/down markets, strike = BTC price at market open, so this computes
        # P(predicted_close > open_price) = P(price goes up) — exactly what we want.
        kronos_raw = self._kronos.run_monte_carlo(self._store, threshold=strike)
        kronos_cal = self._calibrator.transform(kronos_raw)
        kronos_direction = 1 if kronos_cal >= 0.5 else 0

        try:
            regime_result = self._regime.get_regime(self._regime_features())
            regime_prob = regime_result["prob_up"]
            regime_direction = regime_result["direction"]

            # Gate 2: Kronos and regime must agree
            if kronos_direction != regime_direction:
                return None

            combined = _KRONOS_WEIGHT * kronos_cal + _REGIME_WEIGHT * regime_prob
            if deepseek_regime == "high_uncertainty":
                combined = 0.5 + (combined - 0.5) * _UNCERTAINTY_SHRINK

        except NotTrainedError:
            # Regime model not yet trained — Kronos-only with conservative shrink
            regime_prob = math.nan
            regime_direction = -1
            combined = 0.5 + (kronos_cal - 0.5) * _UNCERTAINTY_SHRINK

        direction = 1 if combined >= 0.5 else 0

        return TradingSignal(
            direction=direction,
            calibrated_prob=combined,
            kronos_raw=kronos_raw,
            kronos_calibrated=kronos_cal,
            regime_prob=regime_prob,
            regime_direction=regime_direction,
            deepseek_regime=deepseek_regime,
            timeframe=timeframe,
            strike=strike,
            timestamp=datetime.now(timezone.utc),
        )

    def _regime_features(self) -> dict:
        ctx = self._market_context
        df = self._store.get_ohlcv("5min")
        if df is not None and len(df) >= 12:
            vol = float(df["close"].pct_change().tail(12).std())
        else:
            vol = 0.0
        return {
            "funding_rate": float(ctx.get("funding_rate", 0.0)),
            "funding_rate_trend": float(ctx.get("funding_rate_trend", 0.0)),
            "oi_delta_pct": float(ctx.get("oi_delta_pct", 0.0)),
            "cvd_normalized": float(ctx.get("cvd_normalized", 0.0)),
            "basis_spread_pct": float(ctx.get("basis_spread_pct", 0.0)),
            "brti_volatility_1h": vol,
        }
