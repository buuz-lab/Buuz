"""
SignalFusionEngine — combines Kronos MC forecast, regime model, and DeepSeek
context into a single gated TradingSignal.

Gate 1 (DeepSeek): suppress_trading=True  → return None
Gate 2 (direction): Kronos ≠ regime       → log warning, and return None ONLY when
    config.REGIME_GATE2_ENFORCING=True. Skipped entirely if regime not trained.

Combined probability formula (when both models available):
    combined = 0.6 * kronos_calibrated + 0.4 * regime_prob
    if deepseek_regime == "high_uncertainty":
        combined = 0.5 + (combined - 0.5) * 0.5   # 50% shrink
    elif deepseek_regime == "ranging":
        combined = 0.5 + (combined - 0.5) * 0.7   # 30% shrink — noisy but tradeable

When RegimeModel raises NotTrainedError (regime model not yet trained):
    combined = 0.5 + (kronos_calibrated - 0.5) * _BOOTSTRAP_SHRINK  (0.8, not 0.5)
    Gate 2 is bypassed — trading is allowed with conservative Kronos-only signal.

Note: _BOOTSTRAP_SHRINK (0.8) is intentionally lighter than _UNCERTAINTY_SHRINK (0.5).
During bootstrap the regime model is simply untrained — that is different from
DeepSeek signalling genuine high uncertainty. A 50% shrink was preventing Gate 5
from passing during bootstrap, stalling trade accumulation indefinitely.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

import config
from btc_kalshi_system.data.feature_store import FeatureStore
from btc_kalshi_system.models.calibrator import Calibrator
from btc_kalshi_system.models.deepseek_parser import DeepSeekContextParser
from btc_kalshi_system.models.kronos_engine import KronosEngine
from btc_kalshi_system.models.regime_model import NotTrainedError, RegimeModel

_KRONOS_WEIGHT = 0.6
_REGIME_WEIGHT = 0.4
_UNCERTAINTY_SHRINK = 0.5   # applied when DeepSeek signals high_uncertainty
_RANGING_SHRINK = 0.7       # applied when DeepSeek signals ranging (noisy, not untradeable)
_BOOTSTRAP_SHRINK = 0.8     # applied when RegimeModel is untrained (bootstrap phase)


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
    # Snapshot of the six regime features at signal-creation time. These are the
    # exact values that were (or would be) fed to RegimeModel.get_regime(); they
    # are persisted at trade-write time and form the training X matrix later.
    regime_features: dict = field(default_factory=dict)
    # True if the upstream `regime:features` Redis key was missing/expired when
    # this signal was built. Stale rows are still tradeable in bootstrap mode
    # (regime features are unused), but they must be filtered out before training.
    features_stale: bool = False


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
        logger.debug(
            f"DeepSeek context: suppress={ds['suppress_trading']} "
            f"regime={ds['regime']} confidence={ds.get('confidence', '?')} "
            f"reason={ds.get('suppress_reason')} notes={ds.get('notes', '')[:80]}"
        )
        if ds["suppress_trading"]:
            logger.warning(
                f"Gate 1 (DeepSeek suppress): trading halted — "
                f"regime={ds['regime']} reason={ds.get('suppress_reason')} "
                f"notes={ds.get('notes', '')}"
            )
            return None

        deepseek_regime = ds["regime"]

        # For KXBTC15M up/down markets, strike = close of the last completed 15-min
        # BRTI candle (i.e. the BRTI price at the open of the current 15-min window).
        # This computes P(predicted_5min_close > last_15min_close) ≈ P(market resolves yes).
        # The caller (_extract_strike in main.py) is responsible for supplying the
        # correct reference price — do not substitute spot price or 5-min close here.
        kronos_raw = self._kronos.run_monte_carlo(self._store, threshold=strike)
        kronos_cal = self._calibrator.transform(kronos_raw)
        kronos_direction = 1 if kronos_cal >= 0.5 else 0

        # Compute features ONCE per signal so the values we feed the regime model
        # match the values we persist in trades.db. This is the snapshot used both
        # at inference time and (later) as the training row for this trade.
        regime_features, features_stale = self._regime_features()

        try:
            regime_result = self._regime.get_regime(regime_features)
            regime_prob = regime_result["prob_up"]
            regime_direction = regime_result["direction"]

            # Gate 2: Kronos and regime must agree.
            # In shadow mode (config.REGIME_GATE2_ENFORCING=False) we log the
            # disagreement but continue trading. This lets a freshly trained
            # model run alongside Kronos for ~50 trades so the disagreement rate
            # and regime confidence distribution can be observed before letting
            # the gate block live trades. Flip to True once validated.
            if kronos_direction != regime_direction:
                logger.warning(
                    f"Gate 2 disagreement: kronos_direction={kronos_direction} "
                    f"regime_direction={regime_direction} kronos_cal={kronos_cal:.3f} "
                    f"regime_prob={regime_prob:.3f} regime_confidence={regime_result.get('confidence', 0):.3f} "
                    f"enforcing={config.REGIME_GATE2_ENFORCING}"
                )
                if config.REGIME_GATE2_ENFORCING:
                    return None

            combined = _KRONOS_WEIGHT * kronos_cal + _REGIME_WEIGHT * regime_prob
            if deepseek_regime == "high_uncertainty":
                combined = 0.5 + (combined - 0.5) * _UNCERTAINTY_SHRINK
            elif deepseek_regime == "ranging":
                combined = 0.5 + (combined - 0.5) * _RANGING_SHRINK

        except NotTrainedError:
            # Regime model not yet trained — Kronos-only with a lighter bootstrap shrink.
            # Use _BOOTSTRAP_SHRINK (0.8) here, NOT _UNCERTAINTY_SHRINK (0.5).
            # The regime being untrained is a data-scarcity issue, not a signal of
            # high market uncertainty. Using 0.5 shrink compressed signals so much that
            # Gate 5 almost never passed during bootstrap, creating a deadlock where
            # no paper trades were placed and the calibrator could never train.
            regime_prob = math.nan
            regime_direction = -1
            base_shrink = _BOOTSTRAP_SHRINK
            if deepseek_regime == "high_uncertainty":
                base_shrink = _UNCERTAINTY_SHRINK
            elif deepseek_regime == "ranging":
                base_shrink = _RANGING_SHRINK
            combined = 0.5 + (kronos_cal - 0.5) * base_shrink

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
            regime_features=regime_features,
            features_stale=features_stale,
        )

    def _regime_features(self) -> tuple[dict, bool]:
        """
        Build the six-feature dict consumed by RegimeModel.get_regime() and the
        feature-store at training time.

        Returns (features, stale). `stale=True` means the upstream Binance-derived
        regime context (funding/OI/CVD/basis) was missing or expired in Redis when
        this snapshot was built — five of the six features fell back to 0.0 and
        the row should be excluded from any future training X matrix.

        We intentionally still return numeric values (0.0 fallback) even when
        stale, so XGBoost.predict_proba() doesn't blow up in the trained path.
        The caller is expected to consult `stale` to decide whether to bail out.
        """
        ctx = self._market_context
        stale = not ctx  # empty dict ⇒ Redis read failed or key expired

        # brti_volatility_1h is computed locally from the OHLCV store and is
        # independent of the Redis regime:features key. It can be valid even
        # when ctx is empty, so we don't include it in the staleness check.
        df = self._store.get_ohlcv("5min")
        if df is not None and len(df) >= 12:
            vol = float(df["close"].pct_change().tail(12).std())
        else:
            vol = 0.0

        features = {
            "funding_rate": float(ctx.get("funding_rate", 0.0)),
            "funding_rate_trend": float(ctx.get("funding_rate_trend", 0.0)),
            "oi_delta_pct": float(ctx.get("oi_delta_pct", 0.0)),
            "cvd_normalized": float(ctx.get("cvd_normalized", 0.0)),
            "basis_spread_pct": float(ctx.get("basis_spread_pct", 0.0)),
            "brti_volatility_1h": vol,
        }
        return features, stale
