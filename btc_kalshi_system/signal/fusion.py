"""
SignalFusionEngine — combines Kronos MC forecast, regime model, and DeepSeek
context into a single gated TradingSignal.

Gate 1 (DeepSeek): suppress_trading=True  → return None
Gate 2 (direction): Kronos ≠ regime       → log warning, and return None ONLY when
    config.REGIME_GATE2_ENFORCING=True. Skipped entirely if regime not trained.

Combined probability formula (when both models available):
    combined = 0.8 * kronos_calibrated + 0.2 * regime_prob   ← current (bootstrap)
    # At regime v2 deploy: flip to 0.2 * kronos_calibrated + 0.8 * regime_prob
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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from loguru import logger

import config
from btc_kalshi_system.data.feature_store import FeatureStore
from btc_kalshi_system.models.calibrator import Calibrator
from btc_kalshi_system.models.deepseek_parser import DeepSeekContextParser
from btc_kalshi_system.models.kronos_engine import KronosEngine
from btc_kalshi_system.models.regime_model import NotTrainedError, RegimeModel

_REGIME_PAUSE_FLAG = Path("models/regime_paused.flag")

_KRONOS_WEIGHT = 0.0   # Kronos is anti-correlated at 1-candle lag (Brier 0.37, Acc 46%).
# The apparent 70% same-candle accuracy was leakage (k15[N] vs direction[N] it was
# computed from). Regime v2 correctly fades Kronos — that IS the right behavior.
# Do NOT add explicit Kronos weight back unless 1-candle lag Brier improves to < 0.25.
_REGIME_WEIGHT = 1.0
_UNCERTAINTY_SHRINK = 0.5   # applied when DeepSeek signals high_uncertainty
_RANGING_SHRINK = 0.7       # applied when DeepSeek signals ranging (noisy, not untradeable)
_BOOTSTRAP_SHRINK = 0.8     # applied when RegimeModel is untrained (bootstrap phase)
# In bootstrap, DeepSeek regime still informs Kronos compression. Ranging/high_uncertainty
# have shown anti-correlated Kronos signals — shrink more aggressively to reduce Kelly
# damage without inverting direction. trending_up/down fall back to _BOOTSTRAP_SHRINK.
_BOOTSTRAP_REGIME_SHRINK: dict[str, float] = {
    "ranging":          0.35,
    "high_uncertainty": 0.50,
}


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
    # Snapshot of market context at signal-creation time (Kalshi spread, mid, OI, etc).
    # Used by Gate 12 (dynamic candle progress cap) and other market-aware gates.
    market_context: dict = field(default_factory=dict)
    # True if the upstream `regime:features` Redis key was missing/expired when
    # this signal was built. Stale rows are still tradeable in bootstrap mode
    # (regime features are unused), but they must be filtered out before training.
    features_stale: bool = False
    # True when Deribit options data was absent or an LKG fallback was used.
    # Rows with deribit_stale=True are excluded from the 27-feature retrain.
    # Independent from features_stale — do NOT combine them.
    deribit_stale: bool = False
    # True when OKX funding/OI used the zero fallback (_okx_partial), or when
    # regime:features expired and LKG was used (_lkg). Parallel to deribit_stale.
    # Independent from features_stale — do NOT combine them.
    okx_stale: bool = False


class SignalFusionEngine:
    def __init__(
        self,
        feature_store: FeatureStore,
        kronos_engine: KronosEngine,
        calibrator: Calibrator,
        regime_model: RegimeModel,
        deepseek_parser: DeepSeekContextParser,
        market_context: Optional[dict] = None,
        drift_monitor: Optional["CalibrationDriftMonitor"] = None,
    ) -> None:
        self._store = feature_store
        self._kronos = kronos_engine
        self._calibrator = calibrator
        self._regime = regime_model
        self._deepseek = deepseek_parser
        self._market_context: dict = market_context or {}
        self._drift_monitor = drift_monitor
        # Cached from the most recent fuse() call — None until first cycle runs.
        # Exposed via _regime_features() so the candle logger can log them.
        self._last_kronos_raw_15min: float | None = None
        self._last_kronos_raw_5min: float | None = None
        self._last_kalshi_open_imbalance: float | None = None
        self._last_kalshi_early_drift: float | None = None
        self._last_deepseek_dir_prob: float = 0.5   # 0.5 = no directional view yet

    def update_market_context(self, ctx: dict) -> None:
        self._market_context = ctx

    def update_kalshi_mid(self, mid_cents: float) -> None:
        self._market_context["kalshi_mid_cents"] = mid_cents

    def update_kalshi_spread(self, spread: float) -> None:
        self._market_context["kalshi_spread_normalized"] = spread

    def set_kalshi_imbalance(self, imbalance: float | None) -> None:
        self._last_kalshi_open_imbalance = imbalance

    def set_kalshi_early_drift(self, drift: float | None) -> None:
        self._last_kalshi_early_drift = drift

    def get_signal(
        self,
        timeframe: str,
        strike: float,
        kronos_raw: float | None = None,
        kronos_raw_15min: float | None = None,
    ) -> Optional[TradingSignal]:
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
        self._last_deepseek_dir_prob = float(ds.get("dir_prob_up", 0.5))

        if kronos_raw is None:
            # only used in tests — production always provides kronos_raw from _cached_kronos
            kronos_raw = self._kronos.run_monte_carlo(self._store, threshold=strike)
        self._last_kronos_raw_5min = kronos_raw
        self._last_kronos_raw_15min = kronos_raw_15min
        # Phase 3c: calibrator now trains on regime_prob, not k15_raw.
        # kronos_cal holds raw k15 (for Gate 11 / TradingSignal.kronos_calibrated).
        # combined is computed via calibrator.transform(regime_prob, edge=signal_edge) below.
        # In passthrough mode transform(regime_prob) = regime_prob — identical to pre-3c behaviour.
        kronos_cal = kronos_raw_15min if kronos_raw_15min is not None else kronos_raw
        kronos_direction = 1 if kronos_cal >= 0.5 else 0

        # Compute features ONCE per signal so the values we feed the regime model
        # match the values we persist in trades.db. This is the snapshot used both
        # at inference time and (later) as the training row for this trade.
        regime_features, features_stale, deribit_stale, okx_stale = self._regime_features()

        try:
            if _REGIME_PAUSE_FLAG.exists():
                raise NotTrainedError("regime model paused — drawdown protection active")
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

            _signal_edge   = abs(regime_prob - self._market_context.get("kalshi_mid_cents", 50.0) / 100.0)
            _k15_for_cal   = self._last_kronos_raw_15min if self._last_kronos_raw_15min is not None else 0.5
            _disagreement  = abs(regime_prob - _k15_for_cal)
            _cal_vol       = float(regime_features.get("brti_volatility_1h") or 0.0)
            _cal_spread    = float(regime_features.get("kalshi_spread_normalized") or 0.0)
            combined = self._calibrator.transform(
                regime_prob,
                regime=deepseek_regime,
                edge=_signal_edge,
                disagreement=_disagreement,
                volatility=_cal_vol,
                spread=_cal_spread,
            )
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
            base_shrink = _BOOTSTRAP_REGIME_SHRINK.get(deepseek_regime, _BOOTSTRAP_SHRINK)
            if self._drift_monitor is not None and self._drift_monitor.is_drifting():
                base_shrink = min(base_shrink, 0.4)
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
            market_context=self._market_context.copy(),
            features_stale=features_stale,
            deribit_stale=deribit_stale,
            okx_stale=okx_stale,
        )

    def get_features_snapshot(self) -> tuple[dict, bool, bool, float | None]:
        """
        Returns (features_dict, features_stale, deribit_stale, regime_prob).
        regime_prob is the model's prob_up on the current features, or None when
        not trained or paused. Safe to call from a background loop.
        """
        features, features_stale, deribit_stale, _okx_stale = self._regime_features()
        regime_prob: float | None = None
        if not _REGIME_PAUSE_FLAG.exists():
            try:
                regime_prob = self._regime.get_regime(features)["prob_up"]
            except NotTrainedError:
                pass
        return features, features_stale, deribit_stale, regime_prob

    def _regime_features(self) -> tuple[dict, bool, bool, bool]:
        """
        Build the 27-feature dict consumed by RegimeModel.get_regime() and the
        feature-store at training time.

        Returns (features, stale, deribit_stale, okx_stale).
        - stale=True when regime:features Redis key was missing/LKG used.
        - deribit_stale=True when options:features was absent or LKG used.
        Both flags are independent — do NOT combine them.

        We intentionally still return numeric values (0.0 fallback) even when
        stale, so XGBoost.predict_proba() doesn't blow up in the trained path.
        """
        ctx = self._market_context
        # stale=True when: (a) ctx is empty — Redis key was expired and no LKG
        # existed, OR (b) ctx carries _lkg=True — LKG fallback was used because
        # the primary key expired during an exchange outage. In both cases the
        # row must be excluded from RegimeModel training.
        stale = not ctx or ctx.get("_lkg", False) or ctx.get("_cvd_stale", False)

        # Read OHLCV once — reused by multiple features
        df5  = self._store.get_ohlcv("5min")
        df15 = self._store.get_ohlcv("15min")
        df1h = self._store.get_ohlcv("1h")

        # --- Features 1-6: existing ---
        if df5 is not None and len(df5) >= 12:
            vol = float(df5["close"].pct_change().tail(12).std())
        else:
            vol = 0.0

        funding_rate = float(ctx.get("funding_rate", 0.0))
        funding_rate_trend = float(ctx.get("funding_rate_trend", 0.0))
        oi_delta_pct = float(ctx.get("oi_delta_pct", 0.0))
        cvd_normalized = float(ctx.get("cvd_normalized", 0.0))
        basis_spread_pct = float(ctx.get("basis_spread_pct", 0.0))
        brti_volatility_1h = vol

        # --- Features 7-8: cvd_velocity, cvd_acceleration ---
        entries = self._store._redis.zrange("regime:cvd_history", 0, -1, withscores=True)
        if len(entries) < 5:
            cvd_velocity = 0.0
            cvd_acceleration = 0.0
            stale = True
        elif time.time() - entries[-1][1] > 450:
            # Most recent entry is > 1.875× the 240s refresh interval — feed has missed
            # at least one full cycle plus slack. Count check passes but timestamps are
            # stale, so velocity math would use wrong time windows. Mark stale.
            cvd_velocity = 0.0
            cvd_acceleration = 0.0
            stale = True
        else:
            now_ts = time.time()

            def _parse_cvd(member: bytes) -> float:
                # Member format: "timestamp:cvd_value" (new) or plain float str (old)
                s = member.decode() if isinstance(member, bytes) else str(member)
                return float(s.split(":", 1)[-1])

            def _closest(ents, target_ts):
                return _parse_cvd(min(ents, key=lambda e: abs(e[1] - target_ts))[0])

            cvd_now = _parse_cvd(entries[-1][0])
            cvd_5m_ago = _closest(entries, now_ts - 300)
            cvd_10m_ago = _closest(entries, now_ts - 600)
            cvd_velocity = (cvd_now - cvd_5m_ago) / 5.0
            cvd_velocity_10m = (cvd_now - cvd_10m_ago) / 10.0
            cvd_acceleration = cvd_velocity - cvd_velocity_10m

        # --- Features 9-10: brti_momentum (prior closed candle, non-leaky) ---
        # Use the prior CLOSED 15-min and 5-min candles so the feature is the same
        # at T=0 (training) and T=33% (deployment). The old formula used df5[-1]
        # which at logging time (T+10s) ≈ the just-closed candle's return — leakage.
        try:
            if df15 is not None and len(df15) >= 3 and df5 is not None and len(df5) >= 5:
                prior = df15.iloc[-2]
                brti_momentum_15min = float(prior["close"] / prior["open"] - 1) if prior["open"] != 0 else 0.0
                # Last completed 5-min candle before the current 15-min candle opened
                current_15_open = df15.index[-1]
                df5_prior = df5[df5.index < current_15_open]
                if len(df5_prior) >= 2:
                    brti_momentum_5min = float(df5_prior["close"].iloc[-1] / df5_prior["close"].iloc[-2] - 1)
                else:
                    brti_momentum_5min = 0.0
            elif df5 is not None and len(df5) >= 4:
                brti_momentum_5min  = float(df5["close"].iloc[-1] / df5["close"].iloc[-2] - 1)
                brti_momentum_15min = float(df5["close"].iloc[-1] / df5["close"].iloc[-4] - 1)
            else:
                brti_momentum_5min  = 0.0
                brti_momentum_15min = 0.0
                stale = True
        except Exception:
            brti_momentum_5min  = 0.0
            brti_momentum_15min = 0.0

        # --- Feature 11: candle_progress ---
        candle_progress = float((time.time() % 900) / 900)

        # --- Features 12-13: hour_sin, hour_cos ---
        now_utc = datetime.now(timezone.utc)
        hour_float = now_utc.hour + now_utc.minute / 60.0
        hour_sin = float(math.sin(2 * math.pi * hour_float / 24))
        hour_cos = float(math.cos(2 * math.pi * hour_float / 24))

        # --- Feature 14: kalshi_implied_prob ---
        mid_cents = self._market_context.get("kalshi_mid_cents")
        if mid_cents is None:
            kalshi_implied_prob = 0.5
            # kalshi_implied_prob is excluded from _FEATURE_ORDER (circularity fix,
            # 2026-06-01) so None at candle roll time is harmless — do not mark stale.
        else:
            kalshi_implied_prob = float(mid_cents) / 100.0

        # --- Feature 15: funding_window_proximity ---
        now_utc2 = datetime.now(timezone.utc)
        secs = now_utc2.hour * 3600 + now_utc2.minute * 60 + now_utc2.second
        funding_secs = [0, 8 * 3600, 16 * 3600]
        min_dist = min(min(abs(secs - f), 86400 - abs(secs - f)) for f in funding_secs)
        funding_window_proximity = float(1.0 - min(min_dist / (4 * 3600), 1.0))

        # --- Features 16-17: trend_slope_1h, trend_r2_1h ---
        if df5 is not None and len(df5) >= 4:
            closes = df5["close"].tail(12).values.astype(float)
            x = np.arange(len(closes), dtype=float)
            coeffs = np.polyfit(x, closes, 1)
            slope_raw = coeffs[0]
            mean_price = closes.mean()
            trend_slope_1h = float(slope_raw / mean_price) if mean_price > 0 else 0.0
            y_pred = np.polyval(coeffs, x)
            ss_res = float(np.sum((closes - y_pred) ** 2))
            ss_tot = float(np.sum((closes - mean_price) ** 2))
            trend_r2_1h = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        else:
            trend_slope_1h = 0.0
            trend_r2_1h = 0.0

        # --- Feature 18: hourly_sr_proximity (non-leaky) ---
        # Old formula used df5[-1]["close"] at T+10s ≈ close of the candle being
        # predicted. Fix: use the candle's OPEN price (df15[-2]["open"]) so the
        # feature captures "where was price at the start of this candle vs S/R?"
        if df1h is not None and len(df1h) >= 2 and df15 is not None and len(df15) >= 3:
            current_price = float(df15.iloc[-2]["open"])
            resistance = float(df1h["high"].tail(24).max())
            support = float(df1h["low"].tail(24).min())
            price_range = resistance - support
            if price_range > 1e-6:
                hourly_sr_proximity = float((current_price - support) / price_range)
                hourly_sr_proximity = max(0.0, min(1.0, hourly_sr_proximity))
            else:
                hourly_sr_proximity = 0.5
        else:
            hourly_sr_proximity = 0.5

        # --- Feature 19: range_breakout_flag (prior closed candle, non-leaky) ---
        # Old formula used df5[-1] at T+10s ≈ close of the candle being predicted —
        # leakage. Fix: did the last CLOSED 15-min candle (N-1) break above/below
        # the range of the candle before it (N-2)? Same value at T=0 and T=33%.
        try:
            if df15 is not None and len(df15) >= 4:
                prior  = df15.iloc[-2]   # N-1: last closed 15-min candle
                prior2 = df15.iloc[-3]   # N-2: the candle before that
                box_high  = float(prior2["high"])
                box_low   = float(prior2["low"])
                box_range = box_high - box_low
                curr_high = float(prior["high"])
                curr_low  = float(prior["low"])
                if box_range > 1e-6:
                    breakout_up   = max(0.0, curr_high - box_high) / box_range
                    breakout_down = max(0.0, box_low  - curr_low)  / box_range
                    range_breakout_flag = float(breakout_up - breakout_down)
                else:
                    range_breakout_flag = 0.0
            elif df5 is not None and len(df5) >= 5:
                box = df5.iloc[-5:-2]
                box_high = float(box["high"].max())
                box_low  = float(box["low"].min())
                box_range = box_high - box_low
                curr_high = float(df5["high"].iloc[-1])
                curr_low  = float(df5["low"].iloc[-1])
                if box_range > 1e-6:
                    breakout_up   = max(0.0, curr_high - box_high) / box_range
                    breakout_down = max(0.0, box_low  - curr_low)  / box_range
                    range_breakout_flag = float(breakout_up - breakout_down)
                else:
                    range_breakout_flag = 0.0
            else:
                range_breakout_flag = 0.0
        except Exception:
            range_breakout_flag = 0.0

        # --- Feature 20: tape_speed_tpm ---
        ticks = self._store.get_raw_ticks(60)
        raw_tpm = float(len(ticks)) if ticks is not None else 0.0
        tape_speed_tpm = raw_tpm / 100.0

        # --- Feature 21: large_print_direction ---
        large_print_direction = float(ctx.get("large_print_direction", 0.0))

        # --- Feature 29: volume_ratio_1h (liquidity context) ---
        # 1.0 = normal volume; <0.5 = thin market; >2.0 = high activity.
        # Default 1.0 (neutral) when unavailable rather than 0.0 (extreme thin).
        volume_ratio_1h = float(ctx.get("volume_ratio_1h") or 1.0)

        # --- Feature 28: btc_24h_return ---
        if df1h is not None and len(df1h) >= 25:
            btc_24h_return = float(df1h["close"].iloc[-1] / df1h["close"].iloc[-25] - 1)
        else:
            btc_24h_return = 0.0
            stale = True

        # --- Features 22-27: Deribit options + Kalshi spread ---
        atm_iv = float(ctx.get("atm_iv") or 0.0)
        iv_rv_spread = float(ctx.get("iv_rv_spread") or 0.0)
        pcr_oi = float(ctx.get("pcr_oi") or 1.0)          # 1.0 not 0.0 — neutral ratio
        term_structure_slope = float(ctx.get("term_structure_slope") or 0.0)
        skew_25d = float(ctx.get("skew_25d") or 0.0)
        kalshi_spread_normalized = float(self._market_context.get("kalshi_spread_normalized") or 0.0)

        # deribit_stale=True when: (a) options:features absent (Deribit down and LKG expired),
        # OR (b) LKG was used (_deribit_lkg=True in ctx). Independent from stale.
        deribit_stale = (
            ctx.get("atm_iv") is None
            or ctx.get("_deribit_lkg", False)
        )

        # okx_stale=True when: (a) regime:features expired and LKG was used,
        # OR (b) OKX funding/OI fetch failed with no Coinglass key (_okx_partial).
        okx_stale = (not ctx) or ctx.get("_lkg", False) or ctx.get("_okx_partial", False)

        # kalshi_implied_prob and kalshi_spread_normalized intentionally excluded —
        # regime model must be independent of Kalshi to avoid circularity with Gates 5/8.
        features = {
            "funding_rate":             funding_rate,
            "funding_rate_trend":       funding_rate_trend,
            "oi_delta_pct":             oi_delta_pct,
            "cvd_normalized":           cvd_normalized,
            "basis_spread_pct":         basis_spread_pct,
            "brti_volatility_1h":       brti_volatility_1h,
            "cvd_velocity":             cvd_velocity,
            "cvd_acceleration":         cvd_acceleration,
            "brti_momentum_5min":       brti_momentum_5min,
            "brti_momentum_15min":      brti_momentum_15min,
            "candle_progress":          candle_progress,
            "hour_sin":                 hour_sin,
            "hour_cos":                 hour_cos,
            "funding_window_proximity": funding_window_proximity,
            "trend_slope_1h":           trend_slope_1h,
            "trend_r2_1h":              trend_r2_1h,
            "hourly_sr_proximity":      hourly_sr_proximity,
            "range_breakout_flag":      range_breakout_flag,
            "tape_speed_tpm":           tape_speed_tpm,
            "large_print_direction":    large_print_direction,
            "volume_ratio_1h":          volume_ratio_1h,
            "atm_iv":                   atm_iv,
            "iv_rv_spread":             iv_rv_spread,
            "pcr_oi":                   pcr_oi,
            "term_structure_slope":     term_structure_slope,
            "skew_25d":                 skew_25d,
            "btc_24h_return":           btc_24h_return,
            # Kronos meta-features — None until first trading cycle fires (bootstrap).
            # XGBoost treats None→NaN as missing; candle logger stores None as SQL NULL.
            "kronos_raw_15min":         self._last_kronos_raw_15min,
            "kronos_raw_5min":          self._last_kronos_raw_5min,
            "kalshi_open_imbalance":    self._last_kalshi_open_imbalance,
            "btc_spx_corr_8d":          float(ctx.get("btc_spx_corr_8d") or 0.0),
            "btc_qqq_corr_8d":          float(ctx.get("btc_qqq_corr_8d") or 0.0),
            "kalshi_early_drift":       self._last_kalshi_early_drift,
            "liq_net_norm":          float(ctx.get("liq_net_norm") or 0.0),
            "eth_direction_15min":   float(ctx.get("eth_direction_15min") if ctx.get("eth_direction_15min") is not None else 0.5),
            "okx_spot_imbalance":    float(ctx.get("okx_spot_imbalance") or 0.0),
            "pcr_delta":             float(ctx.get("pcr_delta") or 0.0),
            "skew_delta":            float(ctx.get("skew_delta") or 0.0),
            "deepseek_dir_prob":     self._last_deepseek_dir_prob,
        }
        return features, stale, deribit_stale, okx_stale
