"""
Unit tests for SignalFusionEngine and TradingSignal.

All external dependencies (KronosEngine, Calibrator, RegimeModel,
DeepSeekContextParser, FeatureStore) are mocked — no network calls, no Redis,
no torch inference.
"""

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import config
from btc_kalshi_system.models.regime_model import NotTrainedError
from btc_kalshi_system.signal.fusion import (
    _BOOTSTRAP_SHRINK,
    _KRONOS_WEIGHT,
    _REGIME_WEIGHT,
    _REGIME_PAUSE_FLAG,
    SignalFusionEngine,
    TradingSignal,
)


# ── Test fixture helpers ───────────────────────────────────────────────────────

def _make_feature_store_mock():
    """Feature store mock with just enough data to avoid spurious stale=True."""
    import pandas as pd
    import numpy as np_
    feature_store = MagicMock()

    # Enough 5-min candles for brti_momentum and trend features
    prices = np_.linspace(95000, 95100, 15).tolist()
    idx = pd.date_range("2024-01-01", periods=15, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [0.0] * 15, "amount": [0.0] * 15,
    }, index=idx)

    # Enough 1h candles for sr_proximity AND btc_24h_return (needs >= 25)
    h_prices = np_.linspace(94000, 96000, 26).tolist()
    h_idx = pd.date_range("2024-01-01", periods=26, freq="1h", tz="UTC")
    df1h = pd.DataFrame({
        "open": h_prices, "high": h_prices, "low": h_prices, "close": h_prices,
        "volume": [0.0] * 26, "amount": [0.0] * 26,
    }, index=h_idx)

    # get_ohlcv("1h") should return df1h, ("5min") should return df5
    def ohlcv_side_effect(tf):
        return df1h if tf == "1h" else df5
    feature_store.get_ohlcv.side_effect = ohlcv_side_effect

    # CVD ring buffer — return 5 entries so cold-start stale is NOT triggered.
    # Use time.time() so the freshness check (> 360s) doesn't fire.
    import time as _time
    now = _time.time()
    feature_store._redis.zrange.return_value = [
        (b"0.1", now - 600),
        (b"0.2", now - 480),
        (b"0.3", now - 360),
        (b"0.4", now - 240),
        (b"0.5", now - 120),
    ]

    # Raw ticks
    feature_store.get_raw_ticks.return_value = None

    return feature_store


def _ds_result(
    regime: str = "trending_up",
    suppress: bool = False,
    suppress_reason: str | None = None,
) -> dict:
    return {
        "regime": regime,
        "confidence": 0.75,
        "suppress_trading": suppress,
        "suppress_reason": suppress_reason,
        "notes": "synthetic",
    }


def make_engine(
    kronos_raw: float = 0.65,
    kronos_cal: float = 0.65,
    regime_prob: float = 0.70,
    regime_direction: int = 1,
    deepseek_result: dict | None = None,
    raise_not_trained: bool = False,
) -> SignalFusionEngine:
    """Return a SignalFusionEngine with all I/O mocked."""
    if deepseek_result is None:
        deepseek_result = _ds_result()

    feature_store = _make_feature_store_mock()

    kronos_engine = MagicMock()
    kronos_engine.run_monte_carlo.return_value = kronos_raw

    # Phase 3c: calibrator transforms regime_prob (not k15_raw).
    # Passthrough behaviour: transform(x) = x — reflects calibrator before first training.
    calibrator = MagicMock()
    calibrator.transform.side_effect = lambda raw, regime=None, edge=None: raw

    regime_model = MagicMock()
    if raise_not_trained:
        regime_model.get_regime.side_effect = NotTrainedError("not trained")
    else:
        regime_model.get_regime.return_value = {
            "prob_up": regime_prob,
            "direction": regime_direction,
            "confidence": abs(regime_prob - 0.5) * 2,
        }

    deepseek_parser = MagicMock()
    deepseek_parser.get_current_context.return_value = deepseek_result

    return SignalFusionEngine(
        feature_store=feature_store,
        kronos_engine=kronos_engine,
        calibrator=calibrator,
        regime_model=regime_model,
        deepseek_parser=deepseek_parser,
    )


# ── Gate 1: suppress_trading ───────────────────────────────────────────────────

def test_gate1_suppress_returns_none():
    engine = make_engine(deepseek_result=_ds_result(suppress=True, suppress_reason="fomc"))
    assert engine.get_signal("5min", 76000.0) is None


def test_gate1_not_suppressed_proceeds():
    engine = make_engine()
    assert engine.get_signal("5min", 76000.0) is not None


def test_gate1_suppress_prevents_kronos_call():
    """KronosEngine must not be invoked when trading is suppressed."""
    engine = make_engine(deepseek_result=_ds_result(suppress=True))
    engine.get_signal("5min", 76000.0)
    engine._kronos.run_monte_carlo.assert_not_called()


# ── Gate 2: direction agreement ────────────────────────────────────────────────

def test_gate2_direction_mismatch_returns_none(monkeypatch):
    """With Gate 2 in enforce mode, direction disagreement must return None.
    (In shadow mode the disagreement is logged but the trade proceeds — see
    test_gate2_shadow_mode_does_not_block below.)"""
    monkeypatch.setattr(config, "REGIME_GATE2_ENFORCING", True)
    # Kronos cal=0.70 → direction=1 (up); regime direction=0 (down)
    engine = make_engine(kronos_cal=0.70, regime_prob=0.30, regime_direction=0)
    assert engine.get_signal("5min", 76000.0) is None


def test_gate2_both_up_returns_signal():
    engine = make_engine(kronos_cal=0.70, regime_prob=0.70, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.direction == 1


def test_gate2_both_down_returns_signal():
    engine = make_engine(kronos_raw=0.35, regime_prob=0.35, regime_direction=0)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.direction == 0


def test_gate2_boundary_kronos_exactly_half_is_down():
    # cal=0.5 → direction=1 (>= 0.5 is up in our convention)
    engine = make_engine(kronos_cal=0.5, regime_prob=0.5, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.direction == 1


# ── Combined probability formula (trained regime model) ───────────────────────

def test_combined_weighted_average():
    """combined = _KRONOS_WEIGHT * kronos_cal + _REGIME_WEIGHT * regime_prob"""
    engine = make_engine(kronos_cal=0.70, regime_prob=0.80, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    expected = _KRONOS_WEIGHT * 0.70 + _REGIME_WEIGHT * 0.80
    assert result.calibrated_prob == pytest.approx(expected)


def test_combined_varies_with_regime_weight():
    """Regime contributes 20% to the final signal."""
    engine_high = make_engine(kronos_cal=0.65, regime_prob=0.90, regime_direction=1)
    engine_low = make_engine(kronos_cal=0.65, regime_prob=0.55, regime_direction=1)
    assert engine_high.get_signal("5min", 76000.0).calibrated_prob > \
           engine_low.get_signal("5min", 76000.0).calibrated_prob


def test_high_uncertainty_shrinks_combined_toward_half():
    """high_uncertainty regime: combined = 0.5 + (base - 0.5) * 0.5"""
    engine = make_engine(
        kronos_cal=0.70,
        regime_prob=0.80,
        regime_direction=1,
        deepseek_result=_ds_result(regime="high_uncertainty"),
    )
    result = engine.get_signal("5min", 76000.0)
    base = _KRONOS_WEIGHT * 0.70 + _REGIME_WEIGHT * 0.80
    expected = 0.5 + (base - 0.5) * 0.5
    assert result.calibrated_prob == pytest.approx(expected)


def test_high_uncertainty_does_not_suppress():
    """high_uncertainty shrinks probability but does NOT suppress trading."""
    engine = make_engine(
        kronos_cal=0.70,
        regime_prob=0.70,
        regime_direction=1,
        deepseek_result=_ds_result(regime="high_uncertainty"),
    )
    assert engine.get_signal("5min", 76000.0) is not None


def test_trending_up_regime_no_shrinkage():
    engine = make_engine(kronos_cal=0.70, regime_prob=0.80, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    expected = _KRONOS_WEIGHT * 0.70 + _REGIME_WEIGHT * 0.80
    assert result.calibrated_prob == pytest.approx(expected)


# ── NotTrainedError fallback ───────────────────────────────────────────────────

def test_not_trained_uses_kronos_only_formula():
    """combined = 0.5 + (k15_raw - 0.5) * _BOOTSTRAP_SHRINK. Phase 3c: uses raw k15, not calibrated."""
    engine = make_engine(kronos_raw=0.70, raise_not_trained=True)
    result = engine.get_signal("5min", 76000.0)
    expected = 0.5 + (0.70 - 0.5) * _BOOTSTRAP_SHRINK
    assert result is not None
    assert result.calibrated_prob == pytest.approx(expected)


def test_not_trained_kronos_below_half():
    engine = make_engine(kronos_raw=0.30, raise_not_trained=True)
    result = engine.get_signal("5min", 76000.0)
    expected = 0.5 + (0.30 - 0.5) * _BOOTSTRAP_SHRINK
    assert result is not None
    assert result.calibrated_prob == pytest.approx(expected)


def test_not_trained_bypasses_gate2():
    """Gate 2 must not fire when regime model is untrained."""
    # Kronos=0.30 → direction=0 (down). If Gate 2 ran against any regime it would
    # require agreement — but with NotTrainedError it must be skipped entirely.
    engine = make_engine(kronos_cal=0.30, raise_not_trained=True)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None   # NOT None — gate bypassed


def test_not_trained_direction_follows_combined():
    engine_up = make_engine(kronos_raw=0.70, raise_not_trained=True)
    engine_dn = make_engine(kronos_raw=0.30, raise_not_trained=True)
    assert engine_up.get_signal("5min", 76000.0).direction == 1
    assert engine_dn.get_signal("5min", 76000.0).direction == 0


def test_not_trained_sentinel_fields():
    """regime_prob=nan and regime_direction=-1 when not trained."""
    engine = make_engine(kronos_cal=0.65, raise_not_trained=True)
    result = engine.get_signal("5min", 76000.0)
    assert math.isnan(result.regime_prob)
    assert result.regime_direction == -1


def test_bootstrap_ranging_applies_035_shrink():
    """In bootstrap + ranging, Kronos signal compressed to 0.35× toward 0.5."""
    engine = make_engine(
        kronos_raw=0.80,
        raise_not_trained=True,
        deepseek_result=_ds_result(regime="ranging"),
    )
    result = engine.get_signal("5min", 76000.0)
    assert result.calibrated_prob == pytest.approx(0.5 + (0.80 - 0.5) * 0.35)


def test_bootstrap_high_uncertainty_applies_050_shrink():
    """In bootstrap + high_uncertainty, Kronos signal compressed to 0.50× toward 0.5."""
    engine = make_engine(
        kronos_raw=0.80,
        raise_not_trained=True,
        deepseek_result=_ds_result(regime="high_uncertainty"),
    )
    result = engine.get_signal("5min", 76000.0)
    assert result.calibrated_prob == pytest.approx(0.5 + (0.80 - 0.5) * 0.50)


def test_bootstrap_trending_up_keeps_080_shrink():
    """In bootstrap + trending_up, full _BOOTSTRAP_SHRINK (0.80) is preserved."""
    engine = make_engine(
        kronos_raw=0.80,
        raise_not_trained=True,
        deepseek_result=_ds_result(regime="trending_up"),
    )
    result = engine.get_signal("5min", 76000.0)
    assert result.calibrated_prob == pytest.approx(0.5 + (0.80 - 0.5) * _BOOTSTRAP_SHRINK)


def test_bootstrap_unknown_regime_falls_back_to_080():
    """In bootstrap + unknown regime string, falls back to _BOOTSTRAP_SHRINK (0.80)."""
    engine = make_engine(
        kronos_raw=0.80,
        raise_not_trained=True,
        deepseek_result=_ds_result(regime="unknown_regime"),
    )
    result = engine.get_signal("5min", 76000.0)
    assert result.calibrated_prob == pytest.approx(0.5 + (0.80 - 0.5) * _BOOTSTRAP_SHRINK)


# ── TradingSignal field correctness ───────────────────────────────────────────

def test_signal_carries_raw_and_calibrated_kronos():
    """Phase 3c: kronos_calibrated = k15_raw (calibrator now operates on regime_prob)."""
    engine = make_engine(kronos_raw=0.60, regime_prob=0.70, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result.kronos_raw == pytest.approx(0.60)
    assert result.kronos_calibrated == pytest.approx(0.60)  # same as raw — calibrator not applied to k15


def test_signal_carries_regime_fields():
    engine = make_engine(kronos_cal=0.65, regime_prob=0.72, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result.regime_prob == pytest.approx(0.72)
    assert result.regime_direction == 1


def test_signal_carries_deepseek_regime():
    engine = make_engine(deepseek_result=_ds_result(regime="ranging"))
    result = engine.get_signal("5min", 76000.0)
    assert result.deepseek_regime == "ranging"


def test_signal_carries_timeframe_and_strike():
    engine = make_engine()
    result = engine.get_signal("15min", 82500.0)
    assert result.timeframe == "15min"
    assert result.strike == pytest.approx(82500.0)


def test_signal_timestamp_is_utc_datetime():
    engine = make_engine()
    result = engine.get_signal("5min", 76000.0)
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo is not None


# ── update_market_context ──────────────────────────────────────────────────────

def test_update_market_context_passed_to_deepseek():
    engine = make_engine()
    new_ctx = {"funding_rate": 0.02, "oi_delta": 0.05}
    engine.update_market_context(new_ctx)
    engine.get_signal("5min", 76000.0)
    engine._deepseek.get_current_context.assert_called_with(new_ctx)


def test_default_market_context_is_empty_dict():
    engine = make_engine()
    engine.get_signal("5min", 76000.0)
    engine._deepseek.get_current_context.assert_called_with({})


# ── Strike is passed to KronosEngine ──────────────────────────────────────────

def test_strike_forwarded_to_kronos():
    engine = make_engine()
    engine.get_signal("5min", 99999.0)
    engine._kronos.run_monte_carlo.assert_called_once_with(
        engine._store, threshold=99999.0
    )


# ── Regime feature snapshot persisted on the signal ───────────────────────────

def test_signal_carries_regime_features_dict():
    """The exact features fed to RegimeModel must travel back on the TradingSignal
    so they can be persisted in trades.db for future training."""
    engine = make_engine()
    engine.update_market_context({
        "funding_rate": 0.0001,
        "funding_rate_trend": -0.00002,
        "oi_delta_pct": 0.012,
        "cvd_normalized": 0.3,
        "basis_spread_pct": 0.0008,
        "kalshi_mid_cents": 55.0,
    })
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    # All keys must be present; kronos_raw_*, kalshi_open_imbalance may be None until first cycle fires
    from btc_kalshi_system.models.regime_model import _FEATURE_ORDER
    _NULLABLE_KEYS = {"kronos_raw_15min", "kronos_raw_5min", "kalshi_open_imbalance"}
    for key in _FEATURE_ORDER:
        assert key in result.regime_features, f"Missing key: {key}"
        if key not in _NULLABLE_KEYS:
            assert isinstance(result.regime_features[key], float), key
    assert result.regime_features["funding_rate"] == pytest.approx(0.0001)
    assert result.regime_features["cvd_normalized"] == pytest.approx(0.3)


def test_signal_features_stale_true_when_market_context_empty():
    """Empty market_context (Redis miss) must produce features_stale=True so
    training pipelines can filter out the row."""
    engine = make_engine()  # default ctx is {}
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.features_stale is True


def test_signal_features_stale_false_when_market_context_populated():
    engine = make_engine()
    engine.update_market_context({"funding_rate": 0.0001, "kalshi_mid_cents": 55.0})
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.features_stale is False


# ── Gate 2 soft-launch (shadow vs enforce) ────────────────────────────────────

def test_gate2_shadow_mode_does_not_block(monkeypatch):
    """When REGIME_GATE2_ENFORCING=False, Kronos/regime disagreements must NOT
    return None — the trade proceeds using regime_prob directly (no neutralization)."""
    monkeypatch.setattr(config, "REGIME_GATE2_ENFORCING", False)
    # Kronos up (0.70), regime down (0.30 / direction=0) — disagreement
    engine = make_engine(kronos_raw=0.70, regime_prob=0.30, regime_direction=0)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    # Regime v2 is the full signal — no neutralization, combined = regime_prob = 0.30
    assert result.calibrated_prob == pytest.approx(0.30)


def test_gate2_enforce_mode_blocks(monkeypatch):
    """When REGIME_GATE2_ENFORCING=True, disagreements return None as before."""
    monkeypatch.setattr(config, "REGIME_GATE2_ENFORCING", True)
    engine = make_engine(kronos_cal=0.70, regime_prob=0.30, regime_direction=0)
    assert engine.get_signal("5min", 76000.0) is None


def test_gate2_shadow_mode_does_not_affect_agreement(monkeypatch):
    """Sanity check: with the flag off and the directions agreeing, behavior is
    unchanged from the original test_gate2_both_up_returns_signal."""
    monkeypatch.setattr(config, "REGIME_GATE2_ENFORCING", False)
    engine = make_engine(kronos_cal=0.70, regime_prob=0.70, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.direction == 1


# ── Disagreement neutralization (Task 3, session 23) ─────────────────────────

def test_regime_signal_used_directly_on_disagreement_bullish_kronos():
    """k15_raw=0.70 (bullish), regime_prob=0.20 (bearish) → regime v2 wins, combined=0.20.
    No neutralization — regime already factored Kronos into its prediction."""
    engine = make_engine(kronos_raw=0.70, regime_prob=0.20, regime_direction=0)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.calibrated_prob == pytest.approx(0.20)
    assert result.regime_prob == pytest.approx(0.20)


def test_regime_signal_used_directly_on_disagreement_bearish_kronos():
    """k15_raw=0.30 (bearish), regime_prob=0.80 (bullish) → regime v2 wins, combined=0.80."""
    engine = make_engine(kronos_raw=0.30, regime_prob=0.80, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.calibrated_prob == pytest.approx(0.80)


def test_regime_signal_used_directly_on_agreement_bullish():
    """Both bullish — regime_prob used directly, same as disagreement case."""
    engine = make_engine(kronos_raw=0.70, regime_prob=0.65, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.calibrated_prob == pytest.approx(0.65)


def test_regime_signal_used_directly_on_agreement_bearish():
    """Both bearish — regime_prob used directly."""
    engine = make_engine(kronos_raw=0.35, regime_prob=0.20, regime_direction=0)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.calibrated_prob == pytest.approx(0.20)


# ── Regime pause flag ─────────────────────────────────────────────────────────

def test_pause_flag_reverts_to_bootstrap(tmp_path):
    """When regime_paused.flag exists, engine falls back to Kronos-only bootstrap signal."""
    pause_flag = tmp_path / "regime_paused.flag"
    pause_flag.touch()
    engine = make_engine(kronos_raw=0.70, regime_prob=0.80, regime_direction=1)
    with patch("btc_kalshi_system.signal.fusion._REGIME_PAUSE_FLAG", pause_flag):
        result = engine.get_signal("5min", 76000.0)
    assert result is not None
    # Falls back to bootstrap: combined = 0.5 + (k15_raw=0.70 - 0.5) * _BOOTSTRAP_SHRINK
    expected = 0.5 + (0.70 - 0.5) * _BOOTSTRAP_SHRINK
    assert result.calibrated_prob == pytest.approx(expected)


def test_pause_flag_absent_uses_regime_normally(tmp_path):
    """When no pause flag exists, regime model is used as normal."""
    pause_flag = tmp_path / "regime_paused.flag"  # does NOT exist
    engine = make_engine(kronos_cal=0.70, regime_prob=0.80, regime_direction=1)
    with patch("btc_kalshi_system.signal.fusion._REGIME_PAUSE_FLAG", pause_flag):
        result = engine.get_signal("5min", 76000.0)
    assert result is not None
    expected = _KRONOS_WEIGHT * 0.70 + _REGIME_WEIGHT * 0.80
    assert result.calibrated_prob == pytest.approx(expected)


# ── Kalshi imbalance cache ────────────────────────────────────────────────────

def test_set_kalshi_imbalance_updates_regime_features():
    """set_kalshi_imbalance() causes kalshi_open_imbalance to appear in _regime_features."""
    engine = make_engine()
    engine.set_kalshi_imbalance(0.42)
    features, _, _, _ = engine._regime_features()
    assert features.get("kalshi_open_imbalance") == 0.42


def test_set_kalshi_imbalance_none_passes_through():
    """None imbalance (REST fallback) is passed through as None."""
    engine = make_engine()
    engine.set_kalshi_imbalance(None)
    features, _, _, _ = engine._regime_features()
    assert features.get("kalshi_open_imbalance") is None


def test_kalshi_imbalance_defaults_to_none_before_set():
    """Before set_kalshi_imbalance() is called, value is None."""
    engine = make_engine()
    features, _, _, _ = engine._regime_features()
    assert features.get("kalshi_open_imbalance") is None
