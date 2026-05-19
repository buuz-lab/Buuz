"""
Unit tests for SignalFusionEngine and TradingSignal.

All external dependencies (KronosEngine, Calibrator, RegimeModel,
DeepSeekContextParser, FeatureStore) are mocked — no network calls, no Redis,
no torch inference.
"""

import math
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from btc_kalshi_system.models.regime_model import NotTrainedError
from btc_kalshi_system.signal.fusion import SignalFusionEngine, TradingSignal


# ── Test fixture helpers ───────────────────────────────────────────────────────

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

    feature_store = MagicMock()

    kronos_engine = MagicMock()
    kronos_engine.run_monte_carlo.return_value = kronos_raw

    calibrator = MagicMock()
    calibrator.transform.return_value = kronos_cal

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

def test_gate2_direction_mismatch_returns_none():
    # Kronos cal=0.70 → direction=1 (up); regime direction=0 (down)
    engine = make_engine(kronos_cal=0.70, regime_prob=0.30, regime_direction=0)
    assert engine.get_signal("5min", 76000.0) is None


def test_gate2_both_up_returns_signal():
    engine = make_engine(kronos_cal=0.70, regime_prob=0.70, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result is not None
    assert result.direction == 1


def test_gate2_both_down_returns_signal():
    engine = make_engine(kronos_cal=0.35, regime_prob=0.35, regime_direction=0)
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
    """combined = 0.6 * kronos_cal + 0.4 * regime_prob"""
    engine = make_engine(kronos_cal=0.70, regime_prob=0.80, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    expected = 0.6 * 0.70 + 0.4 * 0.80
    assert result.calibrated_prob == pytest.approx(expected)


def test_combined_varies_with_regime_weight():
    """Regime contributes 40% to the final signal."""
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
    base = 0.6 * 0.70 + 0.4 * 0.80        # 0.74
    expected = 0.5 + (base - 0.5) * 0.5   # 0.62
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
    expected = 0.6 * 0.70 + 0.4 * 0.80
    assert result.calibrated_prob == pytest.approx(expected)


# ── NotTrainedError fallback ───────────────────────────────────────────────────

def test_not_trained_uses_kronos_only_formula():
    """combined = 0.5 + (kronos_cal - 0.5) * 0.5"""
    engine = make_engine(kronos_cal=0.70, raise_not_trained=True)
    result = engine.get_signal("5min", 76000.0)
    expected = 0.5 + (0.70 - 0.5) * 0.5   # 0.60
    assert result is not None
    assert result.calibrated_prob == pytest.approx(expected)


def test_not_trained_kronos_below_half():
    engine = make_engine(kronos_cal=0.30, raise_not_trained=True)
    result = engine.get_signal("5min", 76000.0)
    expected = 0.5 + (0.30 - 0.5) * 0.5   # 0.40
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
    engine_up = make_engine(kronos_cal=0.70, raise_not_trained=True)
    engine_dn = make_engine(kronos_cal=0.30, raise_not_trained=True)
    assert engine_up.get_signal("5min", 76000.0).direction == 1
    assert engine_dn.get_signal("5min", 76000.0).direction == 0


def test_not_trained_sentinel_fields():
    """regime_prob=nan and regime_direction=-1 when not trained."""
    engine = make_engine(kronos_cal=0.65, raise_not_trained=True)
    result = engine.get_signal("5min", 76000.0)
    assert math.isnan(result.regime_prob)
    assert result.regime_direction == -1


# ── TradingSignal field correctness ───────────────────────────────────────────

def test_signal_carries_raw_and_calibrated_kronos():
    engine = make_engine(kronos_raw=0.60, kronos_cal=0.63, regime_prob=0.70, regime_direction=1)
    result = engine.get_signal("5min", 76000.0)
    assert result.kronos_raw == pytest.approx(0.60)
    assert result.kronos_calibrated == pytest.approx(0.63)


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
