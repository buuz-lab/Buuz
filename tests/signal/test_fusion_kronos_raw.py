"""
TDD tests for get_signal() kronos_raw parameter.

When kronos_raw is provided: skip run_monte_carlo(), use the provided value.
When kronos_raw is None (default): call run_monte_carlo() as before (backward compat).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from btc_kalshi_system.models.regime_model import NotTrainedError
from btc_kalshi_system.signal.fusion import SignalFusionEngine


def _make_feature_store_mock():
    fs = MagicMock()
    prices = np.linspace(95000, 95100, 15).tolist()
    idx = pd.date_range("2024-01-01", periods=15, freq="5min", tz="UTC")
    df5 = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": [0.0] * 15, "amount": [0.0] * 15},
        index=idx,
    )
    h_prices = np.linspace(94000, 96000, 5).tolist()
    h_idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df1h = pd.DataFrame(
        {"open": h_prices, "high": h_prices, "low": h_prices, "close": h_prices,
         "volume": [0.0] * 5, "amount": [0.0] * 5},
        index=h_idx,
    )
    fs.get_ohlcv.side_effect = lambda tf: df1h if tf == "1h" else df5
    import time
    now = time.time()
    fs._redis.zrange.return_value = [
        (b"0.1", now - 600), (b"0.2", now - 480), (b"0.3", now - 360),
        (b"0.4", now - 240), (b"0.5", now - 120),
    ]
    fs.get_raw_ticks.return_value = None
    return fs


def _make_engine(kronos_mc_return: float = 0.50, kronos_cal: float = 0.65) -> SignalFusionEngine:
    kronos = MagicMock()
    kronos.run_monte_carlo.return_value = kronos_mc_return

    calibrator = MagicMock()
    calibrator.transform.return_value = kronos_cal

    regime = MagicMock()
    regime.get_regime.return_value = {
        "prob_up": 0.65, "direction": 1, "confidence": 0.5,
    }

    ds = MagicMock()
    ds.get_current_context.return_value = {
        "regime": "trending_up", "confidence": 0.75,
        "suppress_trading": False, "suppress_reason": None, "notes": "",
    }

    return SignalFusionEngine(
        feature_store=_make_feature_store_mock(),
        kronos_engine=kronos,
        calibrator=calibrator,
        regime_model=regime,
        deepseek_parser=ds,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_get_signal_skips_mc_when_kronos_raw_provided():
    """run_monte_carlo must NOT be called when kronos_raw is supplied."""
    engine = _make_engine(kronos_mc_return=0.50)
    engine.get_signal("15min", 95000.0, kronos_raw=0.70)
    engine._kronos.run_monte_carlo.assert_not_called()


def test_get_signal_calls_mc_when_kronos_raw_none():
    """run_monte_carlo IS called when kronos_raw is None (backward compat for tests)."""
    engine = _make_engine(kronos_mc_return=0.65)
    engine.get_signal("15min", 95000.0, kronos_raw=None)
    engine._kronos.run_monte_carlo.assert_called_once()


def test_get_signal_uses_provided_kronos_raw_value():
    """Phase 3c: provided kronos_raw is stored on the signal; calibrator transforms regime_prob."""
    engine = _make_engine(kronos_mc_return=0.50, kronos_cal=0.50)
    result = engine.get_signal("15min", 95000.0, kronos_raw=0.90)
    # kronos_raw stored on signal
    assert result.kronos_raw == pytest.approx(0.90)
    assert result.kronos_calibrated == pytest.approx(0.90)  # Phase 3c: k15_raw, not calibrated
    # calibrator called with regime_prob (0.65 from mock), not kronos_raw (0.90)
    call_args = engine._calibrator.transform.call_args
    assert call_args.args[0] == pytest.approx(0.65)  # regime_prob from mock
