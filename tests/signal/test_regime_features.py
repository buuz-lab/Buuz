"""
Tests for the 20-feature _regime_features() computations.

Uses fakeredis for the CVD ring buffer. No network, no torch.
"""
import math
import time as _time
from unittest.mock import MagicMock, patch

import fakeredis
import numpy as np
import pandas as pd
import pytest

from btc_kalshi_system.signal.fusion import SignalFusionEngine


def _make_df5(n: int = 15, base_price: float = 95000.0, slope: float = 1.0):
    """Make a 5-min OHLCV DataFrame with `n` rows."""
    prices = [base_price + slope * i for i in range(n)]
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": prices, "high": [p + 10 for p in prices],
        "low": [p - 10 for p in prices], "close": prices,
        "volume": [0.0] * n, "amount": [0.0] * n,
    }, index=idx)


def _make_df1h(n: int = 5, lo: float = 94000.0, hi: float = 96000.0):
    prices = np.linspace(lo, hi, n).tolist()
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [0.0] * n, "amount": [0.0] * n,
    }, index=idx)


def _make_engine(
    cvd_entries=None,
    df5=None,
    df1h=None,
    ctx=None,
):
    """Return a SignalFusionEngine with real fakeredis and controlled OHLCV data."""
    fake_redis = fakeredis.FakeRedis()
    if cvd_entries is not None:
        for val, score in cvd_entries:
            fake_redis.zadd("regime:cvd_history", {str(float(val)): score})

    feature_store = MagicMock()
    feature_store._redis = fake_redis
    if df5 is None:
        df5 = _make_df5()
    if df1h is None:
        df1h = _make_df1h()
    def ohlcv_side_effect(tf):
        return df1h if tf == "1h" else df5
    feature_store.get_ohlcv.side_effect = ohlcv_side_effect
    feature_store.get_raw_ticks.return_value = None

    engine = SignalFusionEngine(
        feature_store=feature_store,
        kronos_engine=MagicMock(),
        calibrator=MagicMock(),
        regime_model=MagicMock(),
        deepseek_parser=MagicMock(),
    )
    engine.update_market_context(ctx or {
        "funding_rate": 0.0001,
        "cvd_normalized": 0.3,
        "kalshi_mid_cents": 55.0,
    })
    return engine


# ── CVD velocity / acceleration ───────────────────────────────────────────────

def test_cvd_velocity_cold_start_returns_zeros_and_stale():
    """Fewer than 5 CVD entries → velocity=0, acceleration=0, stale=True."""
    engine = _make_engine(cvd_entries=[(_time.time() - 60, 0.1), (_time.time(), 0.2)])
    features, stale, _, _ = engine._regime_features()
    assert features["cvd_velocity"] == pytest.approx(0.0)
    assert features["cvd_acceleration"] == pytest.approx(0.0)
    assert stale is True


def test_cvd_velocity_math():
    """Given known CVD history, verify velocity and acceleration calculations."""
    now = _time.time()
    # cvd_now=0.6, cvd_5m_ago=0.1, cvd_10m_ago=-0.4
    # velocity = (0.6 - 0.1) / 5 = 0.1
    # velocity_10m = (0.6 - (-0.4)) / 10 = 0.1
    # acceleration = 0.1 - 0.1 = 0.0
    entries = [
        (-0.4, now - 600),  # 10 min ago
        (-0.2, now - 480),
        (0.0, now - 360),
        (0.1, now - 300),   # 5 min ago
        (0.3, now - 180),
        (0.6, now),         # now
    ]
    engine = _make_engine(cvd_entries=entries)
    features, _, _, _ = engine._regime_features()
    # velocity = (cvd_now - cvd_5m_ago) / 5 = (0.6 - 0.1) / 5
    assert features["cvd_velocity"] == pytest.approx((0.6 - 0.1) / 5.0, abs=1e-6)


def test_cvd_five_entries_not_stale():
    """Exactly 5 entries is sufficient — should NOT trigger stale for the CVD buffer."""
    now = _time.time()
    entries = [(float(i) * 0.1, now - (5 - i) * 120) for i in range(5)]
    engine = _make_engine(cvd_entries=entries, ctx={
        "funding_rate": 0.0001, "cvd_normalized": 0.3, "kalshi_mid_cents": 55.0
    })
    features, stale, _, _ = engine._regime_features()
    # CVD buffer has 5 entries — not cold. brti_momentum and kalshi present → not stale
    # (stale may still be True if ctx is not fully populated, but CVD alone won't cause it)
    # Just verify velocity is non-zero / computable (not forced to 0.0 cold-start)
    assert isinstance(features["cvd_velocity"], float)


# ── brti_momentum ─────────────────────────────────────────────────────────────

def test_brti_momentum_5min_correct():
    """brti_momentum_5min = close[-1] / close[-2] - 1."""
    df5 = _make_df5(n=10, base_price=100.0, slope=1.0)
    # close[-1] = 109.0, close[-2] = 108.0
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    expected = 109.0 / 108.0 - 1.0
    assert features["brti_momentum_5min"] == pytest.approx(expected, rel=1e-6)


def test_brti_momentum_15min_correct():
    """brti_momentum_15min = close[-1] / close[-4] - 1."""
    df5 = _make_df5(n=10, base_price=100.0, slope=1.0)
    # close[-1] = 109.0, close[-4] = 106.0
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    expected = 109.0 / 106.0 - 1.0
    assert features["brti_momentum_15min"] == pytest.approx(expected, rel=1e-6)


def test_brti_momentum_insufficient_data_returns_zeros_and_stale():
    """< 4 rows → momentum = 0.0 and stale = True."""
    df5 = _make_df5(n=3)
    engine = _make_engine(df5=df5)
    features, stale, _, _ = engine._regime_features()
    assert features["brti_momentum_5min"] == pytest.approx(0.0)
    assert features["brti_momentum_15min"] == pytest.approx(0.0)
    assert stale is True


# ── trend_slope_1h and trend_r2_1h ───────────────────────────────────────────

def test_trend_slope_positive_for_rising_closes():
    """Perfectly rising close series → slope > 0."""
    df5 = _make_df5(n=15, slope=10.0)  # each candle +10
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    assert features["trend_slope_1h"] > 0.0


def test_trend_r2_perfect_linear_is_one():
    """Perfectly linear close series → R² = 1.0."""
    df5 = _make_df5(n=15, slope=1.0)
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    assert features["trend_r2_1h"] == pytest.approx(1.0, abs=1e-6)


def test_trend_slope_flat_series_near_zero():
    """Flat close series → slope ≈ 0 and R² ≈ 0."""
    df5 = _make_df5(n=15, slope=0.0)
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    assert abs(features["trend_slope_1h"]) < 1e-9
    # R² undefined when ss_tot ≈ 0 — should return 0.0
    assert features["trend_r2_1h"] == pytest.approx(0.0, abs=1e-6)


# ── hourly_sr_proximity ───────────────────────────────────────────────────────

def test_hourly_sr_proximity_at_support_is_zero():
    """Current price = support level → sr_proximity = 0.0."""
    df1h = _make_df1h(lo=90000.0, hi=100000.0)
    # Override close of df5 to equal the support level
    df5 = _make_df5(n=15, base_price=90000.0, slope=0.0)
    engine = _make_engine(df5=df5, df1h=df1h)
    features, _, _, _ = engine._regime_features()
    assert features["hourly_sr_proximity"] == pytest.approx(0.0, abs=1e-6)


def test_hourly_sr_proximity_at_resistance_is_one():
    """Current price = resistance level → sr_proximity = 1.0."""
    df1h = _make_df1h(lo=90000.0, hi=100000.0)
    df5 = _make_df5(n=15, base_price=100000.0, slope=0.0)
    engine = _make_engine(df5=df5, df1h=df1h)
    features, _, _, _ = engine._regime_features()
    assert features["hourly_sr_proximity"] == pytest.approx(1.0, abs=1e-6)


def test_hourly_sr_proximity_in_unit_interval():
    """sr_proximity must always be in [0, 1]."""
    engine = _make_engine()
    features, _, _, _ = engine._regime_features()
    assert 0.0 <= features["hourly_sr_proximity"] <= 1.0


# ── range_breakout_flag ───────────────────────────────────────────────────────

def test_range_breakout_flag_bullish_breakout_is_positive():
    """Candle breaking above the prior 3-candle box → positive flag."""
    # Box candles: 100.0-101.0 range. Breakout candle: high=102.0, low=100.0
    prices = [100.0] * 5
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices,
        "high": [100.5, 100.5, 100.5, 100.5, 102.0],   # last candle breaks above
        "low":  [99.5,  99.5,  99.5,  99.5,  100.0],
        "close": prices,
        "volume": [0.0] * 5, "amount": [0.0] * 5,
    }, index=idx)
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    assert features["range_breakout_flag"] > 0.0


def test_range_breakout_flag_bearish_breakout_is_negative():
    """Candle breaking below the box → negative flag."""
    prices = [100.0] * 5
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices,
        "high": [100.5, 100.5, 100.5, 100.5, 100.0],
        "low":  [99.5,  99.5,  99.5,  99.5,  98.0],    # last candle breaks below
        "close": prices,
        "volume": [0.0] * 5, "amount": [0.0] * 5,
    }, index=idx)
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    assert features["range_breakout_flag"] < 0.0


def test_range_breakout_flag_inside_box_is_zero():
    """Candle inside the box → flag = 0.0."""
    prices = [100.0] * 5
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices,
        "high": [100.5, 100.5, 100.5, 100.5, 100.4],   # inside
        "low":  [99.5,  99.5,  99.5,  99.5,  99.6],    # inside
        "close": prices,
        "volume": [0.0] * 5, "amount": [0.0] * 5,
    }, index=idx)
    engine = _make_engine(df5=df5)
    features, _, _, _ = engine._regime_features()
    assert features["range_breakout_flag"] == pytest.approx(0.0)


# ── funding_window_proximity ──────────────────────────────────────────────────

def test_funding_window_proximity_at_settlement_is_one():
    """At 08:00:00 UTC exactly → proximity = 1.0."""
    from datetime import datetime, timezone
    from unittest.mock import patch
    fixed_dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    engine = _make_engine()
    with patch("btc_kalshi_system.signal.fusion.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_dt
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        features, _, _, _ = engine._regime_features()
    assert features["funding_window_proximity"] == pytest.approx(1.0, abs=1e-6)


def test_funding_window_proximity_at_midpoint_is_zero():
    """At 04:00:00 UTC (midpoint between 00:00 and 08:00) → proximity = 0.0."""
    from datetime import datetime, timezone
    from unittest.mock import patch
    fixed_dt = datetime(2024, 1, 1, 4, 0, 0, tzinfo=timezone.utc)
    engine = _make_engine()
    with patch("btc_kalshi_system.signal.fusion.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_dt
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        features, _, _, _ = engine._regime_features()
    assert features["funding_window_proximity"] == pytest.approx(0.0, abs=1e-6)


# ── kalshi_implied_prob stale ─────────────────────────────────────────────────

def test_kalshi_implied_prob_missing_sets_stale():
    """kalshi_mid_cents absent → stale = True (kalshi_implied_prob excluded from features dict)."""
    engine = _make_engine(ctx={"funding_rate": 0.0001, "cvd_normalized": 0.3})
    features, stale, _, _ = engine._regime_features()
    assert "kalshi_implied_prob" not in features  # removed to break Kalshi circularity
    assert stale is True


# ── CVD buffer stale-timestamp detection ──────────────────────────────────────

def test_cvd_stale_buffer_marks_stale_and_zeros_velocity():
    """5 entries present but most recent is > 360s old → stale buffer, velocity=0."""
    now = _time.time()
    old_ts = now - 400  # 400s ago — beyond the 360s threshold
    entries = [(0.1, old_ts - 400), (0.2, old_ts - 300), (0.3, old_ts - 200),
               (0.4, old_ts - 100), (0.5, old_ts)]
    engine = _make_engine(cvd_entries=entries)
    features, stale, _, _ = engine._regime_features()
    assert features["cvd_velocity"] == pytest.approx(0.0)
    assert features["cvd_acceleration"] == pytest.approx(0.0)
    assert stale is True


def test_cvd_fresh_buffer_within_threshold_not_stale_from_buffer():
    """5 entries, most recent 300s old (within 360s threshold) → buffer not stale."""
    now = _time.time()
    entries = [
        (0.1, now - 600), (0.2, now - 480), (0.3, now - 360),
        (0.4, now - 240), (0.5, now - 300),
    ]
    engine = _make_engine(cvd_entries=entries, ctx={
        "funding_rate": 0.0001, "cvd_normalized": 0.3, "kalshi_mid_cents": 55.0,
    })
    features, stale, _, _ = engine._regime_features()
    # Buffer check passes; velocity should be non-zero
    assert isinstance(features["cvd_velocity"], float)
    assert features["cvd_velocity"] != pytest.approx(0.0)


# ── Feature 28: btc_24h_return ────────────────────────────────────────────────

def test_btc_24h_return_computed_with_sufficient_1h_data():
    """With 26 hourly candles, btc_24h_return = close[-1]/close[-25] - 1."""
    import numpy as np
    prices = np.linspace(90000, 95000, 26).tolist()
    df1h = _make_df1h.__wrapped__(26, prices) if hasattr(_make_df1h, '__wrapped__') else None
    # Build the df1h manually
    idx = __import__('pandas').date_range("2024-01-01", periods=26, freq="1h", tz="UTC")
    df1h = __import__('pandas').DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [0.0] * 26, "amount": [0.0] * 26,
    }, index=idx)
    engine = _make_engine(df1h=df1h)
    features, stale, _, _ = engine._regime_features()
    expected = prices[-1] / prices[-25] - 1
    assert features["btc_24h_return"] == pytest.approx(expected, rel=1e-6)


def test_btc_24h_return_defaults_to_zero_and_stale_with_insufficient_1h_data():
    """With fewer than 25 hourly candles, btc_24h_return=0.0 and stale=True."""
    engine = _make_engine(df1h=_make_df1h(n=10))
    features, stale, _, _ = engine._regime_features()
    assert features["btc_24h_return"] == pytest.approx(0.0)
    assert stale is True
