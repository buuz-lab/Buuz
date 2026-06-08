"""Tests for fusion.py Deribit option features (22-27) — TDD first."""
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from btc_kalshi_system.signal.fusion import SignalFusionEngine


_ALL_41_KEYS = [
    # kalshi_implied_prob, kalshi_spread_normalized, kalshi_open_imbalance, and
    # kalshi_early_drift intentionally excluded — regime model must be independent
    # of Kalshi to avoid circularity with Gates 5/8 — see regime_model.py
    "funding_rate", "funding_rate_trend", "oi_delta_pct", "cvd_normalized",
    "basis_spread_pct", "brti_volatility_1h", "cvd_velocity", "cvd_acceleration",
    "brti_momentum_5min", "brti_momentum_15min", "candle_progress",
    "hour_sin", "hour_cos", "funding_window_proximity",
    "trend_slope_1h", "trend_r2_1h", "hourly_sr_proximity", "range_breakout_flag",
    "tape_speed_tpm", "large_print_direction", "volume_ratio_1h",
    "atm_iv", "iv_rv_spread", "pcr_oi", "term_structure_slope", "skew_25d",
    "btc_24h_return",
    "kronos_raw_15min", "kronos_raw_5min",
    "btc_spx_corr_8d", "btc_qqq_corr_8d",
    # Session 39 — cascade momentum, cross-asset, order flow, options delta, LLM direction
    "liq_net_norm", "eth_direction_15min", "okx_spot_imbalance",
    "pcr_delta", "skew_delta", "deepseek_dir_prob",
    # Session 40 — microstructure divergence and directional trend context
    "cvd_price_divergence", "recent_up_fraction",
    # Session 42 — k15/Kalshi interaction features
    "k15_kalshi_alignment", "k15_delta",
]


def _make_feature_store_mock():
    feature_store = MagicMock()
    prices = np.linspace(95000, 95100, 15).tolist()
    idx = pd.date_range("2024-01-01", periods=15, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [0.0] * 15, "amount": [0.0] * 15,
    }, index=idx)
    # 26 candles for btc_24h_return (needs >= 25)
    h_prices = np.linspace(94000, 96000, 26).tolist()
    h_idx = pd.date_range("2024-01-01", periods=26, freq="1h", tz="UTC")
    df1h = pd.DataFrame({
        "open": h_prices, "high": h_prices, "low": h_prices, "close": h_prices,
        "volume": [0.0] * 26, "amount": [0.0] * 26,
    }, index=h_idx)
    def ohlcv_side_effect(tf):
        return df1h if tf == "1h" else df5
    feature_store.get_ohlcv.side_effect = ohlcv_side_effect
    import time
    now = time.time()
    feature_store._redis.zrange.return_value = [
        (b"0.1", now - 600),
        (b"0.2", now - 480),
        (b"0.3", now - 360),
        (b"0.4", now - 240),
        (b"0.5", now - 120),
    ]
    feature_store.get_raw_ticks.return_value = None
    return feature_store


def _make_engine(ctx: dict) -> SignalFusionEngine:
    engine = SignalFusionEngine(
        feature_store=_make_feature_store_mock(),
        kronos_engine=MagicMock(),
        calibrator=MagicMock(),
        regime_model=MagicMock(),
        deepseek_parser=MagicMock(),
    )
    engine.update_market_context(ctx)
    return engine


def _base_ctx() -> dict:
    return {
        "funding_rate": 0.0001,
        "funding_rate_trend": 0.00002,
        "oi_delta_pct": 0.001,
        "cvd_normalized": 0.3,
        "basis_spread_pct": 0.0005,
        "kalshi_mid_cents": 55.0,
    }


# ── test_regime_features_includes_all_27_keys ─────────────────────────────────

def test_regime_features_includes_all_28_keys():
    """_regime_features() must return exactly 41 keys in the correct order."""
    engine = _make_engine(_base_ctx())
    features, stale, deribit_stale, _ = engine._regime_features()
    keys = list(features.keys())
    assert keys == _ALL_41_KEYS, (
        f"Key mismatch.\nExpected: {_ALL_41_KEYS}\nGot:      {keys}"
    )
    assert len(keys) == 41


# ── test_deribit_stale flags ──────────────────────────────────────────────────

def test_deribit_stale_true_when_options_features_absent():
    """Context without atm_iv → deribit_stale=True."""
    engine = _make_engine(_base_ctx())  # no atm_iv
    _, _, deribit_stale, _ = engine._regime_features()
    assert deribit_stale is True


def test_deribit_stale_true_when_lkg_used():
    """Context with _deribit_lkg=True → deribit_stale=True."""
    ctx = {**_base_ctx(), "atm_iv": 55.0, "_deribit_lkg": True}
    engine = _make_engine(ctx)
    _, _, deribit_stale, _ = engine._regime_features()
    assert deribit_stale is True


def test_deribit_stale_false_when_options_fresh():
    """Context with valid atm_iv and no _deribit_lkg → deribit_stale=False."""
    ctx = {**_base_ctx(), "atm_iv": 55.0, "pcr_oi": 1.1}
    engine = _make_engine(ctx)
    _, _, deribit_stale, _ = engine._regime_features()
    assert deribit_stale is False


# ── test_iv_rv_spread_in_context ──────────────────────────────────────────────

def test_iv_rv_spread_in_features():
    """When ctx has atm_iv and brti_volatility_1h, iv_rv_spread must appear in features."""
    # brti_volatility_1h in ctx is the raw value from Redis (used by DeepSeek)
    # the OHLCV-based volatility is computed separately in _regime_features
    # iv_rv_spread = atm_iv - brti_volatility_1h (from ctx)
    ctx = {**_base_ctx(), "atm_iv": 60.0, "iv_rv_spread": 59.99}
    engine = _make_engine(ctx)
    features, _, _, _ = engine._regime_features()
    assert "iv_rv_spread" in features
    # Should use ctx's iv_rv_spread (already merged by _get_market_context)
    assert features["iv_rv_spread"] == pytest.approx(59.99, rel=1e-4)


def test_iv_rv_spread_defaults_to_zero_when_absent():
    """Missing iv_rv_spread in ctx → 0.0 in features (no crash)."""
    ctx = {**_base_ctx(), "atm_iv": 60.0}  # no iv_rv_spread
    engine = _make_engine(ctx)
    features, _, _, _ = engine._regime_features()
    assert features["iv_rv_spread"] == pytest.approx(0.0)


# ── test_kalshi_spread_in_regime_features ─────────────────────────────────────

def test_kalshi_spread_in_regime_features():
    """kalshi_spread_normalized excluded from features dict to break Kalshi circularity."""
    engine = _make_engine(_base_ctx())
    engine.update_kalshi_spread(0.05)
    features, _, _, _ = engine._regime_features()
    assert "kalshi_spread_normalized" not in features


def test_kalshi_spread_defaults_to_zero():
    """kalshi_spread_normalized excluded from features dict to break Kalshi circularity."""
    engine = _make_engine(_base_ctx())
    features, _, _, _ = engine._regime_features()
    assert "kalshi_spread_normalized" not in features


# ── test_pcr_oi_default_is_one ────────────────────────────────────────────────

def test_pcr_oi_default_is_one():
    """ctx missing pcr_oi → pcr_oi == 1.0 in features (not 0.0)."""
    ctx = {**_base_ctx(), "atm_iv": 55.0}  # no pcr_oi
    engine = _make_engine(ctx)
    features, _, _, _ = engine._regime_features()
    assert features["pcr_oi"] == pytest.approx(1.0)


# ── test_deribit_stale_independent_of_features_stale ─────────────────────────

def test_deribit_stale_independent_of_features_stale():
    """deribit_stale and features_stale are independent flags — test they can differ."""
    # Fresh Deribit data, but LKG regime features (stale=True)
    ctx = {"_lkg": True, "atm_iv": 55.0, "pcr_oi": 1.0,
           "kalshi_mid_cents": 55.0}
    engine = _make_engine(ctx)
    features, features_stale, deribit_stale, _ = engine._regime_features()
    assert features_stale is True    # regime LKG used
    assert deribit_stale is False    # Deribit data is fresh


# ── test_numeric_fallbacks_for_zero_inputs ────────────────────────────────────

def test_numeric_fallbacks_are_floats():
    """All 6 new features must be floats (not None) even when ctx provides nothing."""
    engine = _make_engine(_base_ctx())  # no deribit data at all
    features, _, _, _ = engine._regime_features()
    for key in ("atm_iv", "iv_rv_spread", "pcr_oi", "term_structure_slope", "skew_25d"):
        v = features[key]
        assert isinstance(v, float), f"{key} should be float, got {type(v)}"
        assert not math.isnan(v), f"{key} should not be NaN"
