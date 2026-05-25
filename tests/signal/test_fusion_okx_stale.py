"""Tests for okx_stale flag in fusion._regime_features()."""
import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from btc_kalshi_system.signal.fusion import SignalFusionEngine


def _make_feature_store_mock():
    feature_store = MagicMock()
    prices = np.linspace(95000, 95100, 15).tolist()
    idx = pd.date_range("2024-01-01", periods=15, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [0.0] * 15, "amount": [0.0] * 15,
    }, index=idx)
    h_prices = np.linspace(94000, 96000, 5).tolist()
    h_idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df1h = pd.DataFrame({
        "open": h_prices, "high": h_prices, "low": h_prices, "close": h_prices,
        "volume": [0.0] * 5, "amount": [0.0] * 5,
    }, index=h_idx)
    feature_store.get_ohlcv.side_effect = lambda tf: df1h if tf == "1h" else df5
    now = time.time()
    feature_store._redis.zrange.return_value = [
        (b"0.1", now - 600), (b"0.2", now - 480), (b"0.3", now - 360),
        (b"0.4", now - 240), (b"0.5", now - 120),
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
        "cvd_normalized": 0.3,
        "kalshi_mid_cents": 55.0,
    }


# ── tests ─────────────────────────────────────────────────────────────────────

def test_okx_stale_true_when_lkg_used():
    """ctx with _lkg=True → okx_stale=True."""
    ctx = {**_base_ctx(), "_lkg": True}
    engine = _make_engine(ctx)
    _, _, _, okx_stale = engine._regime_features()
    assert okx_stale is True


def test_okx_stale_true_when_partial():
    """ctx with _okx_partial=True (no _lkg) → okx_stale=True."""
    ctx = {**_base_ctx(), "_okx_partial": True}
    engine = _make_engine(ctx)
    _, _, _, okx_stale = engine._regime_features()
    assert okx_stale is True


def test_okx_stale_false_when_fresh():
    """ctx with neither _lkg nor _okx_partial → okx_stale=False."""
    ctx = {**_base_ctx(), "atm_iv": 30.0}
    engine = _make_engine(ctx)
    _, _, _, okx_stale = engine._regime_features()
    assert okx_stale is False
