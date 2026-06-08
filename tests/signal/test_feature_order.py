"""
Feature order consistency test.

_FEATURE_ORDER in regime_model.py, _FEATURE_COLS in train_regime.py (non-legacy),
and the keys returned from fusion._regime_features() must be identical.
A mismatch silently corrupts the XGBoost model at training time.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from btc_kalshi_system.models.regime_model import _FEATURE_ORDER
# Import _FEATURE_COLS from train_regime
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_regime",
    Path(__file__).resolve().parent.parent.parent / "scripts" / "train_regime.py",
)
_train_regime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_train_regime)
_FEATURE_COLS = _train_regime._FEATURE_COLS


def _make_feature_store_mock():
    """Feature store mock with enough data for all new features."""
    feature_store = MagicMock()
    prices = np.linspace(95000, 95100, 15).tolist()
    idx = pd.date_range("2024-01-01", periods=15, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [0.0] * 15, "amount": [0.0] * 15,
    }, index=idx)
    # 26 hourly candles to provide enough data for btc_24h_return (needs >= 25)
    h_prices = np.linspace(94000, 96000, 26).tolist()
    h_idx = pd.date_range("2024-01-01", periods=26, freq="1h", tz="UTC")
    df1h = pd.DataFrame({
        "open": h_prices, "high": h_prices, "low": h_prices, "close": h_prices,
        "volume": [0.0] * 26, "amount": [0.0] * 26,
    }, index=h_idx)
    def ohlcv_side_effect(tf):
        return df1h if tf == "1h" else df5
    feature_store.get_ohlcv.side_effect = ohlcv_side_effect
    now = 1704067200.0
    feature_store._redis.zrange.return_value = [
        (b"0.1", now - 600),
        (b"0.2", now - 480),
        (b"0.3", now - 360),
        (b"0.4", now - 240),
        (b"0.5", now - 120),
    ]
    feature_store.get_raw_ticks.return_value = None
    return feature_store


def _get_fusion_feature_keys():
    """Call fusion._regime_features() with a fully-mocked engine and return the key list."""
    from btc_kalshi_system.signal.fusion import SignalFusionEngine
    engine = SignalFusionEngine(
        feature_store=_make_feature_store_mock(),
        kronos_engine=MagicMock(),
        calibrator=MagicMock(),
        regime_model=MagicMock(),
        deepseek_parser=MagicMock(),
    )
    engine.update_market_context({
        "funding_rate": 0.0001,
        "funding_rate_trend": 0.00002,
        "oi_delta_pct": 0.001,
        "cvd_normalized": 0.3,
        "basis_spread_pct": 0.0005,
        "kalshi_mid_cents": 55.0,
    })
    features, _, _, _ = engine._regime_features()
    return list(features.keys())


def test_feature_order_regime_model_vs_train_regime():
    """_FEATURE_ORDER (regime_model.py) must exactly match _FEATURE_COLS (train_regime.py)."""
    assert _FEATURE_ORDER == _FEATURE_COLS, (
        f"Mismatch between regime_model._FEATURE_ORDER and train_regime._FEATURE_COLS.\n"
        f"In regime_model but not train_regime: {set(_FEATURE_ORDER) - set(_FEATURE_COLS)}\n"
        f"In train_regime but not regime_model: {set(_FEATURE_COLS) - set(_FEATURE_ORDER)}"
    )


def test_feature_order_regime_model_vs_fusion():
    """_FEATURE_ORDER (regime_model.py) must exactly match keys from fusion._regime_features()."""
    fusion_keys = _get_fusion_feature_keys()
    assert _FEATURE_ORDER == fusion_keys, (
        f"Mismatch between regime_model._FEATURE_ORDER and fusion._regime_features() keys.\n"
        f"Order: {_FEATURE_ORDER}\nvs fusion: {fusion_keys}"
    )


def test_feature_order_all_three_match():
    """All three sources must be identical including ORDER (not just membership)."""
    fusion_keys = _get_fusion_feature_keys()
    assert _FEATURE_ORDER == _FEATURE_COLS == fusion_keys
    assert len(_FEATURE_ORDER) == 39  # was 41; removed kalshi_open_imbalance, kalshi_early_drift


def test_feature_order_includes_kronos_features():
    """Kronos meta-features must appear in _FEATURE_ORDER for regime model training."""
    assert "kronos_raw_15min" in _FEATURE_ORDER
    assert "kronos_raw_5min" in _FEATURE_ORDER
