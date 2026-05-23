import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import fakeredis

from btc_kalshi_system.data.derivatives_feed import DerivativesFeed


def make_feed() -> DerivativesFeed:
    """DerivativesFeed with fakeredis and no real ccxt exchange."""
    feed = DerivativesFeed.__new__(DerivativesFeed)
    feed._redis = fakeredis.FakeRedis()
    feed._exchange = MagicMock()
    return feed


# ── funding_rate_trend (4h delta) ──────────────────────────────────────────────

def test_funding_rate_trend_is_4h_delta():
    feed = make_feed()
    # Two funding rate entries 4h apart
    history = [
        {"timestamp": 0,           "fundingRate": 0.01},
        {"timestamp": 4 * 3600_000, "fundingRate": 0.03},
    ]
    trend = feed._funding_rate_trend(history)
    assert trend == pytest.approx(0.02)


def test_funding_rate_trend_returns_zero_when_insufficient_history():
    feed = make_feed()
    trend = feed._funding_rate_trend([{"timestamp": 0, "fundingRate": 0.01}])
    assert trend == pytest.approx(0.0)


def test_funding_rate_trend_returns_zero_when_no_entry_older_than_window():
    feed = make_feed()
    # Both entries within the 4-hour lookback — no entry older than cutoff
    _1h_ms = 3_600_000
    history = [
        {"timestamp": 0,      "fundingRate": 0.01},
        {"timestamp": _1h_ms, "fundingRate": 0.03},
    ]
    assert feed._funding_rate_trend(history) == pytest.approx(0.0)


# ── oi_delta_pct ───────────────────────────────────────────────────────────────

def test_oi_delta_pct_positive_growth():
    feed = make_feed()
    delta = feed._oi_delta_pct(prev_oi=1000.0, curr_oi=1100.0)
    assert delta == pytest.approx(0.10)


def test_oi_delta_pct_negative_growth():
    feed = make_feed()
    delta = feed._oi_delta_pct(prev_oi=1000.0, curr_oi=900.0)
    assert delta == pytest.approx(-0.10)


def test_oi_delta_pct_zero_when_prev_is_zero():
    feed = make_feed()
    delta = feed._oi_delta_pct(prev_oi=0.0, curr_oi=1000.0)
    assert delta == pytest.approx(0.0)


# ── cvd_normalized ─────────────────────────────────────────────────────────────

def test_cvd_normalized_all_buys_is_positive_one():
    feed = make_feed()
    trades = [
        {"amount": 1.0, "side": "buy"},
        {"amount": 2.0, "side": "buy"},
    ]
    cvd = feed._cvd_normalized(trades)
    assert cvd == pytest.approx(1.0)


def test_cvd_normalized_all_sells_is_negative_one():
    feed = make_feed()
    trades = [
        {"amount": 1.0, "side": "sell"},
        {"amount": 3.0, "side": "sell"},
    ]
    cvd = feed._cvd_normalized(trades)
    assert cvd == pytest.approx(-1.0)


def test_cvd_normalized_balanced_buys_and_sells_is_zero():
    feed = make_feed()
    trades = [
        {"amount": 5.0, "side": "buy"},
        {"amount": 5.0, "side": "sell"},
    ]
    cvd = feed._cvd_normalized(trades)
    assert cvd == pytest.approx(0.0)


def test_cvd_normalized_returns_zero_for_empty_trades():
    feed = make_feed()
    assert feed._cvd_normalized([]) == pytest.approx(0.0)


# ── brti_volatility_1h ─────────────────────────────────────────────────────────

def test_brti_volatility_1h_from_redis_ticks():
    feed = make_feed()
    now = time.time()
    prices = [100.0, 101.0, 99.0, 102.0, 98.0]
    for i, p in enumerate(prices):
        feed._redis.lpush("brti:ticks", f"{now - 100 + i}:{p}")

    vol = feed._brti_volatility_1h()
    expected = float(np.std(prices, ddof=1) / np.mean(prices))
    assert vol == pytest.approx(expected, rel=1e-5)


def test_brti_volatility_1h_returns_zero_when_no_ticks():
    feed = make_feed()
    assert feed._brti_volatility_1h() == pytest.approx(0.0)


# ── write_features_to_redis ────────────────────────────────────────────────────

def test_features_written_to_redis_key_with_ttl():
    feed = make_feed()
    features = {
        "funding_rate": 0.01,
        "funding_rate_trend": 0.002,
        "oi_delta_pct": 0.05,
        "cvd_normalized": 0.3,
        "basis_spread_pct": -0.001,
        "brti_volatility_1h": 0.008,
    }
    feed._write_features(features)
    raw = feed._redis.get("regime:features")
    assert raw is not None
    ttl = feed._redis.ttl("regime:features")
    assert 590 <= ttl <= 600


def test_features_contain_all_six_keys():
    import json
    feed = make_feed()
    features = {
        "funding_rate": 0.01,
        "funding_rate_trend": 0.002,
        "oi_delta_pct": 0.05,
        "cvd_normalized": 0.3,
        "basis_spread_pct": -0.001,
        "brti_volatility_1h": 0.008,
    }
    feed._write_features(features)
    raw = feed._redis.get("regime:features")
    loaded = json.loads(raw)
    for key in ("funding_rate", "funding_rate_trend", "oi_delta_pct",
                "cvd_normalized", "basis_spread_pct", "brti_volatility_1h"):
        assert key in loaded


def test_lkg_key_written_on_successful_write():
    """_write_features must also populate regime:features:lkg with a 24h TTL
    and a _lkg_written_at timestamp so _get_market_context can fall back to
    real features (rather than zeros) during exchange outages."""
    import json, time
    feed = make_feed()
    features = {
        "funding_rate": 0.01,
        "funding_rate_trend": 0.002,
        "oi_delta_pct": 0.05,
        "cvd_normalized": 0.3,
        "basis_spread_pct": -0.001,
        "brti_volatility_1h": 0.008,
    }
    before = time.time()
    feed._write_features(features)

    # Key must exist
    raw_lkg = feed._redis.get("regime:features:lkg")
    assert raw_lkg is not None, "regime:features:lkg was not written"

    # TTL must be ~24 h (allow a couple of seconds of slack)
    ttl = feed._redis.ttl("regime:features:lkg")
    assert 86_390 <= ttl <= 86_400, f"Expected ~86400s TTL, got {ttl}"

    # Payload must contain all six feature keys plus _lkg_written_at
    lkg = json.loads(raw_lkg)
    for key in ("funding_rate", "funding_rate_trend", "oi_delta_pct",
                "cvd_normalized", "basis_spread_pct", "brti_volatility_1h"):
        assert key in lkg, f"LKG key missing: {key}"
    assert "_lkg_written_at" in lkg
    assert lkg["_lkg_written_at"] >= before

    # The six feature values must match what was written
    assert lkg["funding_rate"] == 0.01
    assert lkg["cvd_normalized"] == 0.3
