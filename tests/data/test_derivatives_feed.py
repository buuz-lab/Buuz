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
    feed._kraken_exchange = None
    feed._ccxt_async = MagicMock()
    feed._prev_oi = 0.0
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


# ── Fallback paths ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_coinglass_fallback_when_okx_funding_oi_fails():
    """When OKX funding/OI raises, _coinglass_funding_and_oi() is called and values are non-zero."""
    feed = make_feed()
    feed._prev_oi = 1000.0
    feed._exchange = AsyncMock()
    feed._exchange.fetch_funding_rate_history.side_effect = Exception("OKX unreachable")
    feed._exchange.fetch_open_interest.side_effect = Exception("OKX unreachable")

    coinglass_result = (0.0035, 0.0012, 0.05, False)
    with patch.object(feed, "_coinglass_funding_and_oi", new=AsyncMock(return_value=coinglass_result)):
        curr_funding, trend, oi_delta, okx_partial = await feed._fetch_funding_and_oi()

    assert curr_funding == pytest.approx(0.0035)
    assert trend == pytest.approx(0.0012)
    assert oi_delta == pytest.approx(0.05)
    assert okx_partial is False


@pytest.mark.asyncio
async def test_kraken_fallback_when_okx_trades_fail():
    """When OKX fetch_trades raises, Kraken fallback is called and CVD/basis are non-zero."""
    feed = make_feed()
    feed._exchange = AsyncMock()
    feed._exchange.fetch_trades.side_effect = Exception("OKX unreachable")

    # Fake Kraken exchange returning trades with a buy skew
    kraken_trades = [
        {"amount": 3.0, "side": "buy", "price": 67000.0},
        {"amount": 1.0, "side": "sell", "price": 67000.0},
    ]
    mock_kraken = AsyncMock()
    mock_kraken.fetch_trades.return_value = kraken_trades
    feed._ccxt_async.kraken.return_value = mock_kraken

    # Seed BRTI so basis_spread_pct has a denominator
    feed._redis.set("brti:resolution_estimate", "67000.0")

    cvd, basis, large_print = await feed._fetch_trades_data()

    feed._ccxt_async.kraken.assert_called_once()
    mock_kraken.fetch_trades.assert_called_once()
    assert cvd == pytest.approx(0.5)   # (3-1)/(3+1)
    assert basis == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_coinglass_fallback_skipped_when_api_key_empty():
    """When COINGLASS_API_KEY is empty, _coinglass_funding_and_oi returns zeros without raising."""
    feed = make_feed()
    feed._prev_oi = 0.0

    import btc_kalshi_system.data.derivatives_feed as df_module
    with patch.object(df_module, "COINGLASS_API_KEY", ""):
        curr_funding, trend, oi_delta, okx_partial = await feed._coinglass_funding_and_oi()

    assert curr_funding == pytest.approx(0.0)
    assert trend == pytest.approx(0.0)
    assert oi_delta == pytest.approx(0.0)
    assert okx_partial is True


# ── _large_print_direction ─────────────────────────────────────────────────────

def test_large_print_direction_all_buys_above_threshold():
    feed = make_feed()
    # avg=2.0, threshold=4.0; trade at 5.0 is large and a buy
    trades = [
        {"amount": 1.0, "side": "buy"},
        {"amount": 1.0, "side": "sell"},
        {"amount": 1.0, "side": "buy"},
        {"amount": 5.0, "side": "buy"},
    ]
    # avg=(1+1+1+5)/4=2.0, threshold=4.0, large=[5.0 buy]
    # buy_vol=5, sell_vol=0 → 1.0
    assert feed._large_print_direction(trades) == pytest.approx(1.0)


def test_large_print_direction_all_sells_above_threshold():
    feed = make_feed()
    trades = [
        {"amount": 1.0, "side": "buy"},
        {"amount": 1.0, "side": "sell"},
        {"amount": 1.0, "side": "buy"},
        {"amount": 5.0, "side": "sell"},
    ]
    # avg=2.0, threshold=4.0, large=[5.0 sell]
    # buy_vol=0, sell_vol=5 → -1.0
    assert feed._large_print_direction(trades) == pytest.approx(-1.0)


def test_large_print_direction_no_large_prints():
    feed = make_feed()
    # All trades same size — no trade exceeds 2× avg
    trades = [
        {"amount": 2.0, "side": "buy"},
        {"amount": 2.0, "side": "buy"},
        {"amount": 2.0, "side": "sell"},
    ]
    # avg=2.0, threshold=4.0, no trade > 4.0 → returns 0.0
    assert feed._large_print_direction(trades) == pytest.approx(0.0)


def test_large_print_direction_mixed():
    feed = make_feed()
    # 5 small trades of 1.0, 2 large buys of 10.0, 1 large sell of 10.0
    trades = [
        {"amount": 1.0, "side": "buy"},
        {"amount": 1.0, "side": "sell"},
        {"amount": 1.0, "side": "buy"},
        {"amount": 1.0, "side": "sell"},
        {"amount": 1.0, "side": "buy"},
        {"amount": 10.0, "side": "buy"},
        {"amount": 10.0, "side": "buy"},
        {"amount": 10.0, "side": "sell"},
    ]
    # avg=(5*1 + 3*10)/8=35/8=4.375, threshold=8.75
    # large=[10buy,10buy,10sell], buy_vol=20, sell_vol=10, total=30
    # score=(20-10)/30 = 1/3
    result = feed._large_print_direction(trades)
    assert result == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_large_print_direction_empty_trades():
    feed = make_feed()
    assert feed._large_print_direction([]) == pytest.approx(0.0)
