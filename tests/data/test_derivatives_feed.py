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
    feed._prev_oi = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0}
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
    assert 110 <= ttl <= 120


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
async def test_coinglass_returns_values_when_api_key_set():
    """_coinglass_funding_and_oi returns funding and OI delta when the API key is set."""
    feed = make_feed()
    feed._prev_oi = {"okx": 1000.0, "hyperliquid": 0.0, "kraken_futures": 0.0}

    fr_payload = {"data": [
        {"time": 0, "close": "0.003"},
        {"time": 4 * 3600_000, "close": "0.0035"},
    ]}
    oi_payload = {"data": [
        {"exchange": "OKX", "open_interest_quantity": "1050.0"},
    ]}

    mock_resp_fr = AsyncMock()
    mock_resp_fr.__aenter__ = AsyncMock(return_value=mock_resp_fr)
    mock_resp_fr.__aexit__ = AsyncMock(return_value=False)
    mock_resp_fr.json = AsyncMock(return_value=fr_payload)

    mock_resp_oi = AsyncMock()
    mock_resp_oi.__aenter__ = AsyncMock(return_value=mock_resp_oi)
    mock_resp_oi.__aexit__ = AsyncMock(return_value=False)
    mock_resp_oi.json = AsyncMock(return_value=oi_payload)

    call_count = 0

    def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_resp_fr if call_count == 1 else mock_resp_oi

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=mock_get)

    import btc_kalshi_system.data.derivatives_feed as df_module
    with patch.object(df_module, "COINGLASS_API_KEY", "fake-key"), \
         patch("aiohttp.ClientSession", return_value=mock_session):
        curr_funding, trend, oi_delta, okx_partial = await feed._coinglass_funding_and_oi()

    assert curr_funding == pytest.approx(0.0035)
    assert oi_delta == pytest.approx(0.05)   # (1050 - 1000) / 1000
    assert okx_partial is False


@pytest.mark.asyncio
async def test_kraken_exchange_lazy_init():
    """_get_kraken_exchange() lazy-initializes and caches the Kraken ccxt instance."""
    feed = make_feed()
    feed._kraken_exchange = None
    mock_kraken = AsyncMock()
    feed._ccxt_async.kraken.return_value = mock_kraken

    result = await feed._get_kraken_exchange()

    feed._ccxt_async.kraken.assert_called_once()
    assert result is mock_kraken
    # Second call should return the cached instance without re-calling ccxt
    result2 = await feed._get_kraken_exchange()
    assert result2 is mock_kraken
    feed._ccxt_async.kraken.assert_called_once()  # still only once


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


@pytest.mark.asyncio
async def test_hyperliquid_fetcher_returns_normalized_funding_and_oi():
    """_fetch_hyperliquid_funding_and_oi normalizes 1h→8h funding and returns BTC OI."""
    feed = make_feed()
    feed._prev_oi = {"okx": 0.0, "hyperliquid": 1000.0, "kraken_futures": 0.0}

    hl_response = [
        {"universe": [{"name": "BTC", "szDecimals": 5}]},
        [{"funding": "0.0000125", "openInterest": "1100.0", "markPx": "67000.0"}],
    ]

    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.json = AsyncMock(return_value=hl_response)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        funding, oi_delta = await feed._fetch_hyperliquid_funding_and_oi()

    assert funding == pytest.approx(0.0000125 * 8)   # normalized to 8h
    assert oi_delta == pytest.approx(0.10)            # (1100 - 1000) / 1000


@pytest.mark.asyncio
async def test_kraken_futures_fetcher_returns_normalized_funding_and_oi():
    """_fetch_kraken_futures_funding_and_oi converts annualized rate to 8h and computes OI delta."""
    feed = make_feed()
    feed._prev_oi = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 500_000.0}

    kf_response = {
        "tickers": [
            {"symbol": "PF_XBTUSD", "fundingRate": 0.1095, "openInterest": 550_000.0},
            {"symbol": "FI_XBTUSD_250926", "fundingRate": None, "openInterest": 100_000.0},
        ]
    }

    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.json = AsyncMock(return_value=kf_response)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        funding, oi_delta = await feed._fetch_kraken_futures_funding_and_oi()

    # 0.1095 annualized / 1095 periods = 0.0001 per 8h
    assert funding == pytest.approx(0.0001)
    # (550k - 500k) / 500k = 0.10
    assert oi_delta == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_multi_source_averages_when_all_succeed():
    """When all 3 sources return data, funding and oi_delta are averaged."""
    feed = make_feed()
    feed._prev_oi = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0}

    with (
        patch.object(feed, "_fetch_okx_funding_and_oi", new=AsyncMock(return_value=(0.0002, 0.001, 0.05))),
        patch.object(feed, "_fetch_hyperliquid_funding_and_oi", new=AsyncMock(return_value=(0.0004, 0.02))),
        patch.object(feed, "_fetch_kraken_futures_funding_and_oi", new=AsyncMock(return_value=(0.0003, 0.04))),
    ):
        funding, trend, oi_delta, okx_partial = await feed._fetch_funding_and_oi()

    # funding avg: (0.0002 + 0.0004 + 0.0003) / 3
    assert funding == pytest.approx((0.0002 + 0.0004 + 0.0003) / 3)
    # trend only from OKX (HL and KF provide no history-based trend)
    assert trend == pytest.approx(0.001)
    # oi_delta avg: (0.05 + 0.02 + 0.04) / 3
    assert oi_delta == pytest.approx((0.05 + 0.02 + 0.04) / 3)
    assert okx_partial is False


@pytest.mark.asyncio
async def test_multi_source_uses_available_when_okx_fails():
    """When OKX fails, HL and KF results are averaged; okx_partial stays False."""
    feed = make_feed()
    feed._prev_oi = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0}

    with (
        patch.object(feed, "_fetch_okx_funding_and_oi", new=AsyncMock(side_effect=Exception("geo-blocked"))),
        patch.object(feed, "_fetch_hyperliquid_funding_and_oi", new=AsyncMock(return_value=(0.0004, 0.02))),
        patch.object(feed, "_fetch_kraken_futures_funding_and_oi", new=AsyncMock(return_value=(0.0003, 0.04))),
    ):
        funding, trend, oi_delta, okx_partial = await feed._fetch_funding_and_oi()

    assert funding == pytest.approx((0.0004 + 0.0003) / 2)
    assert trend == pytest.approx(0.0)   # no OKX → no history-based trend
    assert oi_delta == pytest.approx((0.02 + 0.04) / 2)
    assert okx_partial is False


@pytest.mark.asyncio
async def test_multi_source_okx_partial_only_when_all_fail():
    """okx_partial=True only when every source throws."""
    feed = make_feed()
    feed._prev_oi = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0}

    with (
        patch.object(feed, "_fetch_okx_funding_and_oi", new=AsyncMock(side_effect=Exception("blocked"))),
        patch.object(feed, "_fetch_hyperliquid_funding_and_oi", new=AsyncMock(side_effect=Exception("timeout"))),
        patch.object(feed, "_fetch_kraken_futures_funding_and_oi", new=AsyncMock(side_effect=Exception("error"))),
    ):
        funding, trend, oi_delta, okx_partial = await feed._fetch_funding_and_oi()

    assert funding == pytest.approx(0.0)
    assert oi_delta == pytest.approx(0.0)
    assert okx_partial is True


@pytest.mark.asyncio
async def test_volume_ratio_falls_back_to_kraken_when_okx_fails():
    """When OKX fetch_ohlcv raises, Kraken spot candles are used for volume ratio."""
    feed = make_feed()
    feed._exchange = AsyncMock()
    feed._exchange.fetch_ohlcv.side_effect = Exception("geo-blocked")

    # 31 candles: 30 historical (avg vol=100) + 1 current (vol=200) → ratio=2.0
    candles = [[0, 0, 0, 0, 0, 100.0]] * 30 + [[0, 0, 0, 0, 0, 200.0]]
    mock_kraken = AsyncMock()
    mock_kraken.fetch_ohlcv = AsyncMock(return_value=candles)
    feed._kraken_exchange = mock_kraken

    ratio = await feed._fetch_volume_ratio()
    assert ratio == pytest.approx(2.0)


def test_fetch_features_includes_macro_correlations(monkeypatch):
    """MacroFeed correlations are merged into the features dict."""
    from btc_kalshi_system.data.macro_feed import MacroFeed
    from unittest.mock import MagicMock

    feed = make_feed_with_mock_accumulator()
    feed._last_slow_fetch = time.time()  # skip slow tier

    # Stub out fast-tier helpers
    async def _zero_liq(): return 0.0
    async def _zero_imbalance(): return 0.0
    monkeypatch.setattr(feed, "_fetch_liquidations", _zero_liq)
    monkeypatch.setattr(feed, "_fetch_okx_spot_imbalance", _zero_imbalance)
    monkeypatch.setattr(feed, "_brti_volatility_1h", lambda: 0.0)

    # Stub MacroFeed
    mock_macro = MagicMock(spec=MacroFeed)
    mock_macro.get_correlations.return_value = {"btc_spx_corr_8d": 0.42, "btc_qqq_corr_8d": 0.38}
    feed._macro_feed = mock_macro

    import asyncio
    features = asyncio.get_event_loop().run_until_complete(feed._fetch_features())

    assert features["btc_spx_corr_8d"] == 0.42
    assert features["btc_qqq_corr_8d"] == 0.38


# ── Helper functions for new method tests ──────────────────────────────────────

def _make_mock_session(mock_data: dict):
    """Return a context-manager AsyncMock that yields mock_data as JSON."""
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=mock_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


async def _mock_liq_call(feed, mock_data: dict) -> float:
    with patch("btc_kalshi_system.data.derivatives_feed.aiohttp.ClientSession", return_value=_make_mock_session(mock_data)):
        return await feed._fetch_liquidations()


async def _mock_books_call(feed, mock_data: dict) -> float:
    with patch("btc_kalshi_system.data.derivatives_feed.aiohttp.ClientSession", return_value=_make_mock_session(mock_data)):
        return await feed._fetch_okx_spot_imbalance()


# ── _fetch_liquidations ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_liquidations_net_norm_short_heavy():
    """More short liquidations → positive liq_net_norm."""
    feed = make_feed()
    now_ms = int(time.time() * 1000)
    mock_data = {
        "code": "0",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "details": [
                {"side": "buy",  "sz": "30", "bkPx": "95000", "ts": str(now_ms - 60_000)},
                {"side": "buy",  "sz": "20", "bkPx": "95000", "ts": str(now_ms - 120_000)},
                {"side": "sell", "sz": "10", "bkPx": "95000", "ts": str(now_ms - 60_000)},
            ]
        }]
    }
    result = await _mock_liq_call(feed, mock_data)
    # short_sz=50, long_sz=10, net=40, total=60 → 40/60
    assert result == pytest.approx(40 / 60, rel=1e-3)


@pytest.mark.asyncio
async def test_fetch_liquidations_below_noise_floor_returns_zero():
    """Total < 10 contracts → return 0.0 (noise floor)."""
    feed = make_feed()
    now_ms = int(time.time() * 1000)
    mock_data = {
        "code": "0",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "details": [
                {"side": "buy",  "sz": "3", "bkPx": "95000", "ts": str(now_ms - 60_000)},
                {"side": "sell", "sz": "2", "bkPx": "95000", "ts": str(now_ms - 60_000)},
            ]
        }]
    }
    result = await _mock_liq_call(feed, mock_data)
    assert result == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_fetch_liquidations_old_entries_excluded():
    """Liquidations older than 15 min are excluded; recent ones still counted."""
    feed = make_feed()
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - 16 * 60_000   # 16 min ago — outside 15-min window
    recent_ms = now_ms - 60_000      # 1 min ago — inside window

    mock_data = {
        "code": "0",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "details": [
                # Old entry — must be excluded
                {"side": "buy", "sz": "50", "bkPx": "95000", "ts": str(old_ms)},
                # Recent entry — must be included
                {"side": "sell", "sz": "20", "bkPx": "95000", "ts": str(recent_ms)},
            ]
        }]
    }
    result = await _mock_liq_call(feed, mock_data)
    # Only recent_ms sell entry qualifies: short_sz=0, long_sz=20, total=20
    # liq_net_norm = (0 - 20) / 20 = -1.0
    assert result == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_fetch_liquidations_returns_zero_on_api_failure():
    """API failure → safe fallback 0.0."""
    feed = make_feed()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(side_effect=Exception("timeout"))
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("btc_kalshi_system.data.derivatives_feed.aiohttp.ClientSession", return_value=mock_session):
        result = await feed._fetch_liquidations()
    assert result == pytest.approx(0.0)


# ── _fetch_eth_direction ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_eth_direction_up_when_close_above_open():
    feed = make_feed()
    ohlcv = [
        [1_000_000_000, 3000.0, 3100.0, 2950.0, 3080.0, 100.0],  # closed candle, up
        [1_000_000_900, 3080.0, 3150.0, 3050.0, 3120.0,  50.0],  # currently open
    ]
    feed._exchange.fetch_ohlcv = AsyncMock(return_value=ohlcv)
    result = await feed._fetch_eth_direction()
    assert result == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_fetch_eth_direction_down_when_close_below_open():
    feed = make_feed()
    ohlcv = [
        [1_000_000_000, 3000.0, 3100.0, 2900.0, 2950.0, 100.0],  # closed candle, down
        [1_000_000_900, 2950.0, 3000.0, 2900.0, 2980.0,  50.0],
    ]
    feed._exchange.fetch_ohlcv = AsyncMock(return_value=ohlcv)
    result = await feed._fetch_eth_direction()
    assert result == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_fetch_eth_direction_returns_half_on_insufficient_data():
    feed = make_feed()
    feed._exchange.fetch_ohlcv = AsyncMock(return_value=[[1_000_000_000, 3000.0, 3100.0, 2900.0, 3080.0, 100.0]])
    result = await feed._fetch_eth_direction()
    assert result == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_fetch_eth_direction_returns_half_on_failure():
    feed = make_feed()
    feed._exchange.fetch_ohlcv = AsyncMock(side_effect=Exception("network error"))
    result = await feed._fetch_eth_direction()
    assert result == pytest.approx(0.5)


# ── _fetch_okx_spot_imbalance ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_okx_spot_imbalance_bid_heavy():
    feed = make_feed()
    mock_data = {
        "code": "0",
        "data": [{
            "bids": [["95000", "3.0", "0", "1"]],
            "asks": [["95100", "1.0", "0", "1"]],
            "ts": "1234567890000",
        }]
    }
    result = await _mock_books_call(feed, mock_data)
    assert result == pytest.approx(0.5)   # (3-1)/(3+1)


@pytest.mark.asyncio
async def test_fetch_okx_spot_imbalance_ask_heavy():
    feed = make_feed()
    mock_data = {
        "code": "0",
        "data": [{
            "bids": [["95000", "1.0", "0", "1"]],
            "asks": [["95100", "3.0", "0", "1"]],
            "ts": "1234567890000",
        }]
    }
    result = await _mock_books_call(feed, mock_data)
    assert result == pytest.approx(-0.5)  # (1-3)/(1+3)


@pytest.mark.asyncio
async def test_fetch_okx_spot_imbalance_returns_zero_on_empty_book():
    feed = make_feed()
    mock_data = {"code": "0", "data": [{"bids": [], "asks": [], "ts": "123"}]}
    result = await _mock_books_call(feed, mock_data)
    assert result == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_fetch_okx_spot_imbalance_returns_zero_on_api_failure():
    feed = make_feed()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(side_effect=Exception("timeout"))
    mock_session.__aexit__ = AsyncMock(return_value=False)
    with patch("btc_kalshi_system.data.derivatives_feed.aiohttp.ClientSession", return_value=mock_session):
        result = await feed._fetch_okx_spot_imbalance()
    assert result == pytest.approx(0.0)


# ── Slow-tier cache tests ──────────────────────────────────────────────────────

def make_feed_with_mock_accumulator():
    """Instantiate DerivativesFeed without calling __init__; inject mock accumulator."""
    feed = DerivativesFeed.__new__(DerivativesFeed)
    feed._redis = fakeredis.FakeRedis()
    feed._macro_feed = MagicMock()
    feed._macro_feed.get_correlations.return_value = {}
    feed._exchange = MagicMock()
    feed._exchange_name = "okx"
    feed._prev_oi = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0, "deribit": 0.0}
    feed._kraken_exchange = None
    feed._last_slow_fetch = 0.0
    feed._cached_funding_result = (0.001, 0.0, 0.0, False)
    feed._cached_eth_dir = 0.5
    feed._cached_volume_ratio = 1.0
    acc = MagicMock()
    acc.cvd_normalized = 0.3
    acc.large_print_direction = 0.0
    acc.last_price = 95000.0
    acc.is_stale = False
    feed._cvd_accumulator = acc
    return feed


async def test_slow_tier_not_refetched_within_60s():
    """When _last_slow_fetch < 60s ago, funding/OI/ETH/volume are read from cache."""
    feed = make_feed_with_mock_accumulator()
    feed._last_slow_fetch = time.time()

    feed._fetch_funding_and_oi = AsyncMock(return_value=(0.001, 0.0, 0.0, False))
    feed._fetch_eth_direction   = AsyncMock(return_value=0.5)
    feed._fetch_volume_ratio    = AsyncMock(return_value=1.0)
    feed._fetch_liquidations       = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    await feed._fetch_features()

    feed._fetch_funding_and_oi.assert_not_called()
    feed._fetch_eth_direction.assert_not_called()
    feed._fetch_volume_ratio.assert_not_called()


async def test_slow_tier_refetched_after_60s():
    """When _last_slow_fetch > 60s ago, funding/OI/ETH/volume are re-fetched."""
    feed = make_feed_with_mock_accumulator()
    feed._last_slow_fetch = time.time() - 61

    feed._fetch_funding_and_oi = AsyncMock(return_value=(0.002, 0.001, 0.01, False))
    feed._fetch_eth_direction   = AsyncMock(return_value=1.0)
    feed._fetch_volume_ratio    = AsyncMock(return_value=1.5)
    feed._fetch_liquidations       = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    await feed._fetch_features()

    feed._fetch_funding_and_oi.assert_called_once()
    feed._fetch_eth_direction.assert_called_once()
    feed._fetch_volume_ratio.assert_called_once()


async def test_cvd_read_from_accumulator_not_http():
    """CVD comes from the accumulator (0.3); no HTTP trade fetch."""
    feed = make_feed_with_mock_accumulator()
    feed._last_slow_fetch = time.time()
    feed._fetch_liquidations       = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    features = await feed._fetch_features()

    assert features["cvd_normalized"] == pytest.approx(0.3)


async def test_cvd_stale_flag_set_when_accumulator_stale():
    """When accumulator.is_stale=True, features dict contains _cvd_stale=True."""
    feed = make_feed_with_mock_accumulator()
    feed._cvd_accumulator.is_stale = True
    feed._last_slow_fetch = time.time()
    feed._fetch_liquidations       = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    features = await feed._fetch_features()
    assert features.get("_cvd_stale") is True
