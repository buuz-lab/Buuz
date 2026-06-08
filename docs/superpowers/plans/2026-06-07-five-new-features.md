# Five New Regime Model Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 new regime model features — OKX liquidation pressure (`liq_net_norm`), ETH prior 15-min candle direction (`eth_direction_15min`), OKX spot order book imbalance (`okx_spot_imbalance`), PCR rate-of-change (`pcr_delta`), skew rate-of-change (`skew_delta`), and DeepSeek directional probability (`deepseek_dir_prob`) — growing the feature set from 33 to 39.

**Architecture:** Features 1–3 are computed in `DerivativesFeed._fetch_features()` and written to the `regime:features` Redis key, which main.py merges into `ctx`. Features 4–5 are computed from instance state in `DeribitOptionsFeed` and written to `options:features`, which main.py also merges into `ctx`. Feature 6 is extracted from the existing DeepSeek API response (new JSON field) and cached as `_last_deepseek_dir_prob` in `SignalFusionEngine`. All 6 are injected into `_regime_features()` and added to `_FEATURE_ORDER`. Historical rows get `NULL` / NaN (XGBoost handles missing values natively).

**Tech Stack:** aiohttp (OKX REST), ccxt (ETH OHLCV), asyncio.gather, fakeredis + pytest-asyncio (tests)

---

## File Map

| File | Change |
|------|--------|
| `btc_kalshi_system/data/derivatives_feed.py` | Add `_fetch_liquidations()`, `_fetch_eth_direction()`, `_fetch_okx_spot_imbalance()`; wire into `_fetch_features()` |
| `btc_kalshi_system/data/deribit_options_feed.py` | Add `_prev_pcr_oi` / `_prev_skew_25d` instance state; compute `pcr_delta`/`skew_delta` in `_compute_features()` |
| `btc_kalshi_system/models/deepseek_parser.py` | Add `dir_prob_up` to prompt, `_REQUIRED_KEYS`, `_parse_response()`, `NEUTRAL_DEFAULT`, `SAFE_DEFAULT` |
| `btc_kalshi_system/signal/fusion.py` | Add `_last_deepseek_dir_prob`; set in `get_signal()`; inject all 6 in `_regime_features()` |
| `btc_kalshi_system/models/regime_model.py` | Append 6 entries to `_FEATURE_ORDER` |
| `main.py` | Add 6 `_CANDLE_FEATURES_COLUMN_MIGRATIONS` entries; add 6 columns to candle_features INSERT |
| `tests/data/test_derivatives_feed.py` | New tests for all three new methods |
| `tests/data/test_deribit_options_feed.py` | New tests for delta computation |
| `tests/models/test_deepseek_parser.py` | New tests for `dir_prob_up` parsing |
| `tests/signal/test_feature_order.py` | Update count assertion 33 → 39 |

---

## Task 1: DerivativesFeed — Three New Fetch Methods

**Files:**
- Modify: `btc_kalshi_system/data/derivatives_feed.py`
- Test: `tests/data/test_derivatives_feed.py`

### Background

`DerivativesFeed._fetch_features()` uses `asyncio.gather()` to fan out I/O calls and returns a dict that gets JSON-serialized into Redis `regime:features`. We add three new methods to the same gather and include their results in the dict. Each method returns a single float with a safe fallback (0.0 or 0.5) on any exception.

- **`_fetch_liquidations()`** → `liq_net_norm` (float, -1 to +1)
  - Calls OKX public REST `/api/v5/public/liquidation-orders?instType=SWAP&instId=BTC-USDT-SWAP&state=filled&limit=100`
  - `side="buy"` = short position liquidated (forced buy = upward cascade pressure) → `short_sz`
  - `side="sell"` = long position liquidated (forced sell = downward cascade pressure) → `long_sz`
  - Filter by `ts` to last 15 min; if total < 10 contracts, return 0.0 (noise floor)
  - Returns `(short_sz - long_sz) / total` where total = short_sz + long_sz

- **`_fetch_eth_direction()`** → `eth_direction_15min` (0.0 = down, 1.0 = up, 0.5 = unknown)
  - Calls `self._exchange.fetch_ohlcv("ETH/USDT:USDT", "15m", limit=3)` via ccxt
  - `ohlcv[-2]` is the last CLOSED 15-min candle (index -1 may be the currently open one)
  - Returns `1.0` if `close > open`, `0.0` otherwise; `0.5` on any failure

- **`_fetch_okx_spot_imbalance()`** → `okx_spot_imbalance` (float, -1 to +1)
  - Calls OKX public REST `/api/v5/market/books?instId=BTC-USDT&sz=5` (top 5 levels)
  - `bid_depth = sum(qty for level in bids)`, `ask_depth = sum(qty for level in asks)`
  - Returns `(bid_depth - ask_depth) / (bid_depth + ask_depth)`; 0.0 if total near zero

---

- [ ] **Step 1.1: Write failing tests for `_fetch_liquidations()`**

Add to `tests/data/test_derivatives_feed.py`:

```python
import time
from unittest.mock import AsyncMock, patch

import pytest


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
    """Liquidations older than 15 min are excluded."""
    feed = make_feed()
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - 16 * 60_000  # 16 min ago — outside window
    mock_data = {
        "code": "0",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "details": [
                {"side": "buy", "sz": "100", "bkPx": "95000", "ts": str(old_ms)},
            ]
        }]
    }
    result = await _mock_liq_call(feed, mock_data)
    assert result == pytest.approx(0.0)  # total=0 → noise floor → 0.0


@pytest.mark.asyncio
async def test_fetch_liquidations_returns_zero_on_api_failure():
    """API failure → safe fallback 0.0."""
    feed = make_feed()

    async def broken_session():
        raise Exception("timeout")

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(side_effect=Exception("timeout"))
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await feed._fetch_liquidations()
    assert result == pytest.approx(0.0)


def _make_mock_session(mock_data: dict):
    """Helper: return a context-manager AsyncMock that yields mock_data as JSON."""
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
    with patch("aiohttp.ClientSession", return_value=_make_mock_session(mock_data)):
        return await feed._fetch_liquidations()
```

- [ ] **Step 1.2: Run to confirm they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -k "liquidat" -v 2>&1 | tail -20
```

Expected: `ERRORS` or `AttributeError: '_fetch_liquidations'`

- [ ] **Step 1.3: Implement `_fetch_liquidations()`**

At the top of `btc_kalshi_system/data/derivatives_feed.py`, after existing constants, add:

```python
_OKX_LIQ_URL   = "https://www.okx.com/api/v5/public/liquidation-orders"
_OKX_BOOKS_URL  = "https://www.okx.com/api/v5/market/books"
_LIQ_WINDOW_MS  = 15 * 60 * 1000  # 15 minutes
_LIQ_NOISE_FLOOR = 10.0            # contracts below which we ignore
```

Add method to `DerivativesFeed` class:

```python
async def _fetch_liquidations(self) -> float:
    """OKX BTC-USDT-SWAP liquidations from the last 15 min.

    Returns liq_net_norm = (short_liq_sz - long_liq_sz) / total_sz.
    Positive = more shorts liquidated = upward cascade pressure.
    Negative = more longs liquidated = downward cascade pressure.
    Returns 0.0 when quiet (< 10 contracts total) or on any failure.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        params = {
            "instType": "SWAP",
            "instId": "BTC-USDT-SWAP",
            "state": "filled",
            "limit": "100",
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(_OKX_LIQ_URL, params=params) as resp:
                data = await resp.json()
        cutoff_ms = time.time() * 1000 - _LIQ_WINDOW_MS
        short_sz = 0.0
        long_sz = 0.0
        for record in data.get("data", []):
            for detail in record.get("details", []):
                ts = float(detail.get("ts", 0))
                if ts < cutoff_ms:
                    continue
                sz = float(detail.get("sz", 0))
                if detail.get("side") == "buy":
                    short_sz += sz
                else:
                    long_sz += sz
        total = short_sz + long_sz
        if total < _LIQ_NOISE_FLOOR:
            return 0.0
        return (short_sz - long_sz) / total
    except Exception as exc:
        logger.debug(f"DerivativesFeed: liquidations fetch failed — {exc}")
        return 0.0
```

- [ ] **Step 1.4: Run liquidation tests to confirm they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -k "liquidat" -v 2>&1 | tail -20
```

Expected: all 4 pass.

- [ ] **Step 1.5: Write failing tests for `_fetch_eth_direction()`**

Add to `tests/data/test_derivatives_feed.py`:

```python
@pytest.mark.asyncio
async def test_fetch_eth_direction_up_when_close_above_open():
    feed = make_feed()
    # ohlcv row: [timestamp, open, high, low, close, volume]
    # index -2 is the last CLOSED 15-min candle
    ohlcv = [
        [1_000_000_000, 3000.0, 3100.0, 2950.0, 3080.0, 100.0],  # closed, up
        [1_000_000_900, 3080.0, 3150.0, 3050.0, 3120.0,  50.0],  # current open
    ]
    feed._exchange.fetch_ohlcv = AsyncMock(return_value=ohlcv)
    result = await feed._fetch_eth_direction()
    assert result == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_fetch_eth_direction_down_when_close_below_open():
    feed = make_feed()
    ohlcv = [
        [1_000_000_000, 3000.0, 3100.0, 2900.0, 2950.0, 100.0],  # closed, down
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
```

- [ ] **Step 1.6: Run to confirm they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -k "eth_dir" -v 2>&1 | tail -10
```

- [ ] **Step 1.7: Implement `_fetch_eth_direction()`**

Add to `DerivativesFeed` class:

```python
async def _fetch_eth_direction(self) -> float:
    """Previous closed ETH/USDT 15-min candle direction.

    Returns 1.0 (up), 0.0 (down), or 0.5 (unknown / insufficient data).
    Uses the ccxt exchange that is already resolved for funding/OI calls.
    """
    try:
        ohlcv = await self._exchange.fetch_ohlcv("ETH/USDT:USDT", "15m", limit=3)
        if len(ohlcv) < 2:
            return 0.5
        prev = ohlcv[-2]   # index -1 may be the currently-open candle
        return 1.0 if prev[4] > prev[1] else 0.0   # close > open
    except Exception as exc:
        logger.debug(f"DerivativesFeed: ETH direction fetch failed — {exc}")
        return 0.5
```

- [ ] **Step 1.8: Run ETH direction tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -k "eth_dir" -v 2>&1 | tail -10
```

Expected: all 4 pass.

- [ ] **Step 1.9: Write failing tests for `_fetch_okx_spot_imbalance()`**

Add to `tests/data/test_derivatives_feed.py`:

```python
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
    # bid_depth=3.0, ask_depth=1.0 → (3-1)/(3+1) = 0.5
    assert result == pytest.approx(0.5)


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
    # (1-3)/(1+3) = -0.5
    assert result == pytest.approx(-0.5)


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
    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await feed._fetch_okx_spot_imbalance()
    assert result == pytest.approx(0.0)


async def _mock_books_call(feed, mock_data: dict) -> float:
    with patch("aiohttp.ClientSession", return_value=_make_mock_session(mock_data)):
        return await feed._fetch_okx_spot_imbalance()
```

- [ ] **Step 1.10: Run to confirm they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -k "spot_imbalance" -v 2>&1 | tail -10
```

- [ ] **Step 1.11: Implement `_fetch_okx_spot_imbalance()`**

Add to `DerivativesFeed` class:

```python
async def _fetch_okx_spot_imbalance(self) -> float:
    """OKX spot BTC/USDT order book imbalance (top 5 levels).

    Returns (bid_depth - ask_depth) / total_depth.
    +1 = all bids (buy pressure), -1 = all asks (sell pressure), 0 = balanced.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        params = {"instId": "BTC-USDT", "sz": "5"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(_OKX_BOOKS_URL, params=params) as resp:
                data = await resp.json()
        book = data.get("data", [{}])[0]
        bid_depth = sum(float(b[1]) for b in book.get("bids", []))
        ask_depth = sum(float(a[1]) for a in book.get("asks", []))
        total = bid_depth + ask_depth
        if total < 1e-8:
            return 0.0
        return (bid_depth - ask_depth) / total
    except Exception as exc:
        logger.debug(f"DerivativesFeed: spot imbalance fetch failed — {exc}")
        return 0.0
```

- [ ] **Step 1.12: Wire all three into `_fetch_features()`**

Replace the current `_fetch_features()` method:

```python
async def _fetch_features(self) -> dict:
    results = await asyncio.gather(
        self._fetch_funding_and_oi(),
        self._fetch_trades_data(),
        self._fetch_volume_ratio(),
        self._fetch_liquidations(),
        self._fetch_eth_direction(),
        self._fetch_okx_spot_imbalance(),
    )
    curr_funding, trend, oi_delta, okx_partial = results[0]
    cvd, basis, large_print, trades_available = results[1]
    volume_ratio = results[2]
    liq_net_norm = results[3]
    eth_direction_15min = results[4]
    okx_spot_imbalance = results[5]
    vol = self._brti_volatility_1h()
    fg = fetch_fear_greed(self._redis)
    features: dict = {
        "funding_rate":          curr_funding,
        "funding_rate_trend":    trend,
        "oi_delta_pct":          oi_delta,
        "cvd_normalized":        cvd,
        "basis_spread_pct":      basis,
        "brti_volatility_1h":    vol,
        "large_print_direction": large_print,
        "volume_ratio_1h":       volume_ratio,
        "fear_greed_value":      fg["value"] if fg else None,
        "fear_greed_label":      fg["label"] if fg else None,
        "liq_net_norm":          liq_net_norm,
        "eth_direction_15min":   eth_direction_15min,
        "okx_spot_imbalance":    okx_spot_imbalance,
    }
    macro = self._macro_feed.get_correlations()
    features.update(macro)
    if okx_partial:
        features["_okx_partial"] = True
    if not trades_available:
        features["_cvd_stale"] = True
    return features
```

- [ ] **Step 1.13: Run all derivatives_feed tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 1.14: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/data/derivatives_feed.py tests/data/test_derivatives_feed.py && git commit -m "feat: add liq_net_norm, eth_direction_15min, okx_spot_imbalance to DerivativesFeed"
```

---

## Task 2: DeribitOptionsFeed — PCR and Skew Delta

**Files:**
- Modify: `btc_kalshi_system/data/deribit_options_feed.py`
- Test: `tests/data/test_deribit_options_feed.py`

### Background

`DeribitOptionsFeed` refreshes every 5 minutes. It writes `{atm_iv, pcr_oi, term_structure_slope, skew_25d}` to `options:features`. We add `pcr_delta` and `skew_delta` as 5-minute rates of change by storing the previous call's values as instance state and computing the delta before writing.

- First call after startup: delta is `0.0` (no previous value available)
- `pcr_delta > 0` = PCR rising = more put-buying = increasing downside hedging demand
- `skew_delta > 0` = skew steepening = puts getting more expensive relative to calls

---

- [ ] **Step 2.1: Write failing tests for delta computation**

Add to `tests/data/test_deribit_options_feed.py`:

```python
def test_compute_features_includes_pcr_delta_and_skew_delta():
    """After the first call, both delta features are present in the returned dict."""
    feed = DeribitOptionsFeed.__new__(DeribitOptionsFeed)
    feed._redis = MagicMock()
    feed._prev_pcr_oi = 1.0
    feed._prev_skew_25d = 0.0
    # Build a minimal instruments list that produces deterministic pcr/skew values.
    # We'll call _compute_features with a known mock chain and check the deltas.
    # Use the existing _minimal_chain fixture helper from this test file if it exists,
    # otherwise use _good_instruments() defined below.
    instruments = _good_instruments()
    result = feed._compute_features(instruments)
    assert "pcr_delta" in result
    assert "skew_delta" in result


def test_pcr_delta_is_change_from_previous():
    """pcr_delta = current_pcr - prev_pcr_oi."""
    feed = DeribitOptionsFeed.__new__(DeribitOptionsFeed)
    feed._redis = MagicMock()
    feed._prev_pcr_oi = 0.8       # previous value
    feed._prev_skew_25d = 0.0
    instruments = _good_instruments()
    result = feed._compute_features(instruments)
    # pcr_delta must equal the computed pcr_oi minus 0.8
    assert result["pcr_delta"] == pytest.approx(result["pcr_oi"] - 0.8, abs=1e-6)


def test_skew_delta_is_change_from_previous():
    """skew_delta = current_skew - prev_skew_25d."""
    feed = DeribitOptionsFeed.__new__(DeribitOptionsFeed)
    feed._redis = MagicMock()
    feed._prev_pcr_oi = 1.0
    feed._prev_skew_25d = 5.0     # previous value
    instruments = _good_instruments()
    result = feed._compute_features(instruments)
    assert result["skew_delta"] == pytest.approx(result["skew_25d"] - 5.0, abs=1e-6)


def test_prev_values_updated_after_compute():
    """Instance state is updated to current values after each compute call."""
    feed = DeribitOptionsFeed.__new__(DeribitOptionsFeed)
    feed._redis = MagicMock()
    feed._prev_pcr_oi = 1.0
    feed._prev_skew_25d = 0.0
    instruments = _good_instruments()
    result = feed._compute_features(instruments)
    assert feed._prev_pcr_oi == pytest.approx(result["pcr_oi"])
    assert feed._prev_skew_25d == pytest.approx(result["skew_25d"])
```

Note: `_good_instruments()` is a helper that may already exist in the file as a fixture that produces a valid option chain. If it doesn't exist, add:

```python
def _good_instruments():
    """Minimal valid BTC options chain: one 7-day expiry with 4 strikes."""
    from datetime import datetime, timezone, timedelta
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    expiry_str = expiry.strftime("%d%b%y").upper()
    underlying = 95000.0
    iv = 55.0
    return [
        {"instrument_name": f"BTC-{expiry_str}-94000-P", "mark_iv": iv + 2, "open_interest": 100, "underlying_price": underlying},
        {"instrument_name": f"BTC-{expiry_str}-95000-C", "mark_iv": iv,     "open_interest": 150, "underlying_price": underlying},
        {"instrument_name": f"BTC-{expiry_str}-95000-P", "mark_iv": iv + 1, "open_interest": 120, "underlying_price": underlying},
        {"instrument_name": f"BTC-{expiry_str}-96000-C", "mark_iv": iv - 1, "open_interest":  80, "underlying_price": underlying},
    ]
```

- [ ] **Step 2.2: Run to confirm they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_deribit_options_feed.py -k "delta" -v 2>&1 | tail -15
```

Expected: `AttributeError: '_prev_pcr_oi'` or similar.

- [ ] **Step 2.3: Add instance state to `DeribitOptionsFeed.__init__`**

The class currently has no `__init__`. Add one:

```python
def __init__(self, redis_url: str = REDIS_URL) -> None:
    self._redis = redis.from_url(redis_url)
    self._prev_pcr_oi: float = 1.0    # neutral default (ratio)
    self._prev_skew_25d: float = 0.0  # neutral default
```

- [ ] **Step 2.4: Add delta computation to `_compute_features()`**

In `_compute_features()`, at the very end just before the `return` statement, add:

```python
            # Rate-of-change features: 5-min delta (one refresh cycle)
            pcr_delta = pcr_oi - self._prev_pcr_oi
            skew_delta = skew_25d - self._prev_skew_25d
            self._prev_pcr_oi = pcr_oi
            self._prev_skew_25d = skew_25d

            return {
                "atm_iv": near_iv,
                "pcr_oi": pcr_oi,
                "term_structure_slope": term_structure_slope,
                "skew_25d": skew_25d,
                "pcr_delta": pcr_delta,
                "skew_delta": skew_delta,
            }
```

Replace the existing `return {...}` block inside `_compute_features()` (at line ~122) with the above. The `except` block's fallback return also needs the new keys:

```python
        except Exception as exc:
            logger.warning(f"DeribitOptionsFeed: feature computation failed — {exc}")
            return {"atm_iv": None, "pcr_oi": 1.0, "term_structure_slope": 0.0, "skew_25d": 0.0,
                    "pcr_delta": 0.0, "skew_delta": 0.0}
```

- [ ] **Step 2.5: Run delta tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_deribit_options_feed.py -k "delta" -v 2>&1 | tail -15
```

Expected: all 4 pass.

- [ ] **Step 2.6: Run full deribit_options_feed test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_deribit_options_feed.py -v 2>&1 | tail -20
```

Expected: all existing tests still pass (no regression on `atm_iv`, `pcr_oi`, etc.).

- [ ] **Step 2.7: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/data/deribit_options_feed.py tests/data/test_deribit_options_feed.py && git commit -m "feat: add pcr_delta and skew_delta to DeribitOptionsFeed"
```

---

## Task 3: DeepSeek Directional Probability

**Files:**
- Modify: `btc_kalshi_system/models/deepseek_parser.py`
- Test: `tests/models/test_deepseek_parser.py`

### Background

The existing DeepSeek call produces a regime label. We add `dir_prob_up` — the model's explicit probability that BTC closes higher at the end of the next 15-minute candle. This is added to the JSON output spec in the prompt, to `_REQUIRED_KEYS`, and extracted/validated in `_parse_response()`. The field is also added to `NEUTRAL_DEFAULT` (0.5 = no directional view) and `SAFE_DEFAULT` (0.5).

The note in the file header says "LLM outputs are poorly calibrated for numeric prediction" — this is why `dir_prob_up` goes into the regime model as a FEATURE (where XGBoost learns its weight) rather than directly driving trades.

---

- [ ] **Step 3.1: Write failing tests for `dir_prob_up`**

Add to `tests/models/test_deepseek_parser.py`. First, update `_good_response()` to include `dir_prob_up`:

```python
def _good_response() -> str:
    return json.dumps({
        "regime": "trending_up",
        "confidence": 0.72,
        "suppress_trading": False,
        "suppress_reason": None,
        "notes": "Strong ETF inflow with positive funding.",
        "dir_prob_up": 0.68,
    })
```

Then add tests:

```python
def test_good_response_includes_dir_prob_up():
    """Successful parse returns dir_prob_up as a float."""
    parser = DeepSeekContextParser(api_key="test-key")
    with patch.object(parser, "_call_api", return_value=_good_response()):
        result = parser.get_current_context(_good_context())
    assert "dir_prob_up" in result
    assert result["dir_prob_up"] == pytest.approx(0.68)


def test_response_missing_dir_prob_up_returns_safe_default():
    """Missing dir_prob_up field → _parse_response returns None → SAFE_DEFAULT."""
    parser = DeepSeekContextParser(api_key="test-key")
    bad_response = json.dumps({
        "regime": "trending_up",
        "confidence": 0.72,
        "suppress_trading": False,
        "suppress_reason": None,
        "notes": "Strong ETF inflow.",
        # dir_prob_up absent
    })
    with patch.object(parser, "_call_api", return_value=bad_response):
        result = parser.get_current_context(_good_context())
    assert result == SAFE_DEFAULT


def test_dir_prob_up_out_of_range_returns_safe_default():
    """dir_prob_up > 1.0 is invalid → SAFE_DEFAULT."""
    parser = DeepSeekContextParser(api_key="test-key")
    bad_response = json.dumps({
        "regime": "trending_up",
        "confidence": 0.72,
        "suppress_trading": False,
        "suppress_reason": None,
        "notes": "ok",
        "dir_prob_up": 1.5,   # out of range
    })
    with patch.object(parser, "_call_api", return_value=bad_response):
        result = parser.get_current_context(_good_context())
    assert result == SAFE_DEFAULT


def test_neutral_default_has_dir_prob_up():
    """NEUTRAL_DEFAULT must include dir_prob_up = 0.5."""
    from btc_kalshi_system.models.deepseek_parser import NEUTRAL_DEFAULT
    assert "dir_prob_up" in NEUTRAL_DEFAULT
    assert NEUTRAL_DEFAULT["dir_prob_up"] == pytest.approx(0.5)


def test_safe_default_has_dir_prob_up():
    """SAFE_DEFAULT must include dir_prob_up = 0.5."""
    from btc_kalshi_system.models.deepseek_parser import SAFE_DEFAULT
    assert "dir_prob_up" in SAFE_DEFAULT
    assert SAFE_DEFAULT["dir_prob_up"] == pytest.approx(0.5)
```

- [ ] **Step 3.2: Run to confirm they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_deepseek_parser.py -k "dir_prob" -v 2>&1 | tail -15
```

Expected: failures because `dir_prob_up` is not in the returned dict.

- [ ] **Step 3.3: Update `NEUTRAL_DEFAULT` and `SAFE_DEFAULT`**

In `deepseek_parser.py`, update both dicts:

```python
NEUTRAL_DEFAULT: dict[str, Any] = {
    "regime": "ranging",
    "confidence": 0.0,
    "suppress_trading": False,
    "suppress_reason": None,
    "notes": "DeepSeek unavailable — using neutral fallback so signals are not shrunk.",
    "dir_prob_up": 0.5,
}

SAFE_DEFAULT: dict[str, Any] = {
    "regime": "high_uncertainty",
    "confidence": 0.0,
    "suppress_trading": False,
    "suppress_reason": "deepseek_unavailable",
    "notes": "Falling back to safe default — DeepSeek call failed or returned malformed data.",
    "dir_prob_up": 0.5,
}
```

- [ ] **Step 3.4: Update `_REQUIRED_KEYS`**

```python
_REQUIRED_KEYS = ("regime", "confidence", "suppress_trading", "suppress_reason", "notes", "dir_prob_up")
```

- [ ] **Step 3.5: Update `_parse_response()` to extract and validate `dir_prob_up`**

Inside `_parse_response()`, after the `suppress_trading` validation block and before the final `return` statement, add:

```python
        try:
            dir_prob_up = float(parsed["dir_prob_up"])
        except (TypeError, ValueError):
            return None
        if not (0.0 <= dir_prob_up <= 1.0):
            return None
```

Update the `return` at the end of `_parse_response()`:

```python
        return {
            "regime": parsed["regime"],
            "confidence": confidence,
            "suppress_trading": parsed["suppress_trading"],
            "suppress_reason": parsed.get("suppress_reason"),
            "notes": str(parsed.get("notes", "")),
            "dir_prob_up": dir_prob_up,
        }
```

- [ ] **Step 3.6: Update `_PROMPT_TEMPLATE` to request `dir_prob_up`**

After the existing `"notes": "one sentence max"` line in the output JSON spec, add `dir_prob_up`. Replace the `Output exactly this JSON:` block:

```python
Output exactly this JSON:
{{
  "regime": "trending_up" | "trending_down" | "ranging" | "high_uncertainty",
  "confidence": 0.0-1.0,
  "suppress_trading": true | false,
  "suppress_reason": "string or null",
  "notes": "one sentence max",
  "dir_prob_up": 0.0-1.0
}}

dir_prob_up is your probability (0.0-1.0) that BTC closes HIGHER at the end of the
next 15-minute candle, based on current CVD, funding, options flow, and momentum.
0.5 = no directional view. This is independent of your regime classification.
```

- [ ] **Step 3.7: Run all deepseek_parser tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_deepseek_parser.py -v 2>&1 | tail -25
```

Expected: all pass. Note: `test_returned_dict_has_all_required_keys` now also verifies `dir_prob_up` is present because `_REQUIRED_KEYS` drives both the parser and the test implicitly via the good_response fixture — but the test checks the 5 original keys by name. Update that test if it hardcodes the key list:

```python
def test_returned_dict_has_all_required_keys():
    parser = DeepSeekContextParser(api_key="test-key")
    with patch.object(parser, "_call_api", return_value=_good_response()):
        result = parser.get_current_context(_good_context())
    for key in ("regime", "confidence", "suppress_trading", "suppress_reason", "notes", "dir_prob_up"):
        assert key in result
```

- [ ] **Step 3.8: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/models/deepseek_parser.py tests/models/test_deepseek_parser.py && git commit -m "feat: add dir_prob_up to DeepSeek directional prompt and parser"
```

---

## Task 4: Wire All Six Features Into Fusion, Regime Model, and Database

**Files:**
- Modify: `btc_kalshi_system/signal/fusion.py`
- Modify: `btc_kalshi_system/models/regime_model.py`
- Modify: `main.py`
- Test: `tests/signal/test_feature_order.py`

### Background

This task makes all six features visible to the regime model. Three steps:

1. **fusion.py**: Store `_last_deepseek_dir_prob` in `__init__`; set it from the DeepSeek response in `get_signal()`; return all 6 new features from `_regime_features()`
2. **regime_model.py**: Append 6 names to `_FEATURE_ORDER` (order matters — must match fusion dict key order)
3. **main.py**: Add 6 `_CANDLE_FEATURES_COLUMN_MIGRATIONS` entries + 6 INSERT column values

The `test_feature_order.py` test enforces that `_FEATURE_ORDER`, `train_regime._FEATURE_COLS`, and `fusion._regime_features()` keys are all identical and in the same order — all three must be updated atomically.

---

- [ ] **Step 4.1: Update `test_feature_order.py` to expect 39 features**

In `tests/signal/test_feature_order.py`, find and update:

```python
def test_feature_order_all_three_match():
    """All three sources must be identical including ORDER (not just membership)."""
    fusion_keys = _get_fusion_feature_keys()
    assert _FEATURE_ORDER == _FEATURE_COLS == fusion_keys
    assert len(_FEATURE_ORDER) == 39  # was 33; added liq_net_norm, eth_direction_15min, okx_spot_imbalance, pcr_delta, skew_delta, deepseek_dir_prob
```

- [ ] **Step 4.2: Run feature_order test to confirm it now fails (expected)**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_feature_order.py -v 2>&1 | tail -15
```

Expected: `AssertionError: ... len == 33 != 39`

- [ ] **Step 4.3: Add `_last_deepseek_dir_prob` to `SignalFusionEngine.__init__`**

In `fusion.py`, in `__init__`, after `self._last_kalshi_early_drift`:

```python
        self._last_deepseek_dir_prob: float = 0.5   # 0.5 = no directional view yet
```

- [ ] **Step 4.4: Set `_last_deepseek_dir_prob` in `get_signal()`**

In `get_signal()`, the line `deepseek_regime = ds["regime"]` is followed immediately by the kronos_raw checks. After `deepseek_regime = ds["regime"]`, add:

```python
        self._last_deepseek_dir_prob = float(ds.get("dir_prob_up", 0.5))
```

- [ ] **Step 4.5: Add 6 new features to `_regime_features()`**

In `_regime_features()`, in the `features = { ... }` dict (starting at line ~500), add 6 entries after `"kalshi_early_drift": self._last_kalshi_early_drift,`:

```python
            # New features — session 39
            "liq_net_norm":          float(ctx.get("liq_net_norm") or 0.0),
            "eth_direction_15min":   float(ctx.get("eth_direction_15min") if ctx.get("eth_direction_15min") is not None else 0.5),
            "okx_spot_imbalance":    float(ctx.get("okx_spot_imbalance") or 0.0),
            "pcr_delta":             float(ctx.get("pcr_delta") or 0.0),
            "skew_delta":            float(ctx.get("skew_delta") or 0.0),
            "deepseek_dir_prob":     self._last_deepseek_dir_prob,
```

Note: `eth_direction_15min` uses `0.5` not `0.0` as the unknown default (0.5 = no directional information).

- [ ] **Step 4.6: Update `_FEATURE_ORDER` in `regime_model.py`**

Append 6 entries to `_FEATURE_ORDER` after `"kalshi_early_drift"`:

```python
    # Session 39 — cascade momentum, cross-asset, order flow, options delta, LLM direction
    "liq_net_norm",
    "eth_direction_15min",
    "okx_spot_imbalance",
    "pcr_delta",
    "skew_delta",
    "deepseek_dir_prob",
```

- [ ] **Step 4.7: Add 6 column migrations to `main.py`**

Find `_CANDLE_FEATURES_COLUMN_MIGRATIONS` (a list of `(column_name, sql_type)` tuples). Add after the `kalshi_early_drift` entry:

```python
    ("liq_net_norm",         "REAL DEFAULT NULL"),
    ("eth_direction_15min",  "REAL DEFAULT NULL"),
    ("okx_spot_imbalance",   "REAL DEFAULT NULL"),
    ("pcr_delta",            "REAL DEFAULT NULL"),
    ("skew_delta",           "REAL DEFAULT NULL"),
    ("deepseek_dir_prob",    "REAL DEFAULT NULL"),
```

- [ ] **Step 4.8: Add 6 columns to candle_features INSERT in `main.py`**

Find the `candle_features` INSERT statement. It currently ends with `kalshi_early_drift`. Extend the column list and values to include the 6 new features. The values come from `regime_features` dict returned by `get_features_snapshot()`.

Locate the INSERT (search for `INSERT INTO candle_features` or `candle_ts, features_stale`). Add to the column list:

```
liq_net_norm, eth_direction_15min, okx_spot_imbalance, pcr_delta, skew_delta, deepseek_dir_prob
```

And add to the values (using `feats.get("liq_net_norm")` etc., same pattern as existing feature insertions):

```python
feats.get("liq_net_norm"),
feats.get("eth_direction_15min"),
feats.get("okx_spot_imbalance"),
feats.get("pcr_delta"),
feats.get("skew_delta"),
feats.get("deepseek_dir_prob"),
```

- [ ] **Step 4.9: Run feature_order tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_feature_order.py -v 2>&1 | tail -20
```

Expected: all pass, including `len(_FEATURE_ORDER) == 39`.

- [ ] **Step 4.10: Run the full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -30
```

Expected: all 530+ tests pass. If any fail, fix before committing.

- [ ] **Step 4.11: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/signal/fusion.py btc_kalshi_system/models/regime_model.py main.py tests/signal/test_feature_order.py && git commit -m "feat: wire 6 new features into fusion, regime model, and candle_features schema (33→39)"
```

---

## Self-Review

**Spec coverage check:**
- [x] Liquidation cascades: `liq_net_norm` in derivatives_feed → Redis → fusion → `_FEATURE_ORDER` ✓
- [x] PCR/skew delta: `pcr_delta` + `skew_delta` in deribit_options_feed → Redis → fusion → `_FEATURE_ORDER` ✓
- [x] DeepSeek directional: `dir_prob_up` prompt addition → `deepseek_dir_prob` stored in fusion → `_FEATURE_ORDER` ✓
- [x] ETH 15-min direction: `eth_direction_15min` in derivatives_feed → Redis → fusion → `_FEATURE_ORDER` ✓
- [x] OKX spot imbalance: `okx_spot_imbalance` in derivatives_feed → Redis → fusion → `_FEATURE_ORDER` ✓
- [x] DB migrations: all 6 columns added to `_CANDLE_FEATURES_COLUMN_MIGRATIONS` and INSERT ✓
- [x] `train_regime._FEATURE_COLS` auto-syncs from `_FEATURE_ORDER` (verified in test_feature_order.py) ✓
- [x] Candle logger logs all new features via `get_features_snapshot()` → `feats` dict → INSERT ✓

**Missing coverage found:**
- `fusion.update_market_context` test in `test_fusion.py` may mock the context dict. If tests mock the ctx to contain specific keys, they won't have the new keys and new feature reads will default to 0.0/0.5 — which is correct behavior (fallback). No test changes needed there.
- The `_fetch_features()` test for the full gather (`test_derivatives_feed_okx_stale.py`) mocks at the method level, not the individual fetches — should still pass without changes.

**Type consistency check:**
- `liq_net_norm`, `okx_spot_imbalance` → float, default 0.0 ✓
- `eth_direction_15min` → float, default 0.5 (unknown), 0.0 (down), 1.0 (up) ✓
- `pcr_delta`, `skew_delta` → float, default 0.0 ✓
- `deepseek_dir_prob` → float (0.0–1.0), default 0.5 ✓
- `_last_deepseek_dir_prob` in fusion → initialized to 0.5, always set before `_regime_features()` is called from `get_signal()`. From `get_features_snapshot()` (candle logger path), it retains the last `get_signal()` value — same pattern as `_last_kronos_raw_15min`. ✓
