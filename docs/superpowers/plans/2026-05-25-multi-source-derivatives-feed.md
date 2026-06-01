# Multi-Source Derivatives Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OKX as the sole source of funding rate and open interest data by fusing three parallel sources (OKX, Hyperliquid, Kraken Futures) so geo-blocking any single exchange cannot interrupt the feed.

**Architecture:** `DerivativesFeed._fetch_funding_and_oi()` is refactored to query all three sources in parallel, average the results from whichever succeed, and set `okx_partial=True` only when all three fail. Each exchange tracks its own `_prev_oi` to compute per-exchange OI deltas; those deltas are averaged. Funding rates are normalized to 8-hour equivalent before averaging. Hyperliquid and Kraken Futures are called directly via `aiohttp` (not ccxt) since ccxt doesn't offer clean support. Phase 2 updates the training filter to exclude rows where `okx_stale=1` so the retrained model learns from clean multi-source features.

**Tech Stack:** Python 3.11+, aiohttp (already in requirements), fakeredis (tests), pytest-asyncio, SQLite (trades.db)

---

## File Map

| File | Change |
|------|--------|
| `config.py` | Add `HYPERLIQUID_BASE_URL`, `KRAKEN_FUTURES_BASE_URL` constants |
| `btc_kalshi_system/data/derivatives_feed.py` | Add HL + KF fetchers; refactor `_fetch_funding_and_oi` for multi-source fusion; change `_prev_oi: float` → `_prev_oi: dict[str, float]`; add Kraken spot fallback for `_fetch_volume_ratio` |
| `tests/data/test_derivatives_feed.py` | Add tests for new fetchers and multi-source averaging |
| `scripts/train_regime.py` | Add `okx_stale` exclusion filter to `_EXTRA_FILTERS_27` and `_EXTRA_FILTERS_20` |

---

## Task 1: Config constants

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add constants to config.py**

In `config.py`, add after the `COINGLASS_BASE` definition (or at the end of the file, before `PAPER_TRADING`):

```python
HYPERLIQUID_BASE_URL: str = "https://api.hyperliquid.xyz"
KRAKEN_FUTURES_BASE_URL: str = "https://futures.kraken.com/derivatives/api/v3"
```

- [ ] **Step 2: Verify the import in derivatives_feed.py compiles**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "from config import HYPERLIQUID_BASE_URL, KRAKEN_FUTURES_BASE_URL; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add Hyperliquid and Kraken Futures base URL constants"
```

---

## Task 2: Hyperliquid funding/OI fetcher

**Files:**
- Modify: `btc_kalshi_system/data/derivatives_feed.py`
- Test: `tests/data/test_derivatives_feed.py`

Hyperliquid API: `POST https://api.hyperliquid.xyz/info` with `{"type": "metaAndAssetCtxs"}`.
Response is a 2-element list: `[metadata_obj, [per_asset_ctx, ...]]`. BTC is always index 0.
`ctx["funding"]` is the **1-hour** funding rate (string). Normalize to 8h equivalent: `* 8`.
`ctx["openInterest"]` is BTC-denominated (string).

- [ ] **Step 1: Write the failing test**

Add to `tests/data/test_derivatives_feed.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_hyperliquid_fetcher_returns_normalized_funding_and_oi -v
```
Expected: FAIL with `AttributeError: '_fetch_hyperliquid_funding_and_oi'`

- [ ] **Step 3: Change `_prev_oi` from float to dict and add the Hyperliquid method**

In `derivatives_feed.py`, change the `__init__` method's `_prev_oi` line:

```python
# Old:
self._prev_oi: float = 0.0
# New:
self._prev_oi: dict[str, float] = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0}
```

Then add this method to the class (after `_coinglass_funding_and_oi`):

```python
async def _fetch_hyperliquid_funding_and_oi(self) -> tuple[float, float]:
    """Returns (funding_rate_8h_equiv, oi_delta_pct) from Hyperliquid DEX.
    Funding is 1h rate normalized to 8h. Never geo-blocked (it's a DEX)."""
    import aiohttp
    url = f"{_HYPERLIQUID_BASE}/info"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"type": "metaAndAssetCtxs"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json()

    universe = data[0]["universe"]
    btc_idx = next(i for i, u in enumerate(universe) if u["name"] == "BTC")
    ctx = data[1][btc_idx]

    funding_1h = float(ctx["funding"])
    funding_8h = funding_1h * 8

    curr_oi = float(ctx["openInterest"])
    prev = self._prev_oi["hyperliquid"]
    oi_delta = self._oi_delta_pct(prev, curr_oi)
    self._prev_oi["hyperliquid"] = curr_oi

    return funding_8h, oi_delta
```

Also add the module-level constant near the top of the file (after the existing `_COINGLASS_BASE` line):

```python
from config import COINGLASS_API_KEY, HYPERLIQUID_BASE_URL, KRAKEN_FUTURES_BASE_URL, REDIS_URL

_HYPERLIQUID_BASE = HYPERLIQUID_BASE_URL
_KRAKEN_FUTURES_BASE = KRAKEN_FUTURES_BASE_URL
```

(Remove `REDIS_URL` from the old import line since we're consolidating.)

- [ ] **Step 4: Run test to verify it passes**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_hyperliquid_fetcher_returns_normalized_funding_and_oi -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -v
```
Expected: All previously passing tests still pass (some may fail because `_prev_oi` type changed — fix those first if so).

**If `_prev_oi` type change breaks existing tests:** In `make_feed()` in the test file, update:
```python
feed._prev_oi = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0}
```

Also update any test that sets `feed._prev_oi = 1000.0` to `feed._prev_oi = {"okx": 1000.0, "hyperliquid": 0.0, "kraken_futures": 0.0}`.

- [ ] **Step 6: Commit**

```bash
git add btc_kalshi_system/data/derivatives_feed.py tests/data/test_derivatives_feed.py config.py
git commit -m "feat: add Hyperliquid funding/OI fetcher (1h rate normalized to 8h)"
```

---

## Task 3: Kraken Futures funding/OI fetcher

**Files:**
- Modify: `btc_kalshi_system/data/derivatives_feed.py`
- Test: `tests/data/test_derivatives_feed.py`

Kraken Futures API: `GET https://futures.kraken.com/derivatives/api/v3/tickers`.
Response: `{"tickers": [{"symbol": "PF_XBTUSD", "fundingRate": <annualized_float>, "openInterest": <usd_float>}, ...]}`.
`PF_XBTUSD` is the linear BTC-USD perpetual. `fundingRate` is annualized — convert to 8h: `/ (365 * 3)`.
`openInterest` is in USD — consistent units for delta_pct purposes (we only care about % change, not absolute).

- [ ] **Step 1: Write the failing test**

Add to `tests/data/test_derivatives_feed.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_kraken_futures_fetcher_returns_normalized_funding_and_oi -v
```
Expected: FAIL with `AttributeError: '_fetch_kraken_futures_funding_and_oi'`

- [ ] **Step 3: Implement the Kraken Futures fetcher**

Add this method to `DerivativesFeed` in `derivatives_feed.py` (after `_fetch_hyperliquid_funding_and_oi`):

```python
async def _fetch_kraken_futures_funding_and_oi(self) -> tuple[float, float]:
    """Returns (funding_rate_8h_equiv, oi_delta_pct) from Kraken Futures.
    fundingRate from their API is annualized; divide by 1095 to get 8h equivalent.
    openInterest is USD-denominated; consistent for delta_pct calculation."""
    import aiohttp
    url = f"{_KRAKEN_FUTURES_BASE}/tickers"
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json()

    ticker = next(
        (t for t in data["tickers"] if t.get("symbol") == "PF_XBTUSD"),
        None,
    )
    if ticker is None:
        raise ValueError("PF_XBTUSD not found in Kraken Futures tickers")

    funding_annual = float(ticker["fundingRate"] or 0.0)
    funding_8h = funding_annual / (365 * 3)

    curr_oi = float(ticker.get("openInterest") or 0.0)
    prev = self._prev_oi["kraken_futures"]
    oi_delta = self._oi_delta_pct(prev, curr_oi)
    self._prev_oi["kraken_futures"] = curr_oi

    return funding_8h, oi_delta
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_kraken_futures_fetcher_returns_normalized_funding_and_oi -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -v
```
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add btc_kalshi_system/data/derivatives_feed.py tests/data/test_derivatives_feed.py
git commit -m "feat: add Kraken Futures funding/OI fetcher (annualized rate normalized to 8h)"
```

---

## Task 4: Multi-source fusion in `_fetch_funding_and_oi`

**Files:**
- Modify: `btc_kalshi_system/data/derivatives_feed.py`
- Test: `tests/data/test_derivatives_feed.py`

Replace the current OKX-primary → Coinglass-fallback chain with a parallel fetch from all three sources. Average the funding rates and OI deltas of whichever succeed. Set `okx_partial=True` only when all three fail. OKX's `_prev_oi` must now use `self._prev_oi["okx"]` instead of `self._prev_oi`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_derivatives_feed.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -k "multi_source" -v
```
Expected: FAIL — `_fetch_okx_funding_and_oi` does not exist yet.

- [ ] **Step 3: Rename and refactor `_fetch_funding_and_oi`**

Replace the entire `_fetch_funding_and_oi` method (and `_coinglass_funding_and_oi`) in `derivatives_feed.py` with the following. The old Coinglass fallback is kept as a private helper but is no longer the chain's end — it's one more optional source.

First, extract the OKX logic into its own method `_fetch_okx_funding_and_oi`:

```python
async def _fetch_okx_funding_and_oi(self) -> tuple[float, float, float]:
    """Returns (curr_funding_8h, funding_trend, oi_delta_pct) from the active ccxt exchange (OKX/Bybit)."""
    funding_history, oi_data = await asyncio.gather(
        self._exchange.fetch_funding_rate_history(_SYMBOL, limit=10),
        self._exchange.fetch_open_interest(_SYMBOL),
    )
    curr_funding = float(funding_history[-1]["fundingRate"]) if funding_history else 0.0
    trend = self._funding_rate_trend(funding_history)
    curr_oi = float(oi_data.get("openInterestAmount", 0.0))
    oi_delta = self._oi_delta_pct(self._prev_oi["okx"], curr_oi)
    self._prev_oi["okx"] = curr_oi
    return curr_funding, trend, oi_delta
```

Then replace `_fetch_funding_and_oi` with:

```python
async def _fetch_funding_and_oi(self) -> tuple[float, float, float, bool]:
    """Returns (curr_funding, funding_trend, oi_delta_pct, okx_partial).

    Queries OKX (via ccxt), Hyperliquid, and Kraken Futures in parallel.
    Averages results from whichever sources succeed. okx_partial=True only
    when ALL three sources fail — that is the only case worth marking stale.
    """
    results = await asyncio.gather(
        self._fetch_okx_funding_and_oi(),
        self._fetch_hyperliquid_funding_and_oi(),
        self._fetch_kraken_futures_funding_and_oi(),
        return_exceptions=True,
    )
    okx_result, hl_result, kf_result = results

    fundings: list[float] = []
    oi_deltas: list[float] = []
    trend = 0.0

    if not isinstance(okx_result, Exception):
        f, t, d = okx_result
        fundings.append(f)
        oi_deltas.append(d)
        trend = t  # only OKX provides history-based trend
    else:
        logger.warning(f"DerivativesFeed: OKX source failed — {okx_result}")

    if not isinstance(hl_result, Exception):
        f, d = hl_result
        fundings.append(f)
        oi_deltas.append(d)
    else:
        logger.warning(f"DerivativesFeed: Hyperliquid source failed — {hl_result}")

    if not isinstance(kf_result, Exception):
        f, d = kf_result
        fundings.append(f)
        oi_deltas.append(d)
    else:
        logger.warning(f"DerivativesFeed: Kraken Futures source failed — {kf_result}")

    if not fundings:
        logger.error("DerivativesFeed: all derivative sources failed — funding/OI will be zeros")
        return 0.0, 0.0, 0.0, True

    avg_funding = sum(fundings) / len(fundings)
    avg_oi_delta = sum(oi_deltas) / len(oi_deltas)
    sources_used = len(fundings)
    logger.info(f"DerivativesFeed: funding/OI from {sources_used}/3 sources — funding={avg_funding:.6f} oi_delta={avg_oi_delta:.4f}")
    return avg_funding, trend, avg_oi_delta, False
```

- [ ] **Step 4: Run multi-source tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -k "multi_source" -v
```
Expected: All 3 new tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -v
```
Expected: All tests pass. Fix any test that accessed `feed._prev_oi` as a float.

- [ ] **Step 6: Commit**

```bash
git add btc_kalshi_system/data/derivatives_feed.py tests/data/test_derivatives_feed.py
git commit -m "feat: multi-source funding/OI fusion — OKX + Hyperliquid + Kraken Futures in parallel"
```

---

## Task 5: Kraken spot fallback for `_fetch_volume_ratio`

**Files:**
- Modify: `btc_kalshi_system/data/derivatives_feed.py`
- Test: `tests/data/test_derivatives_feed.py`

`_fetch_volume_ratio` currently calls `self._exchange.fetch_ohlcv` (OKX). When OKX is down, it silently returns 1.0. We can do better: Kraken spot has the same 1h OHLCV data and is already connected via `_kraken_exchange`. Use it as a fallback.

- [ ] **Step 1: Write the failing test**

Add to `tests/data/test_derivatives_feed.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_volume_ratio_falls_back_to_kraken_when_okx_fails -v
```
Expected: FAIL (currently returns 1.0 on exception instead of trying Kraken).

- [ ] **Step 3: Add Kraken fallback to `_fetch_volume_ratio`**

Replace the existing `_fetch_volume_ratio` method:

```python
async def _fetch_volume_ratio(self) -> float:
    """1h volume as a multiple of the 30-day hourly average. 1.0 = normal.
    Tries the primary exchange (OKX) first; falls back to Kraken spot."""
    for exchange, symbol in [
        (self._exchange, _SYMBOL),
        (await self._get_kraken_exchange(), _KRAKEN_SYMBOL),
    ]:
        try:
            candles = await exchange.fetch_ohlcv(symbol, "1h", limit=721)
            if len(candles) < 30:
                return 1.0
            avg_volume = sum(c[5] for c in candles[:-1]) / len(candles[:-1])
            if avg_volume == 0:
                return 1.0
            current_volume = candles[-1][5]
            return round(current_volume / avg_volume, 3)
        except Exception as exc:
            logger.warning(f"DerivativesFeed: volume_ratio fetch failed for {symbol} — {exc}")
    return 1.0
```

Also add this helper method (Kraken exchange is already lazily initialized for trades; reuse the same pattern):

```python
async def _get_kraken_exchange(self):
    """Lazy-initialize and return the Kraken ccxt exchange instance."""
    if self._kraken_exchange is None:
        self._kraken_exchange = self._ccxt_async.kraken({"enableRateLimit": True})
    return self._kraken_exchange
```

Update `_kraken_trades_data` to use `_get_kraken_exchange`:

```python
async def _kraken_trades_data(self) -> tuple[float, float, float]:
    kraken = await self._get_kraken_exchange()
    trades = await kraken.fetch_trades(_KRAKEN_SYMBOL, limit=500)
    return self._cvd_normalized(trades), self._basis_spread_pct(trades), self._large_print_direction(trades)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_volume_ratio_falls_back_to_kraken_when_okx_fails -v
```
Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -v
```
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add btc_kalshi_system/data/derivatives_feed.py tests/data/test_derivatives_feed.py
git commit -m "feat: Kraken spot fallback for volume_ratio when OKX unavailable"
```

---

## Task 6: Smoke-test the live feed

**Files:** none modified — this task just validates the running system.

- [ ] **Step 1: Start the system and watch the first derivatives write**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 main.py 2>&1 | grep -E "(DerivativesFeed|regime:features|source)" | head -20
```
Expected: Lines like:
```
DerivativesFeed: funding/OI from 3/3 sources — funding=0.000123 oi_delta=0.0023
DerivativesFeed: wrote regime:features — {...}
```
Or if OKX is blocked:
```
DerivativesFeed: OKX source failed — ...
DerivativesFeed: funding/OI from 2/3 sources — funding=0.000118 oi_delta=0.0019
DerivativesFeed: wrote regime:features — {...}
```
The key: the feed writes real data regardless of OKX status.

- [ ] **Step 2: Manually verify Hyperliquid is reachable**

```bash
curl -s -X POST "https://api.hyperliquid.xyz/info" \
  -H "Content-Type: application/json" \
  -d '{"type": "metaAndAssetCtxs"}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('BTC funding 1h:', d[1][0]['funding'], '8h equiv:', float(d[1][0]['funding'])*8)"
```
Expected: prints BTC funding rate values.

- [ ] **Step 3: Manually verify Kraken Futures is reachable**

```bash
curl -s "https://futures.kraken.com/derivatives/api/v3/tickers" | python3 -c "import json,sys; d=json.load(sys.stdin); btc=[t for t in d['tickers'] if t['symbol']=='PF_XBTUSD'][0]; print('BTC funding annualized:', btc['fundingRate'], '8h equiv:', btc['fundingRate']/1095)"
```
Expected: prints BTC funding rate values.

---

## Task 7: Add `okx_stale` exclusion to training filter

**Files:**
- Modify: `scripts/train_regime.py`
- Test: verify query counts

After Phase 1 is running and new multi-source rows are accumulating, update the training filter so future retrains exclude the exchange-outage rows.

- [ ] **Step 1: Add `okx_stale` filter to both filter sets**

In `scripts/train_regime.py`, update `_EXTRA_FILTERS_20` and `_EXTRA_FILTERS_27`:

```python
_EXTRA_FILTERS_20 = """AND cvd_velocity IS NOT NULL
  AND brti_momentum_5min IS NOT NULL
  AND kalshi_implied_prob IS NOT NULL
  AND funding_window_proximity IS NOT NULL
  AND large_print_direction IS NOT NULL
  AND COALESCE(okx_stale, 0) = 0"""

_EXTRA_FILTERS_27 = _EXTRA_FILTERS_20 + "\n  AND deribit_stale = 0\n  AND atm_iv IS NOT NULL"
```

`COALESCE(okx_stale, 0) = 0` handles rows from before the `okx_stale` column existed (they have NULL, not 0).

- [ ] **Step 2: Verify the updated query produces a non-zero row count**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "
import sqlite3
conn = sqlite3.connect('trades.db')
q = '''
SELECT COUNT(*) FROM trades
WHERE features_stale = 0
  AND funding_rate IS NOT NULL
  AND cvd_velocity IS NOT NULL
  AND brti_momentum_5min IS NOT NULL
  AND kalshi_implied_prob IS NOT NULL
  AND funding_window_proximity IS NOT NULL
  AND large_print_direction IS NOT NULL
  AND COALESCE(okx_stale, 0) = 0
  AND deribit_stale = 0
  AND atm_iv IS NOT NULL
  AND outcome IS NOT NULL
'''
print('Qualifying rows:', conn.execute(q).fetchone()[0])
"
```
Expected: some integer ≥ 0. If it drops significantly compared to before, investigate — `okx_stale=1` rows should be rare.

- [ ] **Step 3: Run a dry-run of train_regime.py to confirm no crashes**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/train_regime.py --dry-run
```
Expected: prints row count and exits without error. If row count is < 500, the model won't retrain — that's expected. Run the actual retrain once ≥ 500 qualifying rows accumulate.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_regime.py
git commit -m "feat: exclude okx_stale rows from regime model training filter"
```

---

## Task 8: Retrain regime model (run when ≥ 500 qualifying rows)

**Files:** `models/regime.pkl` is regenerated. No code changes.

- [ ] **Step 1: Check qualifying row count**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/train_regime.py --dry-run 2>&1 | grep -i "row"
```
Expected: prints row count. Proceed only if ≥ 500.

- [ ] **Step 2: Train**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/train_regime.py --out models/regime.pkl
```
Expected: prints Brier score, accuracy, Kronos agreement. Note these metrics.

- [ ] **Step 3: Validate the new model in shadow mode**

In `.env`, ensure `REGIME_GATE2_ENFORCING=false`. Restart the system and monitor Gate 2 disagreement rate for ~50 trades. If disagreement rate is < 30%, flip `REGIME_GATE2_ENFORCING=true`.

- [ ] **Step 4: Commit the new model**

```bash
git add models/regime.pkl
git commit -m "feat: retrain regime model on multi-source clean data"
```

---

## Self-Review

**Spec coverage:**
- OKX geo-block resilience → Tasks 2-4 (Hyperliquid + KF fetchers + fusion)
- No Coinglass dependency → Coinglass is preserved as optional helper but not required; Tasks 2-4 work without it
- Volume ratio fallback → Task 5
- Model trained on clean multi-source data → Tasks 7-8

**Placeholder scan:** No TBD or TODO patterns found.

**Type consistency:**
- `_prev_oi` changes from `float` to `dict[str, float]` in Task 2; all downstream uses (`_fetch_okx_funding_and_oi`, `_fetch_hyperliquid_funding_and_oi`, `_fetch_kraken_futures_funding_and_oi`) access it with string keys — consistent.
- `_fetch_okx_funding_and_oi` returns `tuple[float, float, float]` — referenced correctly in Task 4 fusion code.
- `_fetch_hyperliquid_funding_and_oi` and `_fetch_kraken_futures_funding_and_oi` return `tuple[float, float]` (no trend) — fusion code unpacks as `f, d = hl_result` — consistent.
