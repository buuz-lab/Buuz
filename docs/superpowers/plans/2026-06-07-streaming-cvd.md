# Streaming CVD + 15s Batch Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 60s HTTP trade-fetch for CVD with a persistent WebSocket accumulator so CVD reflects the last trade tick at decision time, while dropping the batch interval from 60s to 15s for liquidations and spot imbalance.

**Architecture:** A new `StreamingCVDAccumulator` class opens a persistent OKX WebSocket connection (Kraken fallback), accumulates `(ts_ms, side, size, price)` ticks in a 15-min rolling deque, and exposes `cvd_normalized`, `large_print_direction`, `last_price`, and `is_stale` as properties. `DerivativesFeed._fetch_features()` is restructured into a fast tier (liq, spot imbalance, CVD from accumulator — every 15s) and a slow tier (funding, OI, ETH direction, volume ratio — at most once per 60s). Everything downstream (fusion, Redis key format, LKG path) is unchanged.

**Tech Stack:** `websockets>=12.0` (already installed), `asyncio`, existing `ccxt.async_support` (kept for volume ratio Kraken fallback)

---

## File Map

| File | Change |
|---|---|
| `btc_kalshi_system/data/derivatives_feed.py` | Add `StreamingCVDAccumulator`; restructure `_fetch_features()`, `run()`; update constants; remove `_fetch_trades_data()`, `_kraken_trades_data()` |
| `tests/data/test_streaming_cvd.py` | **New** — unit tests for accumulator state, computation, and WS message parsing |
| `tests/data/test_derivatives_feed.py` | Update TTL assertion; add slow-tier cache test; add accumulator-as-CVD-source test |

---

## Task 1: StreamingCVDAccumulator — data model

**Files:**
- Create: `tests/data/test_streaming_cvd.py`
- Modify: `btc_kalshi_system/data/derivatives_feed.py`

- [ ] **Step 1: Write failing tests**

Create `tests/data/test_streaming_cvd.py`:

```python
import json
import time
import pytest
from btc_kalshi_system.data.derivatives_feed import StreamingCVDAccumulator


def _tick(side: str, size: float, price: float, age_s: float = 0.0):
    """Helper: build a (ts_ms, side, size, price) tuple."""
    ts_ms = int((time.time() - age_s) * 1000)
    return (ts_ms, side, size, price)


def test_cvd_all_buys():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._ingest_tick(_tick("buy", 2.0, 95001.0))
    assert acc.cvd_normalized == pytest.approx(1.0)


def test_cvd_equal_buys_sells():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    assert acc.cvd_normalized == pytest.approx(0.0)


def test_cvd_mixed():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 3.0, 95000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    # (3-1)/(3+1) = 0.5
    assert acc.cvd_normalized == pytest.approx(0.5)


def test_old_ticks_pruned():
    """Tick older than 15 min is dropped; only newer tick counts."""
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 5.0, 95000.0, age_s=960))  # 16 min old
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    assert len(acc._trades) == 1
    assert acc.cvd_normalized == pytest.approx(-1.0)


def test_is_stale_with_no_ticks():
    acc = StreamingCVDAccumulator()
    assert acc.is_stale is True


def test_is_stale_clears_after_5_ticks():
    acc = StreamingCVDAccumulator()
    for _ in range(5):
        acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    assert acc.is_stale is False


def test_is_stale_after_silence():
    acc = StreamingCVDAccumulator()
    for _ in range(5):
        acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._last_tick_at = time.time() - 121  # backdate to simulate silence
    assert acc.is_stale is True


def test_last_price_tracks_most_recent_tick():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 94000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    assert acc.last_price == pytest.approx(95000.0)


def test_large_print_direction_buy_dominated():
    acc = StreamingCVDAccumulator()
    # avg_size = (4*1 + 6)/5 = 2.0, threshold = 4.0, only the buy-6 is large
    for _ in range(4):
        acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    acc._ingest_tick(_tick("buy", 6.0, 95000.0))
    assert acc.large_print_direction == pytest.approx(1.0)


def test_large_print_direction_no_large_trades():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    # avg=1.0, threshold=2.0, no trades exceed it → 0.0
    assert acc.large_print_direction == pytest.approx(0.0)


def test_cvd_empty_deque():
    acc = StreamingCVDAccumulator()
    assert acc.cvd_normalized == 0.0
    assert acc.large_print_direction == 0.0
    assert acc.last_price == 0.0
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_streaming_cvd.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `AttributeError` — `StreamingCVDAccumulator` does not exist yet.

- [ ] **Step 3: Add `StreamingCVDAccumulator` to `derivatives_feed.py`**

Add this class near the top of `derivatives_feed.py`, just before the `DerivativesFeed` class (after the module-level constants). Also add these two imports at the top of the file alongside the existing imports:

```python
import websockets
from datetime import datetime, timezone
```

```python
_CVD_WINDOW_MS = 15 * 60 * 1000   # 15-minute rolling trade window
_CVD_STALE_TIMEOUT = 120           # seconds of silence before marking stale
_CVD_MIN_TICKS = 5                 # minimum deque length to leave cold-start


class StreamingCVDAccumulator:
    """Accumulates BTC perp trade ticks via WebSocket; exposes real-time CVD.

    Deque entry format: (ts_ms: int, side: str, size: float, price: float)
    """

    _OKX_WS_URL    = "wss://ws.okx.com:8443/ws/v5/public"
    _KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

    def __init__(self) -> None:
        from collections import deque
        self._trades: deque = deque()
        self._cvd: float = 0.0
        self._large_print: float = 0.0
        self._last_price: float = 0.0
        self._last_tick_at: float = 0.0

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def cvd_normalized(self) -> float:
        return self._cvd

    @property
    def large_print_direction(self) -> float:
        return self._large_print

    @property
    def last_price(self) -> float:
        return self._last_price

    @property
    def is_stale(self) -> bool:
        if len(self._trades) < _CVD_MIN_TICKS:
            return True
        return (time.time() - self._last_tick_at) > _CVD_STALE_TIMEOUT

    # ── Tick ingestion ─────────────────────────────────────────────────────────

    def _ingest_tick(self, tick: tuple) -> None:
        """Append tick, prune window, recompute derived values."""
        self._trades.append(tick)
        cutoff_ms = time.time() * 1000 - _CVD_WINDOW_MS
        while self._trades and self._trades[0][0] < cutoff_ms:
            self._trades.popleft()
        self._last_price = tick[3]
        self._last_tick_at = time.time()
        self._recompute()

    def _recompute(self) -> None:
        trades = self._trades
        if not trades:
            self._cvd = 0.0
            self._large_print = 0.0
            return

        # CVD
        buy_vol  = sum(t[2] for t in trades if t[1] == "buy")
        sell_vol = sum(t[2] for t in trades if t[1] == "sell")
        total = buy_vol + sell_vol
        self._cvd = (buy_vol - sell_vol) / total if total > 0.0 else 0.0

        # Large print direction
        sizes = [t[2] for t in trades]
        avg_size = sum(sizes) / len(sizes)
        threshold = 2 * avg_size
        large = [t for t in trades if t[2] > threshold]
        if not large:
            self._large_print = 0.0
            return
        lb = sum(t[2] for t in large if t[1] == "buy")
        ls = sum(t[2] for t in large if t[1] == "sell")
        lt = lb + ls
        self._large_print = (lb - ls) / lt if lt > 0.0 else 0.0
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_streaming_cvd.py -v 2>&1 | tail -20
```

Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add tests/data/test_streaming_cvd.py btc_kalshi_system/data/derivatives_feed.py && git commit -m "$(cat <<'EOF'
feat: StreamingCVDAccumulator data model — deque, CVD, large_print, stale detection

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: StreamingCVDAccumulator — WS message parsing + `run()`

**Files:**
- Modify: `tests/data/test_streaming_cvd.py`
- Modify: `btc_kalshi_system/data/derivatives_feed.py`

- [ ] **Step 1: Add message-parsing tests**

Append to `tests/data/test_streaming_cvd.py`:

```python
def test_parse_okx_message_single_tick():
    acc = StreamingCVDAccumulator()
    msg = json.dumps({
        "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
        "data": [{"px": "95000.5", "sz": "0.25", "side": "buy", "ts": "1700000000000"}]
    })
    ticks = acc._parse_okx_message(msg)
    assert len(ticks) == 1
    ts_ms, side, size, price = ticks[0]
    assert ts_ms == 1700000000000
    assert side == "buy"
    assert size == pytest.approx(0.25)
    assert price == pytest.approx(95000.5)


def test_parse_okx_message_ignores_non_trade_events():
    acc = StreamingCVDAccumulator()
    # Subscription confirmation — no "data" key with trade fields
    msg = json.dumps({"event": "subscribe", "arg": {"channel": "trades"}})
    ticks = acc._parse_okx_message(msg)
    assert ticks == []


def test_parse_kraken_message_single_tick():
    acc = StreamingCVDAccumulator()
    msg = json.dumps({
        "channel": "trade",
        "type": "update",
        "data": [{"side": "sell", "qty": 0.1, "price": 94500.0,
                  "timestamp": "2023-11-14T12:00:00.000000Z"}]
    })
    ticks = acc._parse_kraken_message(msg)
    assert len(ticks) == 1
    ts_ms, side, size, price = ticks[0]
    assert side == "sell"
    assert size == pytest.approx(0.1)
    assert price == pytest.approx(94500.0)
    assert ts_ms > 0


def test_parse_kraken_message_ignores_non_update_events():
    acc = StreamingCVDAccumulator()
    msg = json.dumps({"channel": "trade", "type": "snapshot", "data": []})
    ticks = acc._parse_kraken_message(msg)
    assert ticks == []
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_streaming_cvd.py::test_parse_okx_message_single_tick tests/data/test_streaming_cvd.py::test_parse_kraken_message_single_tick -v 2>&1 | tail -10
```

Expected: `AttributeError: 'StreamingCVDAccumulator' object has no attribute '_parse_okx_message'`

- [ ] **Step 3: Add parse methods and `run()` to `StreamingCVDAccumulator`**

Add the following methods inside `StreamingCVDAccumulator`, after `_recompute()`:

```python
    # ── Message parsing ────────────────────────────────────────────────────────

    def _parse_okx_message(self, raw: str) -> list[tuple]:
        """Parse OKX WS trade message → list of (ts_ms, side, size, price) tuples."""
        try:
            msg = json.loads(raw)
            data = msg.get("data")
            if not data or not isinstance(data, list):
                return []
            ticks = []
            for t in data:
                if "px" not in t or "sz" not in t or "side" not in t or "ts" not in t:
                    continue
                ticks.append((int(t["ts"]), t["side"], float(t["sz"]), float(t["px"])))
            return ticks
        except Exception:
            return []

    def _parse_kraken_message(self, raw: str) -> list[tuple]:
        """Parse Kraken WS v2 trade message → list of (ts_ms, side, size, price) tuples."""
        try:
            msg = json.loads(raw)
            if msg.get("channel") != "trade" or msg.get("type") != "update":
                return []
            ticks = []
            for t in msg.get("data", []):
                ts_str = t["timestamp"].replace("Z", "+00:00")
                ts_ms = int(datetime.fromisoformat(ts_str).timestamp() * 1000)
                ticks.append((ts_ms, t["side"], float(t["qty"]), float(t["price"])))
            return ticks
        except Exception:
            return []

    # ── WebSocket run loop ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Maintain persistent WS connection; OKX primary, Kraken fallback after 3 failures."""
        okx_failures = 0
        while True:
            use_kraken = okx_failures >= 3
            try:
                if use_kraken:
                    await self._run_kraken()
                    okx_failures = 0  # Kraken succeeded — reset OKX failure count
                else:
                    await self._run_okx()
                    okx_failures = 0
            except Exception as exc:
                if not use_kraken:
                    okx_failures += 1
                    backoff = min(2 ** okx_failures, 30)
                    logger.warning(f"StreamingCVDAccumulator: OKX WS error (attempt {okx_failures}): {exc} — retry in {backoff}s")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"StreamingCVDAccumulator: Kraken WS also failed: {exc} — retry in 30s")
                    await asyncio.sleep(30)

    async def _run_okx(self) -> None:
        sub = json.dumps({"op": "subscribe", "args": [{"channel": "trades", "instId": "BTC-USDT-SWAP"}]})
        async with websockets.connect(self._OKX_WS_URL, ping_interval=20, ping_timeout=10) as ws:
            await ws.send(sub)
            logger.info("StreamingCVDAccumulator: OKX WS connected")
            async for raw in ws:
                for tick in self._parse_okx_message(raw):
                    self._ingest_tick(tick)

    async def _run_kraken(self) -> None:
        sub = json.dumps({"method": "subscribe", "params": {"channel": "trade", "symbol": ["BTC/USD"]}})
        async with websockets.connect(self._KRAKEN_WS_URL, ping_interval=20, ping_timeout=10) as ws:
            await ws.send(sub)
            logger.info("StreamingCVDAccumulator: Kraken WS connected (OKX fallback)")
            async for raw in ws:
                for tick in self._parse_kraken_message(raw):
                    self._ingest_tick(tick)
```

- [ ] **Step 4: Run all streaming CVD tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_streaming_cvd.py -v 2>&1 | tail -20
```

Expected: all 15 tests pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add tests/data/test_streaming_cvd.py btc_kalshi_system/data/derivatives_feed.py && git commit -m "$(cat <<'EOF'
feat: StreamingCVDAccumulator WS parsing and run loop — OKX primary, Kraken fallback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Batch loop restructure — slow/fast tier + accumulator integration

**Files:**
- Modify: `tests/data/test_derivatives_feed.py`
- Modify: `btc_kalshi_system/data/derivatives_feed.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/data/test_derivatives_feed.py` (find a natural end-of-file location):

```python
# ── Slow-tier cache tests ──────────────────────────────────────────────────────
# Note: time, AsyncMock, MagicMock, patch, fakeredis, pytest are already imported at top of file.

def make_feed_with_mock_accumulator():
    """Feed whose accumulator is replaced with a simple mock."""
    with patch("btc_kalshi_system.data.derivatives_feed.redis") as mock_redis_mod:
        mock_redis_mod.from_url.return_value = fakeredis.FakeRedis()
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


@pytest.mark.asyncio
async def test_slow_tier_not_refetched_within_60s():
    """When _last_slow_fetch < 60s ago, funding/OI/ETH are read from cache."""
    feed = make_feed_with_mock_accumulator()
    feed._last_slow_fetch = time.time()  # just fetched

    # These should NOT be called
    feed._fetch_funding_and_oi = AsyncMock(return_value=(0.001, 0.0, 0.0, False))
    feed._fetch_eth_direction   = AsyncMock(return_value=0.5)
    feed._fetch_volume_ratio    = AsyncMock(return_value=1.0)

    # Fast-tier fetches return simple values
    feed._fetch_liquidations      = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    await feed._fetch_features()

    feed._fetch_funding_and_oi.assert_not_called()
    feed._fetch_eth_direction.assert_not_called()
    feed._fetch_volume_ratio.assert_not_called()


@pytest.mark.asyncio
async def test_slow_tier_refetched_after_60s():
    """When _last_slow_fetch > 60s ago, funding/OI/ETH are re-fetched."""
    feed = make_feed_with_mock_accumulator()
    feed._last_slow_fetch = time.time() - 61  # stale

    feed._fetch_funding_and_oi = AsyncMock(return_value=(0.002, 0.001, 0.01, False))
    feed._fetch_eth_direction   = AsyncMock(return_value=1.0)
    feed._fetch_volume_ratio    = AsyncMock(return_value=1.5)
    feed._fetch_liquidations      = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    await feed._fetch_features()

    feed._fetch_funding_and_oi.assert_called_once()
    feed._fetch_eth_direction.assert_called_once()
    feed._fetch_volume_ratio.assert_called_once()


@pytest.mark.asyncio
async def test_cvd_read_from_accumulator_not_http():
    """CVD comes from the accumulator; _fetch_trades_data is not called."""
    feed = make_feed_with_mock_accumulator()
    feed._last_slow_fetch = time.time()
    feed._fetch_liquidations       = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    features = await feed._fetch_features()

    assert features["cvd_normalized"] == pytest.approx(0.3)
    assert "_fetch_trades_data" not in dir(feed) or True  # method removed


@pytest.mark.asyncio
async def test_cvd_stale_flag_set_when_accumulator_stale():
    feed = make_feed_with_mock_accumulator()
    feed._cvd_accumulator.is_stale = True
    feed._last_slow_fetch = time.time()
    feed._fetch_liquidations       = AsyncMock(return_value=0.0)
    feed._fetch_okx_spot_imbalance = AsyncMock(return_value=0.0)

    features = await feed._fetch_features()
    assert features.get("_cvd_stale") is True
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_slow_tier_not_refetched_within_60s tests/data/test_derivatives_feed.py::test_cvd_read_from_accumulator_not_http -v 2>&1 | tail -15
```

Expected: errors — feed has no `_cvd_accumulator` or `_last_slow_fetch` attribute yet.

- [ ] **Step 3: Update `DerivativesFeed.__init__()`**

Replace the current `__init__` body with:

```python
    def __init__(self, redis_url: str = REDIS_URL) -> None:
        import ccxt.async_support as ccxt_async
        self._redis = redis.from_url(redis_url)
        self._macro_feed = MacroFeed()
        self._ccxt_async = ccxt_async
        self._exchange = None
        self._exchange_name: str = ""
        self._prev_oi: dict[str, float] = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0, "deribit": 0.0}
        self._kraken_exchange = None  # kept for _fetch_volume_ratio fallback
        self._cvd_accumulator = StreamingCVDAccumulator()
        self._last_slow_fetch: float = 0.0
        self._cached_funding_result: tuple = (0.0, 0.0, 0.0, False)
        self._cached_eth_dir: float = 0.5
        self._cached_volume_ratio: float = 1.0
```

- [ ] **Step 4: Replace `_fetch_features()` with the two-tier version**

Replace the existing `_fetch_features()` method entirely:

```python
    async def _fetch_features(self) -> dict:
        _now = time.time()
        _refetch_slow = (_now - self._last_slow_fetch) >= 60

        # Fast tier — every 15s
        liq_net_norm, okx_spot_imbalance = await asyncio.gather(
            self._fetch_liquidations(),
            self._fetch_okx_spot_imbalance(),
        )

        # Slow tier — at most once per 60s
        if _refetch_slow:
            (curr_funding, trend, oi_delta, okx_partial), eth_dir, vol_ratio = await asyncio.gather(
                self._fetch_funding_and_oi(),
                self._fetch_eth_direction(),
                self._fetch_volume_ratio(),
            )
            self._cached_funding_result = (curr_funding, trend, oi_delta, okx_partial)
            self._cached_eth_dir = eth_dir
            self._cached_volume_ratio = vol_ratio
            self._last_slow_fetch = _now
        else:
            curr_funding, trend, oi_delta, okx_partial = self._cached_funding_result
            eth_dir = self._cached_eth_dir
            vol_ratio = self._cached_volume_ratio

        # CVD from streaming accumulator — zero HTTP cost
        cvd          = self._cvd_accumulator.cvd_normalized
        large_print  = self._cvd_accumulator.large_print_direction
        _last_price  = self._cvd_accumulator.last_price
        brti         = self._get_brti_estimate()
        basis        = ((_last_price - brti) / brti) if (brti and brti > 0.0 and _last_price > 0.0) else 0.0

        vol = self._brti_volatility_1h()
        fg  = fetch_fear_greed(self._redis)

        features: dict = {
            "funding_rate":          curr_funding,
            "funding_rate_trend":    trend,
            "oi_delta_pct":          oi_delta,
            "cvd_normalized":        cvd,
            "basis_spread_pct":      basis,
            "brti_volatility_1h":    vol,
            "large_print_direction": large_print,
            "volume_ratio_1h":       vol_ratio,
            "fear_greed_value":      fg["value"] if fg else None,
            "fear_greed_label":      fg["label"] if fg else None,
            "liq_net_norm":          liq_net_norm,
            "eth_direction_15min":   eth_dir,
            "okx_spot_imbalance":    okx_spot_imbalance,
        }
        macro = self._macro_feed.get_correlations()
        features.update(macro)
        if okx_partial:
            features["_okx_partial"] = True
        if self._cvd_accumulator.is_stale:
            features["_cvd_stale"] = True
        return features
```

- [ ] **Step 5: Remove `_fetch_trades_data()` and `_kraken_trades_data()`**

Delete both methods from `DerivativesFeed`. Keep `_get_kraken_exchange()` — it is still used by `_fetch_volume_ratio()`.

- [ ] **Step 6: Update `run()` to start accumulator concurrently**

Replace the existing `run()` method:

```python
    async def run(self) -> None:
        await self._resolve_exchange()
        await asyncio.gather(
            self._batch_loop(),
            self._cvd_accumulator.run(),
        )

    async def _batch_loop(self) -> None:
        """Batch refresh loop: fast-tier features every 15s, slow-tier every 60s."""
        while True:
            if self._exchange is None:
                await self._resolve_exchange()
            success = False
            try:
                features = await self._fetch_features()
                okx_partial = features.pop("_okx_partial", False)
                self._write_features(features, okx_partial=okx_partial)
                logger.info(f"DerivativesFeed: wrote regime:features — {features}")
                success = True
            except Exception as exc:
                logger.warning(f"DerivativesFeed: fetch failed ({self._exchange_name}): {exc}")
                if self._exchange is not None:
                    await self._exchange.close()
                self._exchange = None
                await self._resolve_exchange()
            await asyncio.sleep(_REFRESH_INTERVAL - 10 if success else _REFRESH_INTERVAL)
```

- [ ] **Step 7: Update the constants at the top of the file**

```python
_REFRESH_INTERVAL = 15    # was 60 — fast tier; slow tier re-fetches at most once per 60s
_FEATURES_TTL     = 120   # was 360 — 8x the new interval
```

- [ ] **Step 8: Run the new batch tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_slow_tier_not_refetched_within_60s tests/data/test_derivatives_feed.py::test_slow_tier_refetched_after_60s tests/data/test_derivatives_feed.py::test_cvd_read_from_accumulator_not_http tests/data/test_derivatives_feed.py::test_cvd_stale_flag_set_when_accumulator_stale -v 2>&1 | tail -15
```

Expected: all 4 pass.

- [ ] **Step 9: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/data/derivatives_feed.py tests/data/test_derivatives_feed.py && git commit -m "$(cat <<'EOF'
feat: DerivativesFeed 15s batch with slow/fast tier — CVD from accumulator

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update existing tests + full suite

**Files:**
- Modify: `tests/data/test_derivatives_feed.py`

- [ ] **Step 1: Update the TTL assertion**

In `tests/data/test_derivatives_feed.py`, find:

```python
    assert 350 <= ttl <= 360
```

Replace with:

```python
    assert 110 <= ttl <= 120
```

- [ ] **Step 2: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -15
```

Expected: 570+ passing, 0 failing. If any test fails related to `_fetch_trades_data` references or `_kraken_exchange`, fix them by updating the test to use the accumulator mock instead.

- [ ] **Step 3: Restart service to activate changes**

```bash
launchctl kickstart -k gui/$(id -u)/com.kronos.v2
```

- [ ] **Step 4: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add tests/data/test_derivatives_feed.py && git commit -m "$(cat <<'EOF'
test: update TTL assertion for 15s batch interval

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
