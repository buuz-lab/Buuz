import asyncio
import json
import time
from collections import deque

import aiohttp
import numpy as np
import redis
import websockets
from datetime import datetime
from loguru import logger

from config import COINGLASS_API_KEY, HYPERLIQUID_BASE_URL, KRAKEN_FUTURES_BASE_URL, REDIS_URL
from btc_kalshi_system.data.fear_greed import fetch_fear_greed
from btc_kalshi_system.data.macro_feed import MacroFeed

_REFRESH_INTERVAL = 15    # fast tier every 15s; slow tier re-fetches at most once per 60s
_FEATURES_TTL     = 120   # 8x the new interval — survives several failed cycles
_CCXT_TIMEOUT_MS = 10_000  # 10 s — fail fast on DNS timeouts rather than hanging
_LKG_TTL = 86_400         # 24 hours — last-known-good survives multi-hour exchange outages
_FUNDING_LOOKBACK_MS = 4 * 3600_000  # 4 hours in milliseconds
_SYMBOL = "BTC/USDT:USDT"
_KRAKEN_SYMBOL = "BTC/USD"
_COINGLASS_BASE = "https://open-api-v4.coinglass.com"
_HYPERLIQUID_BASE = HYPERLIQUID_BASE_URL
_KRAKEN_FUTURES_BASE = KRAKEN_FUTURES_BASE_URL
_DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
_OKX_LIQ_URL    = "https://www.okx.com/api/v5/public/liquidation-orders"
_OKX_BOOKS_URL  = "https://www.okx.com/api/v5/market/books"
_LIQ_WINDOW_MS  = 15 * 60 * 1000  # 15 minutes in ms
_LIQ_NOISE_FLOOR = 10.0            # contracts — ignore if total below this
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

    def cvd_since_candle_open(self, candle_open_ts_ms: int) -> tuple[float | None, int]:
        """CVD and tick count for trades at or after candle_open_ts_ms (ms epoch).

        Returns (cvd_normalized, tick_count). cvd_normalized is None when no ticks exist
        for the window — caller should treat None as missing data, not zero.
        """
        recent = [t for t in self._trades if t[0] >= candle_open_ts_ms]
        n = len(recent)
        if n == 0:
            return None, 0
        buy_vol  = sum(t[2] for t in recent if t[1] == "buy")
        sell_vol = sum(t[2] for t in recent if t[1] == "sell")
        total = buy_vol + sell_vol
        cvd = (buy_vol - sell_vol) / total if total > 0.0 else 0.0
        return cvd, n

    # ── Tick ingestion ─────────────────────────────────────────────────────────

    def _ingest_tick(self, tick: tuple[int, str, float, float]) -> None:
        """Append tick, prune window, recompute derived values."""
        self._trades.append(tick)
        now = time.time()
        cutoff_ms = now * 1000 - _CVD_WINDOW_MS
        while self._trades and self._trades[0][0] < cutoff_ms:
            self._trades.popleft()
        self._last_price = tick[3]
        self._last_tick_at = now
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
                if "timestamp" not in t or "side" not in t or "qty" not in t or "price" not in t:
                    continue
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
                    okx_failures = 0
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
                    okx_failures = 0  # probe OKX again next cycle
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


class DerivativesFeed:
    """Fetches derivative market features from OKX/Bybit/Hyperliquid/Deribit/Kraken
    and writes them to Redis key 'regime:features'. CVD is updated via a persistent
    WebSocket trade stream; funding, OI, liquidations, and imbalance refresh every 15s
    (slow features capped at 60s).
    """

    # Exchange preference order — first one that connects without a 403/geo-block wins.
    # Bybit geo-blocks US users via CloudFront (HTTP 403).
    # OKX is the fallback: same perp futures data, accessible from the US.
    _EXCHANGE_PREFERENCE = ["okx", "bybit"]

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

    # ── Public entry point ─────────────────────────────────────────────────────

    async def _resolve_exchange(self) -> bool:
        """Try each exchange in preference order; set self._exchange to the first that works."""
        for name in self._EXCHANGE_PREFERENCE:
            try:
                ex = getattr(self._ccxt_async, name)({"enableRateLimit": True, "timeout": _CCXT_TIMEOUT_MS})
                # Lightweight probe — instruments-info or markets call
                await ex.load_markets()
                self._exchange = ex
                self._exchange_name = name
                logger.info(f"DerivativesFeed: using {name} for derivatives data")
                return True
            except Exception as exc:
                logger.warning(f"DerivativesFeed: {name} unavailable ({exc}), trying next …")
                try:
                    await ex.close()
                except Exception:
                    pass
        logger.error("DerivativesFeed: all exchanges unavailable — regime features will be zeros")
        return False

    async def run(self) -> None:
        await self._resolve_exchange()
        await asyncio.gather(
            self._batch_loop(),
            self._cvd_accumulator.run(),
        )

    async def _batch_loop(self) -> None:
        """Batch refresh loop: fast-tier features every cycle, slow-tier capped at 60s."""
        try:
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
        finally:
            if self._exchange is not None:
                await self._exchange.close()

    # ── Feature computation ────────────────────────────────────────────────────

    async def _fetch_features(self) -> dict:
        _now = time.time()
        _refetch_slow = (_now - self._last_slow_fetch) >= 60

        # Fast tier — every cycle
        liq_net_norm, okx_spot_imbalance = await asyncio.gather(
            self._fetch_liquidations(),
            self._fetch_okx_spot_imbalance(),
        )

        # Slow tier — at most once per 60s
        # Hard 30s wall-clock timeout guards against ccxt fetch_ohlcv(limit=721)
        # hanging when OKX REST is flaky — individual ccxt timeouts apply per-request
        # but not to total wall time when pagination or retries occur.
        if _refetch_slow:
            try:
                (curr_funding, trend, oi_delta, okx_partial), eth_dir, vol_ratio = (
                    await asyncio.wait_for(
                        asyncio.gather(
                            self._fetch_funding_and_oi(),
                            self._fetch_eth_direction(),
                            self._fetch_volume_ratio(),
                        ),
                        timeout=30,
                    )
                )
            except asyncio.TimeoutError:
                logger.warning("DerivativesFeed: slow-tier fetch timed out after 30s — using cached values")
                curr_funding, trend, oi_delta, okx_partial = self._cached_funding_result
                eth_dir = self._cached_eth_dir
                vol_ratio = self._cached_volume_ratio
            self._cached_funding_result = (curr_funding, trend, oi_delta, okx_partial)
            self._cached_eth_dir = eth_dir
            self._cached_volume_ratio = vol_ratio
            self._last_slow_fetch = _now
        else:
            curr_funding, trend, oi_delta, okx_partial = self._cached_funding_result
            eth_dir = self._cached_eth_dir
            vol_ratio = self._cached_volume_ratio

        # CVD from streaming accumulator — zero HTTP cost
        cvd         = self._cvd_accumulator.cvd_normalized
        large_print = self._cvd_accumulator.large_print_direction
        _last_price = self._cvd_accumulator.last_price
        brti        = self._get_brti_estimate()
        basis       = ((_last_price - brti) / brti) if (brti and brti > 0.0 and _last_price > 0.0) else 0.0

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

    async def _fetch_funding_and_oi(self) -> tuple[float, float, float, bool]:
        """Returns (curr_funding, funding_trend, oi_delta_pct, okx_partial).

        Queries OKX (via ccxt), Hyperliquid, Deribit, and Kraken Futures in parallel.
        Averages results from whichever sources succeed. okx_partial=True only
        when ALL four sources fail — that is the only case worth marking stale.
        """
        results = await asyncio.gather(
            self._fetch_okx_funding_and_oi(),
            self._fetch_hyperliquid_funding_and_oi(),
            self._fetch_deribit_funding_and_oi(),
            self._fetch_kraken_futures_funding_and_oi(),
            return_exceptions=True,
        )
        okx_result, hl_result, db_result, kf_result = results

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

        if not isinstance(db_result, Exception):
            f, d = db_result
            fundings.append(f)
            oi_deltas.append(d)
        else:
            logger.warning(f"DerivativesFeed: Deribit source failed — {db_result}")

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
        logger.info(f"DerivativesFeed: funding/OI from {sources_used}/4 sources — funding={avg_funding:.6f} oi_delta={avg_oi_delta:.4f}")
        return avg_funding, trend, avg_oi_delta, False

    async def _coinglass_funding_and_oi(self) -> tuple[float, float, float, bool]:
        if not COINGLASS_API_KEY:
            logger.warning("DerivativesFeed: COINGLASS_API_KEY not set — Coinglass fallback skipped")
            return 0.0, 0.0, 0.0, True

        headers = {"CG-API-KEY": COINGLASS_API_KEY}
        fr_url = f"{_COINGLASS_BASE}/api/futures/funding-rate/history"
        oi_url = f"{_COINGLASS_BASE}/api/futures/open-interest/exchange-list"

        async with aiohttp.ClientSession(headers=headers) as session:
            async def _get(url: str, params: dict) -> dict:
                async with session.get(url, params=params) as r:
                    return await r.json()

            fr_data, oi_data = await asyncio.gather(
                _get(fr_url, {"exchange": "OKX", "symbol": "BTCUSDT", "interval": "8h", "limit": "10"}),
                _get(oi_url, {"symbol": "BTC"}),
            )

        history = [
            {"timestamp": item["time"], "fundingRate": float(item["close"])}
            for item in (fr_data.get("data") or [])
        ]
        curr_funding = history[-1]["fundingRate"] if history else 0.0
        trend = self._funding_rate_trend(history)

        okx_row = next(
            (row for row in (oi_data.get("data") or []) if row.get("exchange", "").upper() == "OKX"),
            None,
        )
        curr_oi = float(okx_row["open_interest_quantity"]) if okx_row else 0.0
        oi_delta = self._oi_delta_pct(self._prev_oi["okx"], curr_oi)
        if curr_oi:
            self._prev_oi["okx"] = curr_oi
        return curr_funding, trend, oi_delta, False

    async def _fetch_hyperliquid_funding_and_oi(self) -> tuple[float, float]:
        """Returns (funding_rate_8h_equiv, oi_delta_pct) from Hyperliquid DEX.
        Funding is 1h rate normalized to 8h. Never geo-blocked (it's a DEX)."""
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

    async def _fetch_deribit_funding_and_oi(self) -> tuple[float, float]:
        """Returns (funding_rate_8h, oi_delta_pct) from Deribit BTC-PERPETUAL.
        US-accessible public REST endpoint; no geo-blocking."""
        url = f"{_DERIBIT_BASE}/ticker"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"instrument_name": "BTC-PERPETUAL"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()

        result = data["result"]
        # funding_8h is the settled 8h average; current_funding is mid-period accumulator (0 at period start)
        funding_8h = float(result.get("funding_8h", 0.0))
        curr_oi = float(result.get("open_interest", 0.0))
        prev = self._prev_oi["deribit"]
        oi_delta = self._oi_delta_pct(prev, curr_oi)
        self._prev_oi["deribit"] = curr_oi
        return funding_8h, oi_delta

    async def _fetch_kraken_futures_funding_and_oi(self) -> tuple[float, float]:
        """Returns (funding_rate_8h_equiv, oi_delta_pct) from Kraken Futures.
        fundingRate from their API is annualized; divide by 1095 to get 8h equivalent.
        openInterest is USD-denominated; consistent for delta_pct calculation."""
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

    async def _get_kraken_exchange(self):
        """Lazy-initialize and return the Kraken ccxt exchange instance."""
        if self._kraken_exchange is None:
            self._kraken_exchange = self._ccxt_async.kraken({"enableRateLimit": True, "timeout": _CCXT_TIMEOUT_MS})
        return self._kraken_exchange

    def _funding_rate_trend(self, history: list[dict]) -> float:
        """Funding rate change over the last _FUNDING_LOOKBACK_MS (4 hours).

        Returns 0.0 if:
          - Fewer than 2 history entries exist, OR
          - No entry is older than the lookback window.
        In both cases 0.0 means neutral / unknown, not a real zero trend.

        Do NOT change _FUNDING_LOOKBACK_MS or limit=10 — those must remain
        consistent with what existing training rows were collected under.
        """
        if len(history) < 2:
            return 0.0
        latest_ts = history[-1]["timestamp"]
        cutoff_ts = latest_ts - _FUNDING_LOOKBACK_MS
        old = next(
            (h for h in reversed(history[:-1]) if h["timestamp"] <= cutoff_ts),
            None,
        )
        if old is None:
            return 0.0  # No entry older than lookback window — trend unknown, report neutral
        return float(history[-1]["fundingRate"]) - float(old["fundingRate"])

    def _oi_delta_pct(self, prev_oi: float, curr_oi: float) -> float:
        if prev_oi == 0.0:
            return 0.0
        return (curr_oi - prev_oi) / prev_oi

    def _cvd_normalized(self, trades: list[dict]) -> float:
        """Cumulative volume delta normalized to [-1, 1]."""
        if not trades:
            return 0.0
        buy_vol = sum(t["amount"] for t in trades if t["side"] == "buy")
        sell_vol = sum(t["amount"] for t in trades if t["side"] == "sell")
        total = buy_vol + sell_vol
        if total == 0.0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def _basis_spread_pct(self, trades: list[dict]) -> float:
        """Approximation: last trade price minus BRTI estimate, as fraction of BRTI."""
        brti = self._get_brti_estimate()
        if not trades or brti is None or brti == 0.0:
            return 0.0
        last_price = float(trades[-1]["price"])
        return (last_price - brti) / brti

    def _large_print_direction(self, trades: list[dict]) -> float:
        if not trades:
            return 0.0
        avg_size = sum(t["amount"] for t in trades) / len(trades)
        threshold = 2 * avg_size
        large = [t for t in trades if t["amount"] > threshold]
        if not large:
            return 0.0
        buy_vol = sum(t["amount"] for t in large if t["side"] == "buy")
        sell_vol = sum(t["amount"] for t in large if t["side"] == "sell")
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0.0 else 0.0

    def _brti_volatility_1h(self) -> float:
        """Coefficient of variation of BRTI ticks in the last hour from Redis."""
        raw = self._redis.lrange("brti:ticks", 0, -1)
        if not raw:
            return 0.0
        now = time.time()
        cutoff = now - 3600
        prices = []
        for entry in raw:
            ts_str, price_str = entry.decode().split(":", 1)
            ts = float(ts_str)
            if ts >= cutoff:
                prices.append(float(price_str))
        if len(prices) < 2:
            return 0.0
        arr = np.array(prices)
        return float(np.std(arr, ddof=1) / np.mean(arr))

    def _get_brti_estimate(self) -> float | None:
        val = self._redis.get("brti:resolution_estimate")
        return float(val) if val else None

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

    async def _fetch_eth_direction(self) -> float:
        """Previous closed ETH/USDT 15-min candle direction.

        Returns 1.0 (up), 0.0 (down), or 0.5 (unknown / insufficient data).
        Uses the ccxt exchange that is already resolved for funding/OI calls.
        """
        try:
            ohlcv = await self._exchange.fetch_ohlcv("ETH/USDT:USDT", "15m", limit=3)
            if len(ohlcv) < 2:
                return 0.5
            prev = ohlcv[-2]   # -1 may be the currently-open candle; assumes exchange includes live candle in response
            return 1.0 if prev[4] > prev[1] else 0.0   # close > open
        except Exception as exc:
            logger.debug(f"DerivativesFeed: ETH direction fetch failed — {exc}")
            return 0.5

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

    # ── Redis write ────────────────────────────────────────────────────────────

    def _write_features(self, features: dict, okx_partial: bool = False) -> None:
        # Embed the partial flag in the primary key so fusion.py can detect it.
        payload = dict(features)
        if okx_partial:
            payload["_okx_partial"] = True
        self._redis.set("regime:features", json.dumps(payload), ex=_FEATURES_TTL)

        # Only update LKG with clean data — do NOT overwrite a good LKG with zeros.
        # CVD comes from OKX/Kraken trades (not funding/OI), so it is valid even on
        # a partial write and the ring buffer is always updated.
        if not okx_partial:
            lkg_payload = dict(features)
            lkg_payload["_lkg_written_at"] = time.time()
            self._redis.set("regime:features:lkg", json.dumps(lkg_payload), ex=_LKG_TTL)

        # Use timestamp as the unique member key so duplicate CVD values don't
        # collapse into one entry. The old scheme used str(cvd_value) as the key,
        # meaning repeated 0.0 writes (during exchange outages) could reduce the
        # set below the 5-entry minimum and trigger a silent stale cascade.
        cvd_value = features.get("cvd_normalized", 0.0)
        now_ts = time.time()
        self._redis.zadd("regime:cvd_history", {f"{now_ts:.3f}:{float(cvd_value):.6f}": now_ts})
        self._redis.zremrangebyscore("regime:cvd_history", 0, now_ts - 7200)
        self._redis.zremrangebyrank("regime:cvd_history", 0, -91)
