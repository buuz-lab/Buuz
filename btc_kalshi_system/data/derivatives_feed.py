import asyncio
import json
import time

import aiohttp
import numpy as np
import redis
from loguru import logger

from config import COINGLASS_API_KEY, HYPERLIQUID_BASE_URL, KRAKEN_FUTURES_BASE_URL, REDIS_URL
from btc_kalshi_system.data.fear_greed import fetch_fear_greed

_REFRESH_INTERVAL = 300   # 5 minutes
_FEATURES_TTL = 600       # 2x refresh interval — tolerates one missed cycle without expiring
_LKG_TTL = 86_400         # 24 hours — last-known-good survives multi-hour exchange outages
_FUNDING_LOOKBACK_MS = 4 * 3600_000  # 4 hours in milliseconds
_SYMBOL = "BTC/USDT:USDT"
_KRAKEN_SYMBOL = "BTC/USD"
_COINGLASS_BASE = "https://open-api-v4.coinglass.com"
_HYPERLIQUID_BASE = HYPERLIQUID_BASE_URL
_KRAKEN_FUTURES_BASE = KRAKEN_FUTURES_BASE_URL


class DerivativesFeed:
    """
    Pulls Binance perpetual-futures data via ccxt and writes six regime
    features to Redis key "regime:features" with a 300-second TTL.

    Refreshes every 5 minutes in an async loop.
    """

    # Exchange preference order — first one that connects without a 403/geo-block wins.
    # Bybit geo-blocks US users via CloudFront (HTTP 403).
    # OKX is the fallback: same perp futures data, accessible from the US.
    _EXCHANGE_PREFERENCE = ["okx", "bybit"]

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        import ccxt.async_support as ccxt_async
        self._redis = redis.from_url(redis_url)
        self._ccxt_async = ccxt_async
        self._exchange = None   # resolved lazily on first fetch
        self._exchange_name: str = ""
        self._prev_oi: dict[str, float] = {"okx": 0.0, "hyperliquid": 0.0, "kraken_futures": 0.0}
        self._kraken_exchange = None  # lazy init for Kraken trade fallback

    # ── Public entry point ─────────────────────────────────────────────────────

    async def _resolve_exchange(self) -> bool:
        """Try each exchange in preference order; set self._exchange to the first that works."""
        for name in self._EXCHANGE_PREFERENCE:
            try:
                ex = getattr(self._ccxt_async, name)({"enableRateLimit": True})
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
        """Refresh features every 5 minutes indefinitely."""
        try:
            while not await self._resolve_exchange():
                # No exchange available on startup — keep retrying rather than
                # running a permanent no-op loop. OKX may come back up.
                logger.warning(
                    "DerivativesFeed: no exchange available at startup — "
                    f"retrying in {_REFRESH_INTERVAL}s"
                )
                await asyncio.sleep(_REFRESH_INTERVAL)

            while True:
                success = False
                try:
                    features = await self._fetch_features()
                    okx_partial = features.pop("_okx_partial", False)
                    self._write_features(features, okx_partial=okx_partial)
                    logger.info(f"DerivativesFeed: wrote regime:features — {features}")
                    success = True
                except Exception as exc:
                    logger.warning(f"DerivativesFeed: fetch failed ({self._exchange_name}): {exc}")
                    # Any failure may indicate a dead session (timeout, reset, rate limit,
                    # geo-block, etc.) — always close and re-resolve to get a fresh instance.
                    if self._exchange is not None:
                        await self._exchange.close()
                    self._exchange = None
                    if not await self._resolve_exchange():
                        # All exchanges unavailable right now — don't exit, keep
                        # retrying so a temporary OKX maintenance window doesn't
                        # permanently kill the feed for the rest of the session.
                        logger.warning(
                            "DerivativesFeed: all exchanges unavailable — "
                            f"will retry in {_REFRESH_INTERVAL}s"
                        )
                        await asyncio.sleep(_REFRESH_INTERVAL)
                        continue
                # On success, refresh 60s early so the key (TTL=600s) is always
                # renewed with headroom to spare even if the fetch runs long.
                # On failure, wait the full interval before retrying.
                await asyncio.sleep(_REFRESH_INTERVAL - 60 if success else _REFRESH_INTERVAL)
        finally:
            if self._exchange is not None:
                await self._exchange.close()

    # ── Feature computation ────────────────────────────────────────────────────

    async def _fetch_features(self) -> dict:
        results = await asyncio.gather(
            self._fetch_funding_and_oi(),
            self._fetch_trades_data(),
            self._fetch_volume_ratio(),
        )
        curr_funding, trend, oi_delta, okx_partial = results[0]
        cvd, basis, large_print = results[1]
        volume_ratio = results[2]
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
        }
        if okx_partial:
            features["_okx_partial"] = True
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

    async def _fetch_trades_data(self) -> tuple[float, float, float]:
        """Returns (cvd_normalized, basis_spread_pct, large_print_direction).
        Tries OKX first; falls back to Kraken fetchTrades on any exception."""
        try:
            trades = await self._exchange.fetch_trades(_SYMBOL, limit=500)
            return self._cvd_normalized(trades), self._basis_spread_pct(trades), self._large_print_direction(trades)
        except Exception as exc:
            logger.warning(
                f"DerivativesFeed: OKX trades fetch failed — using Kraken fallback ({exc})"
            )
            return await self._kraken_trades_data()

    async def _kraken_trades_data(self) -> tuple[float, float, float]:
        if self._kraken_exchange is None:
            self._kraken_exchange = self._ccxt_async.kraken({"enableRateLimit": True})
        trades = await self._kraken_exchange.fetch_trades(_KRAKEN_SYMBOL, limit=500)
        return self._cvd_normalized(trades), self._basis_spread_pct(trades), self._large_print_direction(trades)

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
        """1h volume as a multiple of the 30-day hourly average. 1.0 = normal."""
        try:
            candles = await self._exchange.fetch_ohlcv(_SYMBOL, "1h", limit=721)
            if len(candles) < 30:
                return 1.0
            avg_volume = sum(c[5] for c in candles[:-1]) / len(candles[:-1])
            if avg_volume == 0:
                return 1.0
            current_volume = candles[-1][5]
            return round(current_volume / avg_volume, 3)
        except Exception as exc:
            logger.warning(f"DerivativesFeed: volume_ratio fetch failed — {exc}")
            return 1.0

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

        cvd_value = features.get("cvd_normalized", 0.0)
        self._redis.zadd("regime:cvd_history", {str(float(cvd_value)): time.time()})
        self._redis.zremrangebyscore("regime:cvd_history", 0, time.time() - 7200)
        self._redis.zremrangebyrank("regime:cvd_history", 0, -91)
