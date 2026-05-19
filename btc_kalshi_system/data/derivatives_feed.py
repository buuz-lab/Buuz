import asyncio
import json
import time

import numpy as np
import redis
from loguru import logger

from config import REDIS_URL

_REFRESH_INTERVAL = 300   # 5 minutes
_FEATURES_TTL = 300       # Redis TTL matches refresh interval
_FUNDING_LOOKBACK_MS = 4 * 3600_000  # 4 hours in milliseconds
_SYMBOL = "BTC/USDT"


class DerivativesFeed:
    """
    Pulls Binance perpetual-futures data via ccxt and writes six regime
    features to Redis key "regime:features" with a 300-second TTL.

    Refreshes every 5 minutes in an async loop.
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        import ccxt.async_support as ccxt_async
        self._redis = redis.from_url(redis_url)
        self._exchange = ccxt_async.binance({"enableRateLimit": True})
        self._prev_oi: float = 0.0

    # ── Public entry point ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Refresh features every 5 minutes indefinitely."""
        try:
            while True:
                try:
                    features = await self._fetch_features()
                    self._write_features(features)
                    logger.info(f"DerivativesFeed: wrote regime:features — {features}")
                except Exception as exc:
                    logger.warning(f"DerivativesFeed: fetch failed: {exc}")
                await asyncio.sleep(_REFRESH_INTERVAL)
        finally:
            await self._exchange.close()

    # ── Feature computation ────────────────────────────────────────────────────

    async def _fetch_features(self) -> dict:
        funding_history, oi_data, trades = await asyncio.gather(
            self._exchange.fetch_funding_rate_history(_SYMBOL, limit=10),
            self._exchange.fetch_open_interest(_SYMBOL),
            self._exchange.fetch_trades(_SYMBOL, limit=500),
        )

        curr_funding = float(funding_history[-1]["fundingRate"]) if funding_history else 0.0
        trend = self._funding_rate_trend(funding_history)

        curr_oi = float(oi_data.get("openInterestAmount", 0.0))
        oi_delta = self._oi_delta_pct(self._prev_oi, curr_oi)
        self._prev_oi = curr_oi

        cvd = self._cvd_normalized(trades)
        basis = self._basis_spread_pct(trades)
        vol = self._brti_volatility_1h()

        return {
            "funding_rate":       curr_funding,
            "funding_rate_trend": trend,
            "oi_delta_pct":       oi_delta,
            "cvd_normalized":     cvd,
            "basis_spread_pct":   basis,
            "brti_volatility_1h": vol,
        }

    def _funding_rate_trend(self, history: list[dict]) -> float:
        """4-hour delta in funding rate. Returns 0.0 if insufficient history."""
        if len(history) < 2:
            return 0.0
        latest_ts = history[-1]["timestamp"]
        cutoff_ts = latest_ts - _FUNDING_LOOKBACK_MS
        old = next(
            (h for h in reversed(history[:-1]) if h["timestamp"] <= cutoff_ts),
            None,
        )
        if old is None:
            # Fallback: use the oldest entry we have
            old = history[0]
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

    # ── Redis write ────────────────────────────────────────────────────────────

    def _write_features(self, features: dict) -> None:
        self._redis.set("regime:features", json.dumps(features), ex=_FEATURES_TTL)
