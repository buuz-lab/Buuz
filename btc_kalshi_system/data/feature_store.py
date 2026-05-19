import asyncio
import io
import time
from collections import deque

import numpy as np
import pandas as pd
import redis
from loguru import logger

from config import (
    BRTI_RESOLUTION_WINDOW_SECONDS,
    BRTI_TICK_BUFFER_SIZE,
    OHLCV_TIMEFRAMES,
    REDIS_TTL_OHLCV,
    REDIS_TTL_RESOLUTION_ESTIMATE,
    REDIS_URL,
)

_FREQ_MAP = {"5min": "5min", "15min": "15min", "1h": "1h"}


class FeatureStore:
    """
    Async writer: consumes float prices from BRTIAggregator.out_queue,
    maintains a tick deque, and writes to Redis on every new price.

    Sync read API: get_resolution_estimate(), get_ohlcv(), get_raw_ticks()
    called by Kronos engine and signal fusion.

    volume/amount columns are 0.0 in Phase 1 — the composite feed provides no
    per-tick volume. This is a known limitation noted in the design spec.
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._tick_buffer: deque[tuple[float, float]] = deque(maxlen=BRTI_TICK_BUFFER_SIZE)
        self._redis = redis.from_url(redis_url)

    # ── Async writer ───────────────────────────────────────────────────────

    async def run(self, price_queue: asyncio.Queue) -> None:
        while True:
            price = await price_queue.get()
            self._tick_buffer.append((time.time(), price))
            try:
                self._flush_to_redis()
            except Exception as exc:
                logger.warning(f"Redis flush failed: {exc}")

    def _flush_to_redis(self) -> None:
        if not self._tick_buffer:
            return
        ts, price = self._tick_buffer[-1]
        pipe = self._redis.pipeline()

        pipe.lpush("brti:ticks", f"{ts}:{price}")
        pipe.ltrim("brti:ticks", 0, BRTI_TICK_BUFFER_SIZE - 1)

        est = self._resolution_estimate()
        if est is not None:
            pipe.set("brti:resolution_estimate", est, ex=REDIS_TTL_RESOLUTION_ESTIMATE)

        for tf in OHLCV_TIMEFRAMES:
            df = self._resample(tf)
            if df is not None:
                pipe.set(f"brti:ohlcv:{tf}", df.to_json(), ex=REDIS_TTL_OHLCV[tf])

        pipe.execute()

    # ── Synchronous read API ───────────────────────────────────────────────

    def get_resolution_estimate(self) -> float | None:
        """60s rolling BRTI average — mirrors Kalshi resolution logic exactly."""
        val = self._redis.get("brti:resolution_estimate")
        return float(val) if val else None

    def get_ohlcv(self, timeframe: str) -> pd.DataFrame | None:
        """
        OHLCV DataFrame in Kronos format: [open, high, low, close, volume, amount].
        Returns None if insufficient data has accumulated since startup.
        """
        raw = self._redis.get(f"brti:ohlcv:{timeframe}")
        if not raw:
            return None
        df = pd.read_json(io.StringIO(raw.decode()))
        df.index = df.index.tz_localize("UTC")
        return df

    def get_raw_ticks(self, n_seconds: int) -> pd.Series | None:
        """Last n_seconds of BRTI prices as pd.Series indexed by UTC timestamp."""
        now = time.time()
        ticks = [(ts, p) for ts, p in self._tick_buffer if now - ts <= n_seconds]
        if not ticks:
            return None
        timestamps, prices = zip(*ticks)
        return pd.Series(
            list(prices),
            index=pd.to_datetime(list(timestamps), unit="s", utc=True),
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _resolution_estimate(self) -> float | None:
        now = time.time()
        recent = [p for ts, p in self._tick_buffer if now - ts <= BRTI_RESOLUTION_WINDOW_SECONDS]
        return float(np.mean(recent)) if recent else None

    def _resample(self, timeframe: str) -> pd.DataFrame | None:
        if len(self._tick_buffer) < 2:
            return None
        timestamps, prices = zip(*self._tick_buffer)
        df = pd.DataFrame(
            {"price": list(prices)},
            index=pd.to_datetime(list(timestamps), unit="s", utc=True),
        )
        ohlcv = df["price"].resample(_FREQ_MAP[timeframe]).agg(
            open="first", high="max", low="min", close="last"
        ).dropna()
        if ohlcv.empty:
            return None
        ohlcv["volume"] = 0.0
        ohlcv["amount"] = 0.0
        return ohlcv[["open", "high", "low", "close", "volume", "amount"]]
