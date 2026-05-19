import asyncio
import json
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
    REDIS_TTL_OHLCV,  # kept in config; no longer applied to candle hashes
    REDIS_TTL_RESOLUTION_ESTIMATE,
    REDIS_URL,
)

_FREQ_MAP = {"5min": "5min", "15min": "15min", "1h": "1h"}
_PERIOD_SECONDS = {"5min": 300, "15min": 900, "1h": 3600}


class FeatureStore:
    """
    Async writer: consumes float prices from BRTIAggregator.out_queue,
    maintains a tick deque, and writes to Redis on every new price.

    Sync read API: get_resolution_estimate(), get_ohlcv(), get_raw_ticks()
    called by Kronos engine and signal fusion.

    Completed candles are frozen in a Redis hash (brti:candles:{tf}) and
    never overwritten, so the candle count is monotonically non-decreasing.

    volume/amount columns are 0.0 in Phase 1 — the composite feed provides no
    per-tick volume. This is a known limitation noted in the design spec.
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._tick_buffer: deque[tuple[float, float]] = deque(maxlen=BRTI_TICK_BUFFER_SIZE)
        self._redis = redis.from_url(redis_url)
        self._completed_candles: dict[str, dict[str, dict]] = {tf: {} for tf in OHLCV_TIMEFRAMES}
        self._load_tick_buffer_from_redis()
        self._load_completed_candles_from_redis()

    # ── Lazy init (for test helpers that bypass __init__) ──────────────────

    def _ensure_completed_candles(self) -> None:
        if not hasattr(self, "_completed_candles"):
            self._completed_candles = {tf: {} for tf in OHLCV_TIMEFRAMES}

    # ── Startup loaders ───────────────────────────────────────────────────

    def _load_tick_buffer_from_redis(self) -> None:
        """Reload tick history from Redis on startup so restarts don't lose accumulated candles."""
        try:
            raw = self._redis.lrange("brti:ticks", 0, -1)
            if not raw:
                return
            # brti:ticks is newest-first; _tick_buffer must be oldest-first
            entries = []
            for entry in reversed(raw):
                try:
                    ts_str, price_str = entry.decode().split(":", 1)
                    entries.append((float(ts_str), float(price_str)))
                except (ValueError, AttributeError):
                    continue
            self._tick_buffer.extend(entries)
            logger.info(f"FeatureStore: reloaded {len(entries)} ticks from Redis on startup")
        except Exception as exc:
            logger.warning(f"FeatureStore: could not reload tick buffer from Redis — {exc}")

    def _load_completed_candles_from_redis(self) -> None:
        """Reload frozen candles from Redis so completed bars survive restarts."""
        for tf in OHLCV_TIMEFRAMES:
            try:
                raw = self._redis.hgetall(f"brti:candles:{tf}")
                if not raw:
                    continue
                for key, val in raw.items():
                    ts_str = key.decode() if isinstance(key, bytes) else key
                    bar = json.loads(val.decode() if isinstance(val, bytes) else val)
                    self._completed_candles[tf][ts_str] = bar
                n = len(self._completed_candles[tf])
                logger.info(f"FeatureStore: loaded {n} completed {tf} candles from Redis")
            except Exception as exc:
                logger.warning(f"FeatureStore: could not load {tf} candles from Redis — {exc}")

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

        pipe.execute()

        # Trigger incremental completed-candle writes for all timeframes.
        # hset calls are handled inside _resample; no TTL — candles are permanent.
        for tf in OHLCV_TIMEFRAMES:
            self._resample(tf)

    # ── Synchronous read API ───────────────────────────────────────────────

    def get_resolution_estimate(self) -> float | None:
        """60s rolling BRTI average — mirrors Kalshi resolution logic exactly."""
        val = self._redis.get("brti:resolution_estimate")
        return float(val) if val else None

    def get_ohlcv(self, timeframe: str) -> pd.DataFrame | None:
        """
        OHLCV DataFrame in Kronos format: [open, high, low, close, volume, amount].
        Reconstructed from frozen completed candles plus the current in-progress
        candle derived from the live tick buffer.
        Returns None if no data has accumulated.
        """
        self._ensure_completed_candles()
        return self._resample(timeframe)

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

    def candle_count(self, timeframe: str) -> int:
        """
        Number of candles currently available for this timeframe.
        Monotonically non-decreasing — completed candles are never evicted.
        """
        self._ensure_completed_candles()
        df = self._resample(timeframe)
        return len(df) if df is not None else 0

    # ── Internal ──────────────────────────────────────────────────────────

    def _resolution_estimate(self) -> float | None:
        now = time.time()
        recent = [p for ts, p in self._tick_buffer if now - ts <= BRTI_RESOLUTION_WINDOW_SECONDS]
        return float(np.mean(recent)) if recent else None

    def _resample(self, timeframe: str) -> pd.DataFrame | None:
        """
        Build and return a DataFrame of all candles for this timeframe.

        Completed candles (window end < now) are frozen on first observation
        and written to the Redis hash brti:candles:{timeframe} with no TTL.
        The current in-progress candle (if any) is derived fresh from live
        ticks and never persisted until its window closes.
        """
        self._ensure_completed_candles()
        completed = self._completed_candles[timeframe]
        period_secs = _PERIOD_SECONDS[timeframe]
        now_utc = pd.Timestamp.now(tz="UTC")

        in_progress_ts = None
        in_progress_bar = None

        if len(self._tick_buffer) >= 2:
            timestamps, prices = zip(*self._tick_buffer)
            tick_df = pd.DataFrame(
                {"price": list(prices)},
                index=pd.to_datetime(list(timestamps), unit="s", utc=True),
            )
            ohlcv = tick_df["price"].resample(_FREQ_MAP[timeframe]).agg(
                open="first", high="max", low="min", close="last"
            ).dropna()

            for ts, row in ohlcv.iterrows():
                window_end = ts + pd.Timedelta(seconds=period_secs)
                ts_str = ts.isoformat()
                bar = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": 0.0,
                    "amount": 0.0,
                }
                if window_end <= now_utc:
                    # Completed window — freeze and persist if not already stored
                    if ts_str not in completed:
                        completed[ts_str] = bar
                        try:
                            self._redis.hset(
                                f"brti:candles:{timeframe}",
                                ts_str,
                                json.dumps(bar),
                            )
                        except Exception as exc:
                            logger.warning(
                                f"FeatureStore: failed to persist {timeframe} candle {ts_str}: {exc}"
                            )
                else:
                    # In-progress window — keep latest, never persist
                    in_progress_ts = ts
                    in_progress_bar = bar

        if not completed and in_progress_ts is None:
            return None

        # Build sorted list: completed (ascending) + in-progress at end
        rows: list[tuple[pd.Timestamp, dict]] = [
            (pd.Timestamp(ts_str), bar) for ts_str, bar in completed.items()
        ]
        if in_progress_ts is not None:
            rows.append((in_progress_ts, in_progress_bar))

        rows.sort(key=lambda x: x[0])

        index = pd.DatetimeIndex([r[0] for r in rows])
        data = {col: [r[1][col] for r in rows] for col in ["open", "high", "low", "close", "volume", "amount"]}
        result = pd.DataFrame(data, index=index)

        # Guarantee UTC-localized index regardless of how timestamps were stored
        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC")
        elif str(result.index.tz) != "UTC":
            result.index = result.index.tz_convert("UTC")

        return result
