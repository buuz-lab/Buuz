import asyncio
import time

from btc_kalshi_system.data.models import Tick
from config import BRTI_STALE_THRESHOLD_SECONDS


class BRTIAggregator:
    """
    Merges exchange ticks into a volume-weighted composite BRTI price.
    CF Benchmarks plugs in by implementing _cf_benchmarks_source().
    """

    def __init__(self) -> None:
        self._latest: dict[str, Tick] = {}
        self._out_queue: asyncio.Queue[float] = asyncio.Queue()
        self._cf_price: float | None = None  # set by _drain_cf() when CF Benchmarks is implemented

    async def run(self, exchange_queues: list[asyncio.Queue]) -> None:
        """Run forever, draining all exchange queues concurrently. Never returns unless a queue raises."""
        await asyncio.gather(*[self._drain(q) for q in exchange_queues])

    async def _drain(self, queue: asyncio.Queue) -> None:
        while True:
            tick = await queue.get()
            self._latest[tick.exchange] = tick
            price = self._cf_benchmarks_source()
            if price is None:
                price = self._composite()
            if price is not None:
                await self._out_queue.put(price)

    def _composite(self) -> float | None:
        """
        Volume-weighted average of exchanges with fresh ticks.
        Falls back to simple average when all volumes are zero.
        Returns None if no fresh ticks are available.
        """
        now = time.time()
        fresh = {
            e: t for e, t in self._latest.items()
            if now - t.timestamp < BRTI_STALE_THRESHOLD_SECONDS
        }
        if not fresh:
            return None
        total_vol = sum(t.volume for t in fresh.values())
        if total_vol == 0.0:
            return sum(t.price for t in fresh.values()) / len(fresh)
        return sum(t.price * t.volume / total_vol for t in fresh.values())

    def _cf_benchmarks_source(self) -> float | None:
        """
        Primary BRTI from CF Benchmarks REST/WS API.
        Returns self._cf_price, which is None until a separate _drain_cf() coroutine
        (added when CF Benchmarks API key arrives) updates it from the WS feed.
        Reading a cached attribute here avoids hot-path async overhead on every exchange tick.
        """
        return self._cf_price

    @property
    def out_queue(self) -> asyncio.Queue:
        return self._out_queue
