"""
Integration smoke test — runs the full BRTI → Redis stack for N seconds.

Requires: Redis running at REDIS_URL, live internet.

Usage:
    python scripts/smoke_test.py --seconds 30

Exit code: 0 = pass, 1 = fail.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_kalshi_system.data.brti_aggregator import BRTIAggregator
from btc_kalshi_system.data.exchange_feed import BitstampFeed, CoinbaseFeed, GeminiFeed, KrakenFeed
from btc_kalshi_system.data.feature_store import FeatureStore
from btc_kalshi_system.data.models import Tick


async def run_smoke(seconds: int) -> bool:
    coinbase_q: asyncio.Queue[Tick] = asyncio.Queue()
    kraken_q:   asyncio.Queue[Tick] = asyncio.Queue()
    bitstamp_q: asyncio.Queue[Tick] = asyncio.Queue()
    gemini_q:   asyncio.Queue[Tick] = asyncio.Queue()

    agg = BRTIAggregator()
    store = FeatureStore()
    stop_event = asyncio.Event()

    async def timeout() -> None:
        await asyncio.sleep(seconds)
        stop_event.set()

    tasks = [
        asyncio.create_task(CoinbaseFeed().run(coinbase_q)),
        asyncio.create_task(KrakenFeed().run(kraken_q)),
        asyncio.create_task(BitstampFeed().run(bitstamp_q)),
        asyncio.create_task(GeminiFeed().run(gemini_q)),
        asyncio.create_task(agg.run([coinbase_q, kraken_q, bitstamp_q, gemini_q])),
        asyncio.create_task(store.run(agg.out_queue)),
        asyncio.create_task(timeout()),
    ]

    print(f"Running full BRTI → Redis stack for {seconds}s...")
    await stop_event.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    passed = True

    est = store.get_resolution_estimate()
    if est is None:
        print("FAIL  resolution_estimate is None — no ticks received in 60s window")
        passed = False
    else:
        print(f"PASS  resolution_estimate = ${est:,.2f}")

    tick_count = len(store._tick_buffer)
    if tick_count == 0:
        print("FAIL  tick buffer is empty — no prices processed")
        passed = False
    else:
        print(f"PASS  {tick_count} ticks in buffer")

    contributed = set(agg._latest.keys())
    if len(contributed) == 0:
        print("FAIL  no exchanges contributed ticks")
        passed = False
    else:
        status = "PASS" if len(contributed) >= 2 else "WARN"
        print(f"{status}  {len(contributed)} exchange(s) contributed: {contributed}")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    ok = asyncio.run(run_smoke(args.seconds))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
