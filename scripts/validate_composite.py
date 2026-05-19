"""
Run the BRTI composite feed for N minutes and print statistics.

Usage:
    python scripts/validate_composite.py --minutes 10
    python scripts/validate_composite.py --minutes 10 --csv /tmp/brti_ticks.csv
"""

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_kalshi_system.data.brti_aggregator import BRTIAggregator
from btc_kalshi_system.data.exchange_feed import BitstampFeed, CoinbaseFeed, KrakenFeed
from btc_kalshi_system.data.models import Tick


async def run_validation(minutes: int, csv_path: str | None) -> None:
    coinbase_q: asyncio.Queue[Tick] = asyncio.Queue()
    kraken_q:   asyncio.Queue[Tick] = asyncio.Queue()
    bitstamp_q: asyncio.Queue[Tick] = asyncio.Queue()

    agg = BRTIAggregator()
    composite_prices: list[float] = []
    exchange_tick_counts: dict[str, int] = {"coinbase": 0, "kraken": 0, "bitstamp": 0}
    tick_log: list[dict] = []
    stop_event = asyncio.Event()

    async def drain_exchange(name: str, queue: asyncio.Queue[Tick]) -> None:
        while True:
            tick = await queue.get()
            exchange_tick_counts[name] += 1
            agg._latest[tick.exchange] = tick
            price = agg._composite()
            if price is not None:
                await agg.out_queue.put(price)

    async def collect_composite() -> None:
        while True:
            price = await agg.out_queue.get()
            composite_prices.append(price)
            tick_log.append({"timestamp": time.time(), "composite": price})

    async def timeout() -> None:
        await asyncio.sleep(minutes * 60)
        stop_event.set()

    print(f"Running BRTI composite feed for {minutes} minute(s)...")
    print("Exchanges: Coinbase, Kraken, Bitstamp")
    print("-" * 50)

    tasks = [
        asyncio.create_task(CoinbaseFeed().run(coinbase_q)),
        asyncio.create_task(KrakenFeed().run(kraken_q)),
        asyncio.create_task(BitstampFeed().run(bitstamp_q)),
        asyncio.create_task(drain_exchange("coinbase", coinbase_q)),
        asyncio.create_task(drain_exchange("kraken", kraken_q)),
        asyncio.create_task(drain_exchange("bitstamp", bitstamp_q)),
        asyncio.create_task(collect_composite()),
        asyncio.create_task(timeout()),
    ]

    await stop_event.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    total = len(composite_prices)
    print(f"\nTicks received (composite): {total}")
    for name, count in exchange_tick_counts.items():
        print(f"  {name.capitalize()}: {count} ticks")

    if total > 0:
        print(f"Composite range: ${min(composite_prices):,.2f} – ${max(composite_prices):,.2f}")
        print(f"Final composite price:     ${composite_prices[-1]:,.2f}")
        window = composite_prices[-60:] if len(composite_prices) >= 60 else composite_prices
        print(f"Resolution estimate (last {len(window)} prices avg): ${sum(window)/len(window):,.2f}")
        latest_per_exchange = {
            e: t.price for e, t in agg._latest.items()
        }
        if len(latest_per_exchange) >= 2:
            spread = max(latest_per_exchange.values()) - min(latest_per_exchange.values())
            print(f"Final cross-exchange spread: ${spread:,.2f}")

    if csv_path and tick_log:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "composite"])
            writer.writeheader()
            writer.writerows(tick_log)
        print(f"\nTick log written to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BRTI composite feed")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(run_validation(args.minutes, args.csv))


if __name__ == "__main__":
    main()
