"""WebSocket-backed orderbook cache for Kalshi markets.

Maintains a local OrderbookManager per ticker, fed by the pykalshi AsyncFeed.
get_orderbook() returns cached data (eliminating ~300ms REST round-trip) or
None if the cache is absent/stale, triggering a REST fallback in the router.
"""

import asyncio
import dataclasses
import threading
import time

from loguru import logger

from pykalshi import AsyncKalshiClient, OrderbookManager, OrderbookSnapshotMessage


_WS_STALE_SECONDS: float = 10.0
_DRAIN_INTERVAL_SECONDS: float = 1.0


@dataclasses.dataclass
class _TickerState:
    manager: OrderbookManager
    has_snapshot: bool
    last_update_ts: float


class KalshiOrderbookFeed:
    """WebSocket orderbook cache.

    Call run() as a long-lived asyncio task.  Call update_subscriptions() each
    cycle (thread-safe) to tell the feed which tickers are active.  Call
    get_orderbook() from any thread to read cached data.
    """

    def __init__(self, api_key_id: str, private_key_path: str) -> None:
        self._api_key_id = api_key_id
        self._private_key_path = private_key_path

        self._lock = threading.Lock()
        self._books: dict[str, _TickerState] = {}
        self._desired_tickers: set[str] = set()
        # First-snapshot capture per ticker: {mid_prob, spread, depth_bid, depth_ask, ts}
        # Never overwritten after first capture — survives unsubscribe for candle logger lookups.
        self._open_snapshots: dict[str, dict] = {}

        # Only accessed from the event loop — no lock needed.
        self._subscribed: set[str] = set()

        self._ws_reads: int = 0
        self._rest_reads: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop: connect, subscribe, process deltas.

        Probes the WS connection once on startup. If Kalshi returns 401
        (account lacks WS access), logs a one-time warning and exits cleanly
        so the router keeps using REST without retrying indefinitely.
        """
        client = AsyncKalshiClient(
            api_key_id=self._api_key_id,
            private_key_path=self._private_key_path,
        )
        feed = client.feed()

        # Probe: attempt one connection before entering the message loop.
        # A 401 here means the account doesn't have WS access — bail out
        # immediately rather than hammering the endpoint every 30s forever.
        try:
            await asyncio.wait_for(feed.connect(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(
                "WS orderbook: connection timed out — falling back to REST for all orderbook reads"
            )
            return
        except Exception as exc:
            logger.warning(
                f"WS orderbook: connection failed ({type(exc).__name__}: {exc}) — "
                "this account may not have WebSocket API access. "
                "All orderbook reads will use REST. Contact Kalshi support to enable WS access."
            )
            return

        logger.info("WS orderbook: connected — subscriptions will start on next cycle")

        @feed.on("orderbook_delta")
        def _handle(msg):
            ticker = msg.market_ticker
            with self._lock:
                state = self._books.get(ticker)
            if state is None:
                return
            now = time.time()
            if isinstance(msg, OrderbookSnapshotMessage):
                with self._lock:
                    state.manager.apply_snapshot(msg.yes_dollars, msg.no_dollars)
                    state.has_snapshot = True
                    state.last_update_ts = now
                    # Capture opening snapshot once per ticker (first snapshot only).
                    if ticker not in self._open_snapshots:
                        yes_lvls = sorted(state.manager.yes.items(), key=lambda x: float(x[0]))
                        no_lvls  = sorted(state.manager.no.items(),  key=lambda x: float(x[0]))
                        if yes_lvls and no_lvls:
                            best_yes_bid = float(yes_lvls[-1][0])
                            best_no_bid  = float(no_lvls[-1][0])
                            best_yes_ask = 100.0 - best_no_bid
                            self._open_snapshots[ticker] = {
                                "mid_prob":   (best_yes_bid + best_yes_ask) / 200.0,
                                "spread":     (best_yes_ask - best_yes_bid) / 100.0,
                                "depth_bid":  float(yes_lvls[-1][1]),
                                "depth_ask":  float(no_lvls[-1][1]),
                                "ts":         now,
                            }
                logger.debug(f"WS orderbook: snapshot received for {ticker}")
            else:
                with self._lock:
                    state.manager.apply_delta(msg.side, msg.price_dollars, msg.delta_fp)
                    state.last_update_ts = now

        drain_task = asyncio.create_task(self._drain_loop(feed))
        try:
            async for _ in feed:
                pass
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass

    def update_subscriptions(self, active_tickers: set[str]) -> None:
        """Update desired subscriptions.  Thread-safe; called from sync context."""
        with self._lock:
            self._desired_tickers = set(active_tickers)
            for ticker in active_tickers:
                if ticker not in self._books:
                    self._books[ticker] = _TickerState(
                        manager=OrderbookManager(ticker),
                        has_snapshot=False,
                        last_update_ts=0.0,
                    )

    def get_orderbook(self, ticker: str) -> dict | None:
        """Return cached orderbook in orderbook_fp format, or None if unavailable.

        Returns None when: not subscribed, snapshot not yet received, or last
        update was more than 10 seconds ago (forces REST fallback in router).

        Format matches what main._parse_orderbook() already handles:
            {"orderbook_fp": {"yes_dollars": [[price, qty], ...],
                              "no_dollars":  [[price, qty], ...]}}
        Both lists are in ascending price order (best bid/ask is the last entry).
        """
        with self._lock:
            state = self._books.get(ticker)
            if state is None or not state.has_snapshot:
                return None
            if time.time() - state.last_update_ts > _WS_STALE_SECONDS:
                return None

            yes_levels = sorted(state.manager.yes.items(), key=lambda x: float(x[0]))
            no_levels = sorted(state.manager.no.items(), key=lambda x: float(x[0]))

        self._ws_reads += 1
        self._log_read_stats()
        logger.debug(f"WS orderbook cache hit for {ticker}")

        return {
            "orderbook_fp": {
                "yes_dollars": [[p, q] for p, q in yes_levels],
                "no_dollars": [[p, q] for p, q in no_levels],
            }
        }

    def get_open_snapshot(self, ticker: str) -> dict | None:
        """Return the first orderbook snapshot captured for this ticker, or None.

        Keys: mid_prob (float, 0-1), spread (float, 0-1), depth_bid, depth_ask, ts.
        Safe to call from any thread after the contract has opened.
        """
        with self._lock:
            return self._open_snapshots.get(ticker)

    def count_rest_fallback(self) -> None:
        """Called by the router when WS returned None and REST was used instead."""
        self._rest_reads += 1
        self._log_read_stats()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log_read_stats(self) -> None:
        total = self._ws_reads + self._rest_reads
        if total > 0 and total % 100 == 0:
            logger.info(
                f"Orderbook reads (last 100): WS={self._ws_reads} REST={self._rest_reads}"
            )

    async def _drain_loop(self, feed) -> None:
        """Periodically sync feed subscriptions and detect reconnects."""
        last_reconnect_count = 0
        while True:
            await asyncio.sleep(_DRAIN_INTERVAL_SECONDS)

            current_reconnect = feed.reconnect_count
            if current_reconnect != last_reconnect_count:
                last_reconnect_count = current_reconnect
                with self._lock:
                    for state in self._books.values():
                        state.has_snapshot = False
                logger.info(
                    f"WS orderbook: reconnect #{current_reconnect} — "
                    "resetting snapshot state for all tickers"
                )

            self._apply_drain(feed)

    def _apply_drain(self, feed) -> None:
        """Subscribe new desired tickers and unsubscribe removed ones.

        Called from within the event loop (drain_loop), so feed.subscribe() /
        feed.unsubscribe() are safe to call here.
        """
        with self._lock:
            desired = self._desired_tickers.copy()

        to_sub = desired - self._subscribed
        to_unsub = self._subscribed - desired

        for ticker in sorted(to_sub):
            feed.subscribe("orderbook_delta", market_ticker=ticker)
            self._subscribed.add(ticker)
            logger.debug(f"WS orderbook: subscribed {ticker}")

        for ticker in sorted(to_unsub):
            feed.unsubscribe("orderbook_delta", market_ticker=ticker)
            self._subscribed.discard(ticker)
            with self._lock:
                self._books.pop(ticker, None)
            logger.debug(f"WS orderbook: unsubscribed {ticker}")
