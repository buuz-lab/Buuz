import asyncio
import json
import signal
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from btc_kalshi_system.data.brti_aggregator import BRTIAggregator
from btc_kalshi_system.data.derivatives_feed import DerivativesFeed
from btc_kalshi_system.data.exchange_feed import BitstampFeed, CoinbaseFeed, GeminiFeed, KrakenFeed
from btc_kalshi_system.data.feature_store import FeatureStore
from btc_kalshi_system.execution.kelly import KellySizer
from btc_kalshi_system.execution.pretrade_checklist import PreTradeChecklist
from btc_kalshi_system.execution.router import KalshiClientRouter
from btc_kalshi_system.models.calibrator import Calibrator
from btc_kalshi_system.models.deepseek_parser import DeepSeekContextParser
from btc_kalshi_system.models.kronos_engine import KronosEngine
from btc_kalshi_system.models.regime_model import RegimeModel
from btc_kalshi_system.portfolio.circuit_breaker import CircuitBreaker
from btc_kalshi_system.portfolio.monitor import OpenPosition, PortfolioMonitor
from btc_kalshi_system.signal.edge_tracker import EdgeTracker
from btc_kalshi_system.signal.fusion import SignalFusionEngine
import config

# ── Logging ───────────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stderr, level="INFO")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/kronos_{time}.log", rotation="1 day", retention="30 days", level="DEBUG")

# ── Timing ────────────────────────────────────────────────────────────────────

SIGNAL_INTERVAL_SECONDS = 300
DEEPSEEK_REFRESH_SECONDS = 900
RECOVERY_INTERVAL_SECONDS = 3600
RESOLUTION_BLACKOUT_MINUTES = 15

# KXBTC resolves at 6:30 PM EDT (UTC-4 during DST) = 22:30 UTC
RESOLUTION_TIMES_EDT = [(18, 30)]
_EDT_OFFSET_HOURS = 4  # EDT = UTC-4

# ── SQLite schema ─────────────────────────────────────────────────────────────

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    timestamp TEXT,
    ticker TEXT,
    timeframe TEXT,
    direction INTEGER,
    strike REAL,
    market_price REAL,
    kelly_dollars REAL,
    kelly_contracts INTEGER,
    kronos_raw REAL,
    kronos_calibrated REAL,
    regime_prob REAL,
    deepseek_regime TEXT,
    fill_price_cents INTEGER,
    outcome INTEGER DEFAULT NULL,
    pnl_dollars REAL DEFAULT NULL
)
"""


class KronosV2:
    def __init__(self) -> None:
        self._store = FeatureStore()
        self._kronos = KronosEngine()
        self._calibrator = Calibrator()
        self._regime = RegimeModel()
        self._deepseek = DeepSeekContextParser()
        self._fusion = SignalFusionEngine(
            feature_store=self._store,
            kronos_engine=self._kronos,
            calibrator=self._calibrator,
            regime_model=self._regime,
            deepseek_parser=self._deepseek,
        )
        self._edge_tracker = EdgeTracker()
        self._kelly = KellySizer()
        self._checklist = PreTradeChecklist(self._kelly)
        self._router = KalshiClientRouter()
        self._monitor = PortfolioMonitor()
        self._breaker = CircuitBreaker(
            monitor=self._monitor,
            edge_tracker=self._edge_tracker,
            router=self._router,
            calibrator=self._calibrator,
        )

        self._db = sqlite3.connect("trades.db", check_same_thread=False)
        self._db.execute(_CREATE_TRADES_TABLE)
        self._db.commit()

        self._running = False
        self._last_deepseek_refresh = 0.0
        self._last_recovery_attempt = 0.0

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        if config.PAPER_TRADING:
            logger.info("Running in PAPER TRADING mode — no real orders will be placed")
        logger.info("KronosV2 starting up")

        queues = [asyncio.Queue() for _ in range(4)]
        agg = BRTIAggregator()
        feeds = [CoinbaseFeed(), KrakenFeed(), BitstampFeed(), GeminiFeed()]
        deriv = DerivativesFeed()

        await asyncio.gather(
            feeds[0].run(queues[0]),
            feeds[1].run(queues[1]),
            feeds[2].run(queues[2]),
            feeds[3].run(queues[3]),
            agg.run(queues),
            self._store.run(agg.out_queue),
            deriv.run(),
            self._main_loop(),
        )

    async def _main_loop(self) -> None:
        while self._running:
            loop_start = time.time()
            try:
                self._run_cycle()
            except Exception as exc:
                logger.error(f"Unhandled error in main loop cycle: {exc}")
            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0, SIGNAL_INTERVAL_SECONDS - elapsed))
        logger.info("KronosV2 main loop stopped")

    def stop(self) -> None:
        logger.info("KronosV2 stopping")
        self._running = False

    # ── Main cycle ────────────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        # 1. Circuit breaker
        status = self._breaker.check()
        if status.tripped:
            logger.warning(f"Circuit breaker active [{status.reason.value}]: {status.message} — skipping cycle")
            return

        # 2. Refresh DeepSeek / regime context on its own cadence (every 15 min),
        #    then always load the latest from Redis for this cycle.
        now = time.time()
        ctx = self._get_market_context()
        if now - self._last_deepseek_refresh >= DEEPSEEK_REFRESH_SECONDS:
            self._last_deepseek_refresh = now
            logger.debug("DeepSeek context refreshed")
        # Always push the latest context once per cycle
        self._fusion.update_market_context(ctx)

        # 4. Composite price
        composite_price = self._get_composite_price()

        # 5. Active markets
        markets = self._get_active_markets()
        if not markets:
            logger.info("No active BTC markets found")
            return

        # 6. Process each market
        for market in markets:
            try:
                self._process_market(market, composite_price)
            except Exception as exc:
                ticker = market.get("ticker", "unknown")
                logger.error(f"Error processing market {ticker}: {exc}")

        # 7. Check for resolutions
        try:
            self._check_resolutions()
        except Exception as exc:
            logger.error(f"Error checking resolutions: {exc}")

    def _process_market(self, market: dict, composite_price: float) -> None:
        ticker = market.get("ticker", "")
        if not ticker:
            return

        # a. Resolution blackout
        if self._is_in_blackout():
            logger.info(f"In resolution blackout — skipping {ticker}")
            return

        # b. Strike from market data
        strike = self._extract_strike(market)
        if strike is None:
            logger.debug(f"Could not determine strike for {ticker}, skipping")
            return

        timeframe = market.get("market_type", "15min")

        # c. Signal
        signal = self._fusion.get_signal(timeframe=timeframe, strike=strike)
        if signal is None:
            logger.debug(f"No signal for {ticker} (gated out)")
            return

        # d. Orderbook
        try:
            orderbook_resp = self._router.get_orderbook(ticker)
        except Exception as exc:
            logger.warning(f"Failed to fetch orderbook for {ticker}: {exc}")
            return

        # e. Parse orderbook
        best_bid_cents, best_ask_cents, available_contracts = self._parse_orderbook(orderbook_resp)
        if best_ask_cents == 0:
            logger.debug(f"Empty orderbook for {ticker}, skipping")
            return

        # f. Pre-trade checklist
        current_exposure = self._monitor.get_current_exposure()
        same_timeframe_open = self._monitor.has_timeframe_position(timeframe)
        # In paper mode Gate 4 is always open — edge tracker hasn't accumulated yet
        edge_above_threshold = True if config.PAPER_TRADING else self._edge_tracker.is_above_threshold()

        result = self._checklist.run(
            signal=signal,
            best_ask_cents=best_ask_cents,
            best_bid_cents=best_bid_cents,
            available_contracts=available_contracts,
            current_exposure=current_exposure,
            same_timeframe_open=same_timeframe_open,
            composite_price=composite_price,
            edge_above_threshold=edge_above_threshold,
        )

        # g. Checklist failed
        if not result.passed:
            logger.info(f"Pre-trade checklist failed for {ticker} [gate {result.failed_gate}]: {result.failed_reason}")
            return

        # h. Place order (or simulate in paper mode)
        trade_id = str(uuid.uuid4())
        side = "yes" if signal.direction == 1 else "no"
        if config.PAPER_TRADING:
            logger.info(
                f"[PAPER] Simulated fill: {ticker} {side} {result.kelly_contracts}@{best_ask_cents}¢ "
                f"(${result.kelly_dollars:.2f} kelly) trade_id={trade_id}"
            )
        else:
            try:
                self._router.place_order(
                    ticker=ticker,
                    side=side,
                    count=result.kelly_contracts,
                    price_cents=best_ask_cents,
                    client_order_id=trade_id,
                )
                logger.info(
                    f"Order placed: {ticker} {side} {result.kelly_contracts}@{best_ask_cents}¢ "
                    f"(${result.kelly_dollars:.2f} kelly) trade_id={trade_id}"
                )
            except Exception as exc:
                logger.error(f"Order placement failed for {ticker}: {exc}")
                return

        # i. Record position
        position = OpenPosition(
            trade_id=trade_id,
            ticker=ticker,
            timeframe=timeframe,
            direction=signal.direction,
            strike=strike,
            contracts=result.kelly_contracts,
            entry_price_cents=best_ask_cents,
            kelly_dollars=result.kelly_dollars,
            timestamp=time.time(),
            calibrated_prob=signal.calibrated_prob,
        )
        self._monitor.add_position(position)

        # j. Log to SQLite
        self._record_trade_sqlite(trade_id, signal, result, ticker, best_ask_cents)

    # ── Helper methods ────────────────────────────────────────────────────────

    def _is_in_blackout(self) -> bool:
        now_utc = datetime.now(timezone.utc)
        blackout_seconds = RESOLUTION_BLACKOUT_MINUTES * 60
        for hour_edt, minute_edt in RESOLUTION_TIMES_EDT:
            # Convert EDT → UTC
            hour_utc = (hour_edt + _EDT_OFFSET_HOURS) % 24
            resolution_today = now_utc.replace(
                hour=hour_utc, minute=minute_edt, second=0, microsecond=0
            )
            delta = (resolution_today - now_utc).total_seconds()
            if 0 <= delta <= blackout_seconds:
                return True
        return False

    def _get_market_context(self) -> dict:
        try:
            import redis as _redis
            r = _redis.from_url(config.REDIS_URL, decode_responses=True)
            raw_features = r.get("regime:features")
            if raw_features:
                return json.loads(raw_features)
        except Exception as exc:
            logger.debug(f"Failed to read regime features from Redis: {exc}")
        return {}

    def _get_active_markets(self) -> list[dict]:
        # KXBTC15M = 15-min BTC up/down; KXBTCD = hourly BTC above/below
        series = [("KXBTC15M", "15min"), ("KXBTCD", "1h")]
        markets: list[dict] = []
        for series_ticker, market_type in series:
            try:
                resp = self._router._raw._request(
                    "GET", f"/trade-api/v2/markets?series_ticker={series_ticker}&status=open"
                )
                for m in resp.get("markets", []):
                    m["market_type"] = market_type
                    markets.append(m)
            except Exception as exc:
                logger.warning(f"Failed to fetch {series_ticker} markets: {exc}")
        if not markets:
            logger.info("No active 15-min or hourly BTC markets found")
        return markets

    def _extract_strike(self, market: dict) -> float | None:
        for field in ("floor_strike", "cap_strike", "strike_price", "result_at_open"):
            val = market.get(field)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        # Parse from ticker: KXBTC-25JUN-T95000 → 95000.0
        ticker = market.get("ticker", "")
        for part in ticker.split("-"):
            if part.startswith("T") and part[1:].isdigit():
                return float(part[1:])
        # For up/down markets the threshold is "higher than current price",
        # so live BRTI price is the right substitute when no strike field is present.
        price = self._get_composite_price()
        if price > 0.0:
            logger.debug(
                f"No strike field found for {ticker} — using composite price {price:.2f} as fallback"
            )
            return price
        return None

    def _parse_orderbook(self, orderbook: dict) -> tuple[int, int, int]:
        try:
            book = orderbook.get("orderbook", orderbook)
            yes_bids = book.get("yes", [])   # [[price_cents, qty], ...] descending
            no_bids = book.get("no", [])     # [[price_cents, qty], ...] descending

            best_bid_cents = int(yes_bids[0][0]) if yes_bids else 0
            if no_bids:
                best_ask_cents = 100 - int(no_bids[0][0])
                available_contracts = int(no_bids[0][1])
            else:
                return (0, 0, 0)

            return (best_bid_cents, best_ask_cents, available_contracts)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.debug(f"Failed to parse orderbook: {exc}")
            return (0, 0, 0)

    def _get_composite_price(self) -> float:
        try:
            ohlcv = self._store.get_ohlcv("5min")
            if ohlcv is not None and not ohlcv.empty:
                return float(ohlcv.iloc[-1]["close"])
        except Exception:
            pass
        try:
            estimate = self._store.get_resolution_estimate()
            if estimate is not None:
                return estimate
        except Exception:
            pass
        return 0.0

    def _record_trade_sqlite(
        self,
        trade_id: str,
        signal,
        checklist_result,
        ticker: str,
        fill_price_cents: int,
    ) -> None:
        import math
        try:
            self._db.execute(
                """
                INSERT OR IGNORE INTO trades (
                    trade_id, timestamp, ticker, timeframe, direction, strike,
                    market_price, kelly_dollars, kelly_contracts,
                    kronos_raw, kronos_calibrated, regime_prob, deepseek_regime,
                    fill_price_cents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    signal.timestamp.isoformat(),
                    ticker,
                    signal.timeframe,
                    signal.direction,
                    signal.strike,
                    fill_price_cents / 100,
                    checklist_result.kelly_dollars,
                    checklist_result.kelly_contracts,
                    signal.kronos_raw,
                    signal.kronos_calibrated,
                    None if math.isnan(signal.regime_prob) else signal.regime_prob,
                    signal.deepseek_regime,
                    fill_price_cents,
                ),
            )
            self._db.commit()
        except sqlite3.Error as exc:
            logger.error(f"SQLite insert failed for {trade_id}: {exc}")

    def _check_resolutions(self) -> None:
        for position in self._monitor.get_open_positions():
            try:
                resp = self._router._raw._request(
                    "GET", f"/trade-api/v2/markets/{position.ticker}"
                )
                market = resp.get("market", resp)
                status = market.get("status", "")
                if status != "finalized":
                    continue

                result = market.get("result", "")
                if not result:
                    continue

                # result is "yes" or "no" — compare to our direction
                if result == "yes":
                    outcome = 1 if position.direction == 1 else 0
                else:
                    outcome = 1 if position.direction == 0 else 0

                resolved_at = time.time()
                trade = self._monitor.resolve_trade(
                    position.trade_id, outcome=outcome, resolved_at=resolved_at
                )
                if trade is None:
                    continue

                self._edge_tracker.record(
                    predicted_prob=position.calibrated_prob,
                    outcome=outcome,
                    market_price=position.entry_price_cents / 100,
                )

                try:
                    self._db.execute(
                        "UPDATE trades SET outcome=?, pnl_dollars=? WHERE trade_id=?",
                        (outcome, trade.pnl_dollars, position.trade_id),
                    )
                    self._db.commit()
                except sqlite3.Error as exc:
                    logger.error(f"SQLite update failed for {position.trade_id}: {exc}")

                # Refit calibrator with all resolved trades so n_samples grows
                try:
                    rows = self._db.execute(
                        "SELECT kronos_raw, outcome FROM trades WHERE outcome IS NOT NULL"
                    ).fetchall()
                    if rows:
                        import numpy as np
                        raw_probs = np.array([r[0] for r in rows], dtype=float)
                        outcomes = np.array([r[1] for r in rows], dtype=float)
                        self._calibrator.fit(raw_probs, outcomes)
                        logger.debug(f"Calibrator refit with {len(rows)} resolved trades")
                except Exception as exc:
                    logger.warning(f"Calibrator refit failed: {exc}")

                logger.info(
                    f"Trade resolved: {position.ticker} trade_id={position.trade_id} "
                    f"outcome={'WIN' if outcome == 1 else 'LOSS'} pnl=${trade.pnl_dollars:.2f}"
                )
            except Exception as exc:
                logger.warning(f"Failed to check resolution for {position.ticker}: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Route uncaught exceptions through loguru so they appear in the log file,
    # not just on stderr.  Without this, a crash in __init__ leaves empty logs.
    logger.catch(reraise=True)(lambda: None)()  # warm up the handler
    import sys as _sys
    _sys.excepthook = lambda exc_type, exc_val, exc_tb: logger.opt(exception=(exc_type, exc_val, exc_tb)).critical(
        "Uncaught exception — process is exiting"
    )

    try:
        system = KronosV2()
    except Exception:
        logger.exception("Fatal error during KronosV2 initialisation — see traceback above")
        raise

    def _handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig} — shutting down")
        system.stop()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    asyncio.run(system.run())


if __name__ == "__main__":
    main()
