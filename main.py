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

# Per-market blackout: stop new entries this many seconds before close_time
_BLACKOUT_SECONDS = {"15min": 3 * 60, "1h": 10 * 60}

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
    pnl_dollars REAL DEFAULT NULL,
    -- Six regime features as-of signal-creation time. These columns are the
    -- training X matrix for RegimeModel once we have ≥500 non-stale rows.
    funding_rate REAL DEFAULT NULL,
    funding_rate_trend REAL DEFAULT NULL,
    oi_delta_pct REAL DEFAULT NULL,
    cvd_normalized REAL DEFAULT NULL,
    basis_spread_pct REAL DEFAULT NULL,
    brti_volatility_1h REAL DEFAULT NULL,
    -- 1 if the regime:features Redis read was empty/expired at trade time.
    -- Stale rows must be filtered out of the training set.
    features_stale INTEGER DEFAULT 0
)
"""

# Idempotent column additions for databases created before the regime-feature
# columns existed. SQLite's ALTER TABLE ADD COLUMN is fast (rewrites no rows)
# and the OperationalError swallow makes this safe to run on every startup.
_TRADES_COLUMN_MIGRATIONS = [
    ("funding_rate",        "REAL DEFAULT NULL"),
    ("funding_rate_trend",  "REAL DEFAULT NULL"),
    ("oi_delta_pct",        "REAL DEFAULT NULL"),
    ("cvd_normalized",      "REAL DEFAULT NULL"),
    ("basis_spread_pct",    "REAL DEFAULT NULL"),
    ("brti_volatility_1h",  "REAL DEFAULT NULL"),
    ("features_stale",      "INTEGER DEFAULT 0"),
]


class KronosV2:
    def __init__(self) -> None:
        self._store = FeatureStore()
        self._kronos = KronosEngine()
        self._calibrator = Calibrator()
        # Try to load a trained RegimeModel from disk. If the file doesn't exist
        # yet (e.g. system is still in bootstrap mode and hasn't collected enough
        # paper trades to train), fall back to an unfit instance. fusion.py's
        # NotTrainedError branch will then run Kronos-only with _BOOTSTRAP_SHRINK.
        try:
            self._regime = RegimeModel.load(config.REGIME_MODEL_PATH)
            logger.info(f"RegimeModel loaded from {config.REGIME_MODEL_PATH}")
        except FileNotFoundError:
            self._regime = RegimeModel()
            logger.info(
                f"RegimeModel file not found at {config.REGIME_MODEL_PATH} — "
                f"running in bootstrap mode (Kronos-only, Gate 2 bypassed)"
            )
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

        # Preload Kronos here — single-threaded, no event loop, no concurrent I/O.
        # Loading it later (inside asyncio.to_thread while WebSocket feeds run) causes
        # a segfault on Apple Silicon because PyTorch's Accelerate-framework init races
        # with macOS kqueue.  Once loaded, _load() becomes a no-op so the worker thread
        # call is safe.
        self._kronos.preload()

        self._db = sqlite3.connect("trades.db", check_same_thread=False)
        self._db.execute(_CREATE_TRADES_TABLE)
        # Apply additive schema migrations for pre-existing databases. New columns
        # default to NULL on existing rows — that's the correct signal that those
        # historical trades have no captured regime features and must be excluded
        # from training.
        for col_name, col_def in _TRADES_COLUMN_MIGRATIONS:
            try:
                self._db.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                # "duplicate column name" — column already exists. Safe to ignore.
                pass
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
                # Run the entire cycle in a thread-pool worker so PyTorch inference
                # (100 forward passes through the Kronos transformer) never blocks
                # the asyncio event loop thread.  On macOS, running PyTorch inside
                # the kqueue-based event loop thread causes a segfault because
                # PyTorch's Accelerate-framework internals conflict with kqueue's
                # dispatch queue.  Moving it off-thread eliminates the conflict.
                await asyncio.to_thread(self._run_cycle)
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
        if self._market_is_in_blackout(market):
            logger.info(f"Too close to close_time — skipping {ticker}")
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
        # "yes" contracts cost ask_cents; "no" contracts cost (100 - bid_cents).
        fill_price_cents = best_ask_cents if signal.direction == 1 else (100 - best_bid_cents)
        if config.PAPER_TRADING:
            logger.info(
                f"[PAPER] Simulated fill: {ticker} {side} {result.kelly_contracts}@{fill_price_cents}¢ "
                f"(${result.kelly_dollars:.2f} kelly) trade_id={trade_id}"
            )
        else:
            try:
                self._router.place_order(
                    ticker=ticker,
                    side=side,
                    count=result.kelly_contracts,
                    price_cents=fill_price_cents,
                    client_order_id=trade_id,
                )
                logger.info(
                    f"Order placed: {ticker} {side} {result.kelly_contracts}@{fill_price_cents}¢ "
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
            entry_price_cents=fill_price_cents,
            kelly_dollars=result.kelly_dollars,
            timestamp=time.time(),
            calibrated_prob=signal.calibrated_prob,
        )
        self._monitor.add_position(position)

        # j. Log to SQLite
        self._record_trade_sqlite(trade_id, signal, result, ticker, fill_price_cents)

    # ── Helper methods ────────────────────────────────────────────────────────

    def _market_is_in_blackout(self, market: dict) -> bool:
        """Return True if we are too close to this market's close_time to enter a new position."""
        close_time_str = market.get("close_time")
        if not close_time_str:
            return False
        try:
            close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            seconds_until_close = (close_dt - datetime.now(timezone.utc)).total_seconds()
            if seconds_until_close < 0:
                return True  # market already closed
            market_type = market.get("market_type", "15min")
            threshold = _BLACKOUT_SECONDS.get(market_type, 3 * 60)
            return seconds_until_close <= threshold
        except (ValueError, TypeError) as exc:
            logger.debug(f"Could not parse close_time '{close_time_str}': {exc}")
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
        series = [("KXBTC15M", "15min")]
        markets: list[dict] = []
        for series_ticker, market_type in series:
            try:
                resp = self._router._raw._request(
                    "GET", f"/trade-api/v2/markets?series_ticker={series_ticker}&status=open"
                )
                seen_tickers: set[str] = set()
                for m in resp.get("markets", []):
                    ticker = m.get("ticker", "")
                    if ticker in seen_tickers:
                        logger.warning(f"Duplicate ticker in API response, skipping: {ticker}")
                        continue
                    seen_tickers.add(ticker)
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
        # Also handles decimal strikes like KXBTCD-26MAY1922-T73749.99 → 73749.99
        # Note: .isdigit() returns False for decimals, so use a try/float() instead.
        ticker = market.get("ticker", "")
        for part in ticker.split("-"):
            if part.startswith("T"):
                try:
                    return float(part[1:])
                except ValueError:
                    continue
        # KXBTC15M up/down markets have no explicit strike field.
        # These markets resolve "yes" if BRTI at resolution > BRTI at market open,
        # where "market open" = close of the last completed 15-min BRTI candle.
        # Using the live 5-min close as threshold is wrong: if BTC moved $300 during
        # the current 15-min window, the wrong reference price produces the wrong
        # direction signal entirely.
        if market.get("market_type") == "15min":
            price = self._get_15min_reference_price()
            if price > 0.0:
                logger.debug(
                    f"No strike field for {ticker} (15-min up/down) — "
                    f"using last completed 15-min BRTI close {price:.2f} as threshold"
                )
                return price
        # Fallback for other timeframes or if no 15-min candles are available yet.
        price = self._get_composite_price()
        if price > 0.0:
            logger.debug(
                f"No strike field found for {ticker} — using composite price {price:.2f} as fallback"
            )
            return price
        return None

    def _parse_orderbook(self, orderbook: dict) -> tuple[int, int, int]:
        try:
            # Current Kalshi API returns orderbook_fp with yes_dollars/no_dollars.
            # Prices are dollar strings (e.g. "0.5500"), quantities are float strings.
            # Lists are in ASCENDING price order — best bid is the LAST entry.
            book_fp = orderbook.get("orderbook_fp")
            if book_fp:
                yes_bids = book_fp.get("yes_dollars", [])  # ascending dollar strings
                no_bids = book_fp.get("no_dollars", [])    # ascending dollar strings

                if not no_bids:
                    return (0, 0, 0)

                # Best bids = last entry (highest price) in each ascending list
                best_yes_bid = float(yes_bids[-1][0]) if yes_bids else 0.0
                best_no_bid = float(no_bids[-1][0])

                # Convert to integer cents
                best_bid_cents = round(best_yes_bid * 100)
                best_ask_cents = round((1.0 - best_no_bid) * 100)  # implied YES ask
                available_contracts = int(float(no_bids[-1][1]))

                if best_ask_cents <= 0 or best_ask_cents >= 100:
                    return (0, 0, 0)

                return (best_bid_cents, best_ask_cents, available_contracts)

            # Fallback: legacy format — "orderbook" key, integer cents, descending
            book = orderbook.get("orderbook", orderbook)
            yes_bids = book.get("yes", [])
            no_bids = book.get("no", [])

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

    def _get_15min_reference_price(self) -> float:
        """Return the close of the last COMPLETED 15-min BRTI candle.

        KXBTC15M markets resolve as 'yes' if BRTI at resolution >  BRTI at market
        open.  'Market open' for the current 15-min window equals the close of the
        previous completed 15-min candle.  Using the live 5-min close instead
        introduces a systematic error whenever price has drifted during the current
        15-min window: we would be asking Kronos the wrong directional question.
        """
        import pandas as pd
        try:
            df = self._store.get_ohlcv("15min")
            if df is not None and len(df) >= 1:
                now_utc = pd.Timestamp.now(tz="UTC")
                # Walk backwards to find the most recent completed candle.
                # The last row may be the in-progress (open) candle.
                for i in range(len(df) - 1, -1, -1):
                    candle_end = df.index[i] + pd.Timedelta(minutes=15)
                    if candle_end <= now_utc:
                        return float(df.iloc[i]["close"])
        except Exception as exc:
            logger.debug(f"Could not determine 15-min reference price: {exc}")
        # Fallback: if no completed 15-min candle exists yet (system just started),
        # use 5-min close as an approximation until the first candle closes.
        return self._get_composite_price()

    def _record_trade_sqlite(
        self,
        trade_id: str,
        signal,
        checklist_result,
        ticker: str,
        fill_price_cents: int,
    ) -> None:
        import math
        # Pull the six regime features off the signal. These are the exact values
        # that were fed to RegimeModel at inference time (or would have been, in
        # bootstrap mode) — so the persisted row is a faithful training example.
        # `features_stale=1` flags rows where the upstream Redis read failed; the
        # numeric columns are still populated (with the 0.0 fallback that fusion
        # used) so the row is reproducible, but training must filter on stale=0.
        feats = signal.regime_features or {}
        try:
            self._db.execute(
                """
                INSERT OR IGNORE INTO trades (
                    trade_id, timestamp, ticker, timeframe, direction, strike,
                    market_price, kelly_dollars, kelly_contracts,
                    kronos_raw, kronos_calibrated, regime_prob, deepseek_regime,
                    fill_price_cents,
                    funding_rate, funding_rate_trend, oi_delta_pct,
                    cvd_normalized, basis_spread_pct, brti_volatility_1h,
                    features_stale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    feats.get("funding_rate"),
                    feats.get("funding_rate_trend"),
                    feats.get("oi_delta_pct"),
                    feats.get("cvd_normalized"),
                    feats.get("basis_spread_pct"),
                    feats.get("brti_volatility_1h"),
                    1 if signal.features_stale else 0,
                ),
            )
            self._db.commit()
        except sqlite3.Error as exc:
            logger.error(f"SQLite insert failed for {trade_id}: {exc}")

    _MAX_POSITION_AGE_SECONDS: float = 86400.0  # 24 hours

    def _check_resolutions(self) -> None:
        for position in self._monitor.get_open_positions():
            try:
                # Age-out: if a position has been open for more than 24 hours without
                # resolving (e.g. API field names changed, market never finalized),
                # remove it from the monitor so it doesn't block new same-timeframe
                # trades or accumulate forever. We do NOT write an outcome to SQLite —
                # keeping outcome=NULL means these won't pollute calibrator training data.
                age_seconds = time.time() - position.timestamp
                if age_seconds > self._MAX_POSITION_AGE_SECONDS:
                    logger.warning(
                        f"Aging out stale position {position.ticker} "
                        f"trade_id={position.trade_id} "
                        f"(open for {age_seconds / 3600:.1f}h with no resolution)"
                    )
                    self._monitor.remove_position(position.trade_id)
                    continue

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
