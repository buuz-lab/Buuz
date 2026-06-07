import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from loguru import logger

from btc_kalshi_system.data.brti_aggregator import BRTIAggregator
from btc_kalshi_system.data.derivatives_feed import DerivativesFeed
from btc_kalshi_system.data.deribit_options_feed import DeribitOptionsFeed
from btc_kalshi_system.data.exchange_feed import BitstampFeed, CoinbaseFeed, GeminiFeed, KrakenFeed
from btc_kalshi_system.data.feature_store import FeatureStore
from btc_kalshi_system.execution.kalshi_orderbook_feed import KalshiOrderbookFeed
from btc_kalshi_system.execution.kelly import KellySizer
from btc_kalshi_system.execution.position_monitor import PositionMonitor
from btc_kalshi_system.execution.pretrade_checklist import PreTradeChecklist
from btc_kalshi_system.execution.router import KalshiClientRouter
from btc_kalshi_system.models.calibrator import Calibrator
from btc_kalshi_system.models.deepseek_parser import DeepSeekContextParser
from btc_kalshi_system.models.kronos_engine import KronosEngine
from btc_kalshi_system.models.regime_model import RegimeModel, _FEATURE_ORDER
from btc_kalshi_system.portfolio.circuit_breaker import CircuitBreaker
from btc_kalshi_system.portfolio.monitor import OpenPosition, PortfolioMonitor
from btc_kalshi_system.signal.calibration_drift_monitor import CalibrationDriftMonitor
from btc_kalshi_system.signal.direction_win_rate_tracker import DirectionWinRateTracker
from btc_kalshi_system.signal.edge_tracker import EdgeTracker
from btc_kalshi_system.signal.stratified_edge_tracker import StratifiedEdgeTracker
from btc_kalshi_system.signal.fusion import SignalFusionEngine
import config

# ── Logging ───────────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stderr, level="INFO")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/kronos_{time}.log", rotation="1 day", retention="30 days", level="DEBUG", enqueue=True, catch=True)

# ── Timing ────────────────────────────────────────────────────────────────────

SIGNAL_INTERVAL_SECONDS = 300
DEEPSEEK_REFRESH_SECONDS = 900
RECOVERY_INTERVAL_SECONDS = 3600
MAX_POSITIONS_PER_TICKER_PER_SIDE = 2

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
    ("funding_rate",             "REAL DEFAULT NULL"),
    ("funding_rate_trend",       "REAL DEFAULT NULL"),
    ("oi_delta_pct",             "REAL DEFAULT NULL"),
    ("cvd_normalized",           "REAL DEFAULT NULL"),
    ("basis_spread_pct",         "REAL DEFAULT NULL"),
    ("brti_volatility_1h",       "REAL DEFAULT NULL"),
    ("features_stale",           "INTEGER DEFAULT 0"),
    # Phase 1: 14 new feature columns (NULL for rows written before this migration)
    ("cvd_velocity",             "REAL"),
    ("cvd_acceleration",         "REAL"),
    ("brti_momentum_5min",       "REAL"),
    ("brti_momentum_15min",      "REAL"),
    ("candle_progress",          "REAL"),
    ("hour_sin",                 "REAL"),
    ("hour_cos",                 "REAL"),
    ("kalshi_implied_prob",      "REAL"),
    ("funding_window_proximity", "REAL"),
    ("trend_slope_1h",           "REAL"),
    ("trend_r2_1h",              "REAL"),
    ("hourly_sr_proximity",      "REAL"),
    ("range_breakout_flag",      "REAL"),
    ("tape_speed_tpm",           "REAL"),
    ("exit_reason",              "TEXT"),
    ("large_print_direction",    "REAL"),
    # Session 6: Deribit options features (22-27)
    ("atm_iv",                   "REAL DEFAULT NULL"),
    ("iv_rv_spread",             "REAL DEFAULT NULL"),
    ("pcr_oi",                   "REAL DEFAULT NULL"),
    ("term_structure_slope",     "REAL DEFAULT NULL"),
    ("skew_25d",                 "REAL DEFAULT NULL"),
    ("kalshi_spread_normalized", "REAL DEFAULT NULL"),
    ("deribit_stale",            "INTEGER DEFAULT 1"),
    ("okx_stale",                "INTEGER DEFAULT 0"),
    # Feature 28: 24h BTC price return (session 11). NULL for rows written before this migration.
    ("btc_24h_return",           "REAL DEFAULT NULL"),
    # Shadow: 15min-candle Kronos MC prob logged alongside live 5min prob for A/B comparison.
    ("kronos_raw_15min",         "REAL DEFAULT NULL"),
    # k15 calibrated probability — calibrator.transform(kronos_raw_15min) at trade time.
    # Enables k15-primary vs k5-primary counterfactual on actual trades.
    ("k15_calibrated_prob",      "REAL DEFAULT NULL"),
    # 1 if cached k15 was computed from a candle AFTER this market opened, 0 if pre-open.
    # More precise than candle_progress for separating "k15 had current-market data" from "stale".
    ("k15_post_open",            "INTEGER DEFAULT NULL"),
]

_TRADE_SNAPSHOTS_COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("atm_iv",                   "REAL"),
    ("iv_rv_spread",             "REAL"),
    ("pcr_oi",                   "REAL"),
    ("term_structure_slope",     "REAL"),
    ("skew_25d",                 "REAL"),
    ("kalshi_spread_normalized", "REAL"),
]

_CREATE_TRADE_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS trade_snapshots (
    trade_id                TEXT,
    snapshot_window         TEXT,
    snapshot_ts             TEXT,
    funding_rate            REAL,
    funding_rate_trend      REAL,
    oi_delta_pct            REAL,
    cvd_normalized          REAL,
    basis_spread_pct        REAL,
    brti_volatility_1h      REAL,
    cvd_velocity            REAL,
    cvd_acceleration        REAL,
    brti_momentum_5min      REAL,
    brti_momentum_15min     REAL,
    candle_progress         REAL,
    hour_sin                REAL,
    hour_cos                REAL,
    kalshi_implied_prob     REAL,
    funding_window_proximity REAL,
    trend_slope_1h          REAL,
    trend_r2_1h             REAL,
    hourly_sr_proximity     REAL,
    range_breakout_flag     REAL,
    tape_speed_tpm          REAL,
    atm_iv                  REAL,
    iv_rv_spread            REAL,
    pcr_oi                  REAL,
    term_structure_slope    REAL,
    skew_25d                REAL,
    kalshi_spread_normalized REAL,
    kronos_prob             REAL,
    regime_direction        INTEGER,
    exit_triggered          INTEGER,
    PRIMARY KEY (trade_id, snapshot_window)
)
"""

_CREATE_GATE_REJECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS gate_rejections (
    rejection_id     TEXT PRIMARY KEY,
    timestamp        REAL,
    ticker           TEXT,
    timeframe        TEXT,
    direction        INTEGER,
    failed_gate      INTEGER,
    failed_reason    TEXT,
    signal_prob      REAL,
    deepseek_regime  TEXT,
    kalshi_mid_cents INTEGER,
    features         TEXT,
    outcome          INTEGER DEFAULT NULL,
    resolved_at      REAL DEFAULT NULL
)
"""

_GATE_REJECTIONS_COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("aged_out", "INTEGER DEFAULT 0"),
    # shadow=1: Gate 7 would have blocked but didn't. Trade IS in trades.db.
    # Any training query against gate_rejections MUST filter WHERE shadow = 0.
    ("shadow", "INTEGER DEFAULT 0"),
    # Gate 8 (Kalshi consensus): fresh second-fetch mid-price at block time for threshold analysis.
    ("kalshi_mid_at_block", "REAL DEFAULT NULL"),
    # Retained for schema compatibility — populated by removed Gate 9 (edge-flip shadow).
    ("flip_price_cents", "INTEGER DEFAULT NULL"),
    # Shadow: 15min-candle Kronos MC prob at signal time for A/B comparison vs 5min.
    ("kronos_raw_15min", "REAL DEFAULT NULL"),
    # Raw k5 MC probability at signal time — apples-to-apples comparison with kronos_raw_15min.
    ("kronos_raw",       "REAL DEFAULT NULL"),
    # All gates: the price at which the trade would have (or did) fill.
    # shadow=0: hypothetical fill (ask for YES, 100-bid for NO) at signal time.
    # shadow=1 (Gate 7): real fill price from the trade that proceeded.
    ("would_be_fill_cents", "INTEGER DEFAULT NULL"),
    # k15 calibrated probability — calibrator.transform(kronos_raw_15min) at signal time.
    # Cannot be reconstructed post-hoc (calibrator state changes every 25 resolutions).
    # Combined with kronos_raw_15min, would_be_fill_cents, and outcome, enables full
    # k15-primary vs k5-primary counterfactual analysis.
    ("k15_calibrated_prob", "REAL DEFAULT NULL"),
    # Denormalized from features JSON for easy timing-bucket queries (t=0 / t+5 / t+10).
    ("candle_progress",     "REAL DEFAULT NULL"),
    # 1 if cached k15 was computed from a candle AFTER this market opened, 0 if pre-open.
    # More precise than candle_progress for separating "k15 had current-market data" from "stale".
    ("k15_post_open",       "INTEGER DEFAULT NULL"),
    # Regime v2 prob_up at signal time — primary signal once regime model deployed.
    # Enables future analysis: did high-confidence regime blocks (near 0 or 1) save money?
    ("regime_prob",         "REAL DEFAULT NULL"),
    # Computed signal edge (win_prob - market_price) at block time.
    # Enables Gate 14 threshold calibration once Phase 3c calibrator is deployed.
    ("signal_edge",         "REAL DEFAULT NULL"),
    # Phase 3c calibrator context features — logged at block time so the calibrator
    # can train on the full feature set including gate_rejections counterfactuals.
    ("brti_volatility_1h",      "REAL DEFAULT NULL"),
    ("kalshi_spread_normalized", "REAL DEFAULT NULL"),
]


_CREATE_CANDLE_FEATURES_TABLE = (
    """CREATE TABLE IF NOT EXISTS candle_features (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candle_ts     TEXT NOT NULL UNIQUE,
    btc_direction INTEGER,
    logged_at     TEXT NOT NULL,
    features_stale  INTEGER DEFAULT 0,
    deribit_stale   INTEGER DEFAULT 0,
    """
    + ",\n    ".join(f"{col}    REAL" for col in _FEATURE_ORDER)
    + "\n)"
)

_CANDLE_FEATURES_COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("kronos_raw_15min",   "REAL DEFAULT NULL"),
    ("kronos_raw_5min",    "REAL DEFAULT NULL"),
    # Kalshi opening-price snapshot — first WS orderbook snapshot for each contract.
    # Enables regime v2 edge comparison: does our model beat the market's opening estimate?
    ("kalshi_open_mid",          "REAL DEFAULT NULL"),  # mid-price at candle open (0-1)
    ("kalshi_open_spread",       "REAL DEFAULT NULL"),  # bid-ask spread at candle open (0-1)
    ("kalshi_open_depth",        "REAL DEFAULT NULL"),  # total contracts at best bid+ask at open
    ("kalshi_mid_candle_mid",    "REAL DEFAULT NULL"),  # mid-price at ~50% candle progress (0-1)
    ("kalshi_mid_candle_spread", "REAL DEFAULT NULL"),  # spread at ~50% candle progress (0-1)
    ("kalshi_mid_candle_progress", "REAL DEFAULT NULL"),  # actual progress when snapshot taken
    ("kalshi_early_mid",         "REAL DEFAULT NULL"),  # mid-price at ~30s (3-7% progress) — T+30s edge test
    ("kalshi_early_progress",    "REAL DEFAULT NULL"),  # actual progress when early snapshot taken
    ("volume_ratio_1h",          "REAL DEFAULT NULL"),  # hourly volume vs 30-day avg (regime feature)
    ("regime_prob",              "REAL DEFAULT NULL"),  # regime v2 prob_up at candle close (NULL until model deployed)
    ("kalshi_open_imbalance", "REAL DEFAULT NULL"),
    ("btc_spx_corr_8d",      "REAL DEFAULT NULL"),
    ("btc_qqq_corr_8d",      "REAL DEFAULT NULL"),
    ("would_exit",             "INTEGER DEFAULT 0"),
    ("would_exit_price_cents", "REAL DEFAULT NULL"),
    # T+30s data (session 38) — enables future model that predicts from in-candle state
    # rather than prior-candle-close state, eliminating the structural T+30s Kalshi disadvantage.
    ("kalshi_early_drift",  "REAL DEFAULT NULL"),  # kalshi_early_mid - kalshi_open_mid: 30s market reaction
    ("features_early",      "TEXT DEFAULT NULL"),  # JSON of all regime features at T+30s snap time
    ("liq_net_norm",         "REAL DEFAULT NULL"),
    ("eth_direction_15min",  "REAL DEFAULT NULL"),
    ("okx_spot_imbalance",   "REAL DEFAULT NULL"),
    ("pcr_delta",            "REAL DEFAULT NULL"),
    ("skew_delta",           "REAL DEFAULT NULL"),
    ("deepseek_dir_prob",    "REAL DEFAULT NULL"),
    ("cvd_price_divergence", "REAL DEFAULT NULL"),
    ("recent_up_fraction",   "REAL DEFAULT NULL"),
]


class KronosV2:
    def __init__(self) -> None:
        self._store = FeatureStore()
        self._kronos = KronosEngine()
        self._calibrator = Calibrator()
        try:
            self._calibrator = Calibrator.load(config.CALIBRATOR_MODEL_PATH)
            logger.info(f"Calibrator loaded from {config.CALIBRATOR_MODEL_PATH} (n_samples={self._calibrator.n_samples})")
        except FileNotFoundError:
            logger.info("Calibrator file not found — starting fresh (passthrough mode)")
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
        self._edge_tracker = EdgeTracker()
        self._drift_monitor = CalibrationDriftMonitor()
        self._dir_tracker = DirectionWinRateTracker()
        self._fusion = SignalFusionEngine(
            feature_store=self._store,
            kronos_engine=self._kronos,
            calibrator=self._calibrator,
            regime_model=self._regime,
            deepseek_parser=self._deepseek,
            drift_monitor=self._drift_monitor,
        )
        self._stratified_edge = StratifiedEdgeTracker()
        self._kelly = KellySizer()
        self._checklist = PreTradeChecklist(self._kelly)
        self._orderbook_feed = KalshiOrderbookFeed(
            api_key_id=config.KALSHI_API_KEY_ID,
            private_key_path=config.KALSHI_PRIVATE_KEY_PATH,
        )
        self._router = KalshiClientRouter(orderbook_feed=self._orderbook_feed)
        self._monitor = PortfolioMonitor()
        self._breaker = CircuitBreaker(
            monitor=self._monitor,
            edge_tracker=self._edge_tracker,
            router=self._router,
            calibrator=self._calibrator,
        )

        self._position_monitor = PositionMonitor(
            portfolio_monitor=self._monitor,
            regime_model=self._regime,
            kronos_engine=self._kronos,
            feature_store=self._store,
            router=self._router,
            fusion_engine=self._fusion,
            db_path="trades.db",
        )

        import redis as _redis_module
        self._redis = _redis_module.from_url(config.REDIS_URL)

        # Preload Kronos here — single-threaded, no event loop, no concurrent I/O.
        # Loading it later (inside asyncio.to_thread while WebSocket feeds run) causes
        # a segfault on Apple Silicon because PyTorch's Accelerate-framework init races
        # with macOS kqueue.  Once loaded, _load() becomes a no-op so the worker thread
        # call is safe.
        self._kronos.preload()
        # Second independent engine for k15 MC — deepcopy after preload so weights
        # are fully initialised. Two separate instances allow asyncio.gather to run
        # k5 and k15 concurrently without sharing PyTorch model state.
        import copy as _copy
        self._kronos_k15 = _copy.deepcopy(self._kronos)

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
        self._db.execute(_CREATE_TRADE_SNAPSHOTS_TABLE)
        for col_name, col_def in _TRADE_SNAPSHOTS_COLUMN_MIGRATIONS:
            try:
                self._db.execute(
                    f"ALTER TABLE trade_snapshots ADD COLUMN {col_name} {col_def}"
                )
            except sqlite3.OperationalError:
                pass
        self._db.execute(_CREATE_GATE_REJECTIONS_TABLE)
        for col_name, col_def in _GATE_REJECTIONS_COLUMN_MIGRATIONS:
            try:
                self._db.execute(f"ALTER TABLE gate_rejections ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass
        self._db.execute(_CREATE_CANDLE_FEATURES_TABLE)
        for col_name, col_def in _CANDLE_FEATURES_COLUMN_MIGRATIONS:
            try:
                self._db.execute(f"ALTER TABLE candle_features ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass
        self._db.commit()

        self._running = False
        self._last_deepseek_refresh = 0.0
        self._last_recovery_attempt = 0.0
        self._cached_kronos: dict | None = None
        self._cache_updated_event = asyncio.Event()
        # Maps candle-open ISO timestamp → Kalshi ticker (populated in _process_market).
        # Used by _candle_logger_loop to look up the WS opening snapshot for each candle.
        self._candle_ticker_map: dict[str, str] = {}
        # Mid-candle Kalshi snapshots keyed by candle_ts ISO string (captured once per candle
        # when progress crosses 40-60%). Written by _candle_logger_loop, read at close time.
        self._mid_candle_snaps: dict[str, dict] = {}
        self._early_candle_snaps: dict[str, dict] = {}

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        if config.PAPER_TRADING:
            logger.info("Running in PAPER TRADING mode — no real orders will be placed")
        logger.info("KronosV2 starting up")
        self._check_auth()

        queues = [asyncio.Queue() for _ in range(4)]
        agg = BRTIAggregator()
        feeds = [CoinbaseFeed(), KrakenFeed(), BitstampFeed(), GeminiFeed()]
        deriv = DerivativesFeed()
        deribit_feed = DeribitOptionsFeed()

        await asyncio.gather(
            feeds[0].run(queues[0]),
            feeds[1].run(queues[1]),
            feeds[2].run(queues[2]),
            feeds[3].run(queues[3]),
            agg.run(queues),
            self._store.run(agg.out_queue),
            deriv.run(),
            deribit_feed.run(),
            self._orderbook_feed.run(),
            self._main_loop(),
            self._regime_watchdog(),
            self._position_monitor.run(),
            self._kronos_background_loop(),
            self._candle_logger_loop(),
        )

    async def _main_loop(self) -> None:
        while self._running:
            await self._cache_updated_event.wait()
            self._cache_updated_event.clear()
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
        logger.info("KronosV2 main loop stopped")

    async def _kronos_background_loop(self) -> None:
        last_candle_ts = None
        while self._running:
            try:
                df = self._store.get_ohlcv("5min")
                if df is not None and len(df) >= 10:
                    current_ts = df.index[-1]
                    if current_ts != last_candle_ts:
                        last_candle_ts = current_ts
                        strike = await asyncio.to_thread(self._get_15min_reference_price)
                        try:
                            # k5 and k15 run in parallel — each uses its own KronosEngine
                            # instance (self._kronos / self._kronos_k15) so their PyTorch
                            # model states never share memory during concurrent forward().
                            k5_result, k15_result = await asyncio.gather(
                                asyncio.to_thread(
                                    self._kronos.run_monte_carlo, self._store, 100, strike
                                ),
                                asyncio.to_thread(
                                    self._kronos_k15.run_monte_carlo, self._store, 100, strike,
                                    candle_freq="15min",
                                ),
                                return_exceptions=True,
                            )
                            if isinstance(k5_result, Exception):
                                raise k5_result
                            prob = k5_result
                            if isinstance(k15_result, Exception):
                                logger.warning(f"KronosBG: 15min MC failed — {k15_result}")
                                prob_15min: float | None = None
                            else:
                                prob_15min = k15_result
                            # Always assign a new dict — never mutate in place (GIL safety)
                            self._cached_kronos = {
                                "prob": prob,
                                "prob_15min": prob_15min,
                                "candle_ts": current_ts,
                                "computed_at": time.time(),
                                "strike": strike,
                            }
                            self._cache_updated_event.set()
                            _k15_str = f"{prob_15min:.4f}" if prob_15min is not None else "N/A"
                            logger.info(
                                f"KronosBG: prob={prob:.4f} prob_15min={_k15_str} "
                                f"strike={strike:.2f} candle={current_ts}"
                            )
                        except ValueError as exc:
                            logger.warning(f"KronosBG: insufficient OHLCV data — {exc}")
                        except Exception as exc:
                            logger.error(f"KronosBG: MC failed — {exc}")
            except Exception as exc:
                logger.error(f"KronosBG: loop error — {exc}")
            await asyncio.sleep(10)

    async def _regime_watchdog(self) -> None:
        """Warn when regime:features TTL is dangerously low or key has expired."""
        while True:
            await asyncio.sleep(60)
            try:
                ttl = self._store._redis.ttl("regime:features")
                if ttl == -2:
                    logger.error("WATCHDOG: regime:features is EXPIRED — trades firing now will be stale")
                    subprocess.run(
                        ["osascript", "-e",
                         'display notification "regime:features EXPIRED — stale trades possible"'
                         ' with title "KronosV2 ALERT" sound name "Sosumi"'],
                        check=False,
                    )
                elif 0 <= ttl <= 90:
                    logger.warning(
                        f"WATCHDOG: regime:features TTL={ttl}s — dangerously low, refresh may have stalled"
                    )
                    subprocess.run(
                        ["osascript", "-e",
                         f'display notification "regime:features TTL={ttl}s — refresh stalled"'
                         f' with title "KronosV2 Warning" sound name "Basso"'],
                        check=False,
                    )
            except Exception as exc:
                logger.error(f"WATCHDOG: TTL check failed — {exc}")
            try:
                if (
                    self._cached_kronos is not None
                    and time.time() - self._cached_kronos["computed_at"] > 480
                ):
                    logger.error(
                        "WATCHDOG: Kronos cache stale — background loop may be stuck"
                    )
                    subprocess.run(
                        ["osascript", "-e",
                         'display notification "Kronos cache stale — background loop may be stuck"'
                         ' with title "KronosV2 ALERT" sound name "Sosumi"'],
                        check=False,
                    )
            except Exception as exc:
                logger.error(f"WATCHDOG: Kronos cache check failed — {exc}")

    _WOULD_EXIT_THRESHOLD = 0.15  # tune after 50+ resolved would-exit rows under regime v2

    def _check_would_exit(self, ticker: str, mid_candle_mid: float) -> tuple[bool, float | None]:
        """Paper exit check: would we have exited an open trade at this mid-candle price?

        Returns (would_exit, price_cents). price_cents is the mid-candle YES price in cents,
        or None when no exit would trigger.

        Finds open trades for this ticker (outcome IS NULL). Each Kalshi 15-min
        contract has a unique ticker, so this correctly scopes to the current candle.
        """
        rows = self._db.execute(
            "SELECT direction, fill_price_cents FROM trades "
            "WHERE ticker = ? AND outcome IS NULL",
            [ticker],
        ).fetchall()
        for direction, fill_price_cents in rows:
            if direction == 1:  # YES bet
                yes_entry = fill_price_cents / 100.0
                if mid_candle_mid < yes_entry - self._WOULD_EXIT_THRESHOLD:
                    return True, mid_candle_mid * 100.0
            else:  # NO bet; yes_entry = implied YES price at fill time
                yes_entry = (100 - fill_price_cents) / 100.0
                if mid_candle_mid > yes_entry + self._WOULD_EXIT_THRESHOLD:
                    return True, mid_candle_mid * 100.0
        return False, None

    async def _candle_logger_loop(self) -> None:
        """Logs regime features + BTC direction at every 15-min candle close."""
        last_logged_ts = None
        while self._running:
            try:
                await asyncio.sleep(10)
                df15 = self._store.get_ohlcv("15min")
                if df15 is None or len(df15) < 3:
                    continue

                # Candle progress snapshots: early (~30s, 3-7%) and mid-candle (~50%, 40-60%).
                try:
                    _in_progress_ts = df15.index[-1]
                    _in_progress_key = _in_progress_ts.to_pydatetime().astimezone(timezone.utc).replace(
                        second=0, microsecond=0
                    ).isoformat()
                    _now_utc = datetime.now(timezone.utc)
                    _elapsed = (_now_utc - _in_progress_ts.to_pydatetime().astimezone(timezone.utc)).total_seconds()
                    _progress = _elapsed / 900.0
                    _snap_ticker = self._candle_ticker_map.get(_in_progress_key)
                    _snap_ob = self._orderbook_feed.get_orderbook(_snap_ticker) if _snap_ticker else None

                    # Early snapshot at T+30s (3-7% progress) — Kalshi price + regime features
                    # for future T+30s model training.
                    if _in_progress_key not in self._early_candle_snaps and 0.03 <= _progress <= 0.07:
                        if _snap_ob:
                            _bid, _ask, _ = self._parse_orderbook(_snap_ob)
                            if _ask > 0:
                                _early_features, _, _, _ = self._fusion.get_features_snapshot()
                                self._early_candle_snaps[_in_progress_key] = {
                                    "mid_prob":      (_bid + _ask) / 200.0,
                                    "progress":      round(_progress, 3),
                                    "features_early": json.dumps(_early_features) if _early_features else None,
                                }

                    # Mid-candle snapshot at ~50% progress (~7.5 min in)
                    if _in_progress_key not in self._mid_candle_snaps and 0.40 <= _progress <= 0.60:
                        if _snap_ob:
                            _bid, _ask, _ = self._parse_orderbook(_snap_ob)
                            if _ask > 0:
                                self._mid_candle_snaps[_in_progress_key] = {
                                    "mid_prob": (_bid + _ask) / 200.0,
                                    "spread":   (_ask - _bid) / 100.0,
                                    "progress": round(_progress, 3),
                                }
                                # Paper exit check: would we have exited at this price?
                                _would_exit, _would_exit_price = self._check_would_exit(
                                    _snap_ticker, (_bid + _ask) / 200.0
                                )
                                self._mid_candle_snaps[_in_progress_key]["would_exit"] = int(_would_exit)
                                self._mid_candle_snaps[_in_progress_key]["would_exit_price_cents"] = _would_exit_price
                                logger.debug(
                                    f"CandleLogger: mid-candle snapshot {_snap_ticker} "
                                    f"mid={(_bid+_ask)/200:.3f} progress={_progress:.2f}"
                                )
                except Exception as exc:
                    logger.debug(f"CandleLogger: candle snapshot failed — {exc}")

                # Second-to-last row is the most recently closed candle;
                # the last row is the in-progress candle — skip it.
                closed_ts = df15.index[-2]
                if closed_ts == last_logged_ts:
                    continue
                closed_candle = df15.iloc[-2]
                btc_direction = 1 if closed_candle["close"] > closed_candle["open"] else 0
                features, features_stale, deribit_stale, regime_prob = self._fusion.get_features_snapshot()

                # Look up Kalshi opening snapshot for the closed candle.
                _candle_key = closed_ts.to_pydatetime().astimezone(timezone.utc).replace(
                    second=0, microsecond=0
                ).isoformat()
                _ot = self._candle_ticker_map.get(_candle_key)
                _open_snap = self._orderbook_feed.get_open_snapshot(_ot) if _ot else None
                kalshi_open_mid    = _open_snap["mid_prob"] if _open_snap else None
                kalshi_open_spread = _open_snap["spread"]   if _open_snap else None
                kalshi_open_depth  = (
                    (_open_snap["depth_bid"] + _open_snap["depth_ask"]) if _open_snap else None
                )

                # Look up mid-candle snapshot (captured below while candle was in progress).
                _mid_snap = self._mid_candle_snaps.pop(_candle_key, None)
                kalshi_mid_candle_mid      = _mid_snap["mid_prob"]  if _mid_snap else None
                kalshi_mid_candle_spread   = _mid_snap["spread"]    if _mid_snap else None
                kalshi_mid_candle_progress = _mid_snap["progress"]  if _mid_snap else None
                would_exit             = _mid_snap.get("would_exit", 0)           if _mid_snap else 0
                would_exit_price_cents = _mid_snap.get("would_exit_price_cents")  if _mid_snap else None

                _early_snap = self._early_candle_snaps.pop(_candle_key, None)
                kalshi_early_mid      = _early_snap["mid_prob"]  if _early_snap else None
                kalshi_early_progress = _early_snap["progress"]  if _early_snap else None
                features_early        = _early_snap.get("features_early") if _early_snap else None
                kalshi_early_drift    = (
                    round(kalshi_early_mid - kalshi_open_mid, 4)
                    if kalshi_early_mid is not None and kalshi_open_mid is not None
                    else None
                )

                cols = list(_FEATURE_ORDER)
                vals = [features.get(c) for c in cols]
                placeholders = ", ".join(["?"] * (16 + len(cols)))
                col_names = (
                    "candle_ts, btc_direction, logged_at, features_stale, deribit_stale, "
                    "kalshi_open_mid, kalshi_open_spread, kalshi_open_depth, "
                    "kalshi_mid_candle_mid, kalshi_mid_candle_spread, kalshi_mid_candle_progress, "
                    "kalshi_early_mid, kalshi_early_progress, regime_prob, "
                    "would_exit, would_exit_price_cents, "
                    "kalshi_early_drift, features_early, "
                    + ", ".join(cols)
                )
                placeholders = ", ".join(["?"] * (18 + len(cols)))
                self._db.execute(
                    f"INSERT OR IGNORE INTO candle_features ({col_names}) VALUES ({placeholders})",
                    [
                        closed_ts.isoformat(),
                        btc_direction,
                        datetime.utcnow().isoformat(),
                        int(features_stale),
                        int(deribit_stale),
                        kalshi_open_mid,
                        kalshi_open_spread,
                        kalshi_open_depth,
                        kalshi_mid_candle_mid,
                        kalshi_mid_candle_spread,
                        kalshi_mid_candle_progress,
                        kalshi_early_mid,
                        kalshi_early_progress,
                        regime_prob,
                        would_exit,
                        would_exit_price_cents,
                        kalshi_early_drift,
                        features_early,
                        *vals,
                    ],
                )
                self._db.commit()
                # Inject T+30s drift into fusion so next candle's regime features include it.
                self._fusion.set_kalshi_early_drift(kalshi_early_drift)
                last_logged_ts = closed_ts
                logger.info(f"CandleLogger: logged candle {closed_ts} direction={btc_direction}")
            except Exception as exc:
                logger.warning(f"CandleLogger: {exc}")

    def stop(self) -> None:
        logger.info("KronosV2 stopping")
        self._running = False

    # ── Main cycle ────────────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        # Attempt router recovery before the circuit-breaker check so BOTH_FAILED
        # can self-heal without a manual restart. _maybe_attempt_recovery is
        # rate-limited internally (5-min interval) so this is cheap on every cycle.
        self._router._maybe_attempt_recovery()

        # 1. Circuit breaker
        status = self._breaker.check()
        if status.tripped:
            logger.warning(f"Circuit breaker active [{status.reason.value}]: {status.message} — skipping cycle")
            return

        # 2. Refresh DeepSeek / regime context on its own cadence (every 15 min),
        #    then always load the latest from Redis for this cycle.
        now = time.time()
        ctx = self._get_market_context()
        composite_price = self._get_composite_price()
        ctx["composite_price"] = composite_price
        if now - self._last_deepseek_refresh >= DEEPSEEK_REFRESH_SECONDS:
            self._last_deepseek_refresh = now
            logger.debug("DeepSeek context refreshed")
        # Always push the latest context once per cycle
        self._fusion.update_market_context(ctx)

        # 5. Active markets
        markets = self._get_active_markets()
        if not markets:
            logger.info("No active BTC markets found")
            return

        # Keep WS orderbook subscriptions in sync with active tickers
        active_tickers = {m["ticker"] for m in markets}
        self._orderbook_feed.update_subscriptions(active_tickers)

        # Capture opening Kalshi price for any ticker we haven't seen before.
        # Done here (via get_orderbook) rather than in _handle so we always have
        # a live, non-empty book rather than the potentially-empty initial snapshot.
        for m in markets:
            _t = m.get("ticker", "")
            if _t and self._orderbook_feed.get_open_snapshot(_t) is None:
                try:
                    _ob = self._router.get_orderbook(_t)
                    if _ob:
                        _bid, _ask, _ = self._parse_orderbook(_ob)
                        if _ask > 0:
                            self._orderbook_feed.set_open_snapshot(_t, _bid, _ask)
                except Exception:
                    pass

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

        # 8. Resolve gate rejections (counterfactual outcomes)
        try:
            self._resolve_gate_rejections()
        except Exception as exc:
            logger.error(f"Error resolving gate rejections: {exc}")

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

        # c. Orderbook (fetched before signal so kalshi_implied_prob is available)
        try:
            orderbook_resp = self._router.get_orderbook(ticker)
        except Exception as exc:
            logger.warning(f"Failed to fetch orderbook for {ticker}: {exc}")
            return

        # d. Parse orderbook
        best_bid_cents, best_ask_cents, available_contracts = self._parse_orderbook(orderbook_resp)
        if best_ask_cents == 0:
            logger.debug(f"Empty orderbook for {ticker}, skipping")
            return

        # e. Inject current Kalshi mid-price and bid-ask spread so fusion._regime_features()
        #    can compute kalshi_implied_prob and kalshi_spread_normalized.
        mid_cents = (best_bid_cents + best_ask_cents) / 2.0
        kalshi_spread_normalized = (best_ask_cents - best_bid_cents) / 100.0
        self._fusion.update_kalshi_spread(kalshi_spread_normalized)
        self._fusion.update_kalshi_mid(mid_cents)

        # Inject Kalshi open imbalance so regime model can use it as a feature.
        _imbal_snap = self._orderbook_feed.get_open_snapshot(ticker)
        self._fusion.set_kalshi_imbalance(
            _imbal_snap.get("depth_imbalance") if _imbal_snap else None
        )

        # f. Read background MC cache — avoids 23s Kronos call on the critical path
        cached = self._cached_kronos
        if cached is None:
            logger.info(f"Kronos cache not yet populated — skipping cycle for {ticker}")
            return
        _cache_age = time.time() - cached["computed_at"]
        if _cache_age > 600:
            logger.error(
                f"Kronos cache too stale ({_cache_age:.0f}s) — skipping cycle for {ticker}"
            )
            return

        # Determine if k15 cache was computed from a candle AFTER this market opened.
        # More precise than candle_progress: 1 = k15 has current-market data, 0 = pre-open.
        try:
            _close_str = market.get("close_time", "")
            _close_dt = datetime.fromisoformat(_close_str.replace("Z", "+00:00"))
            _market_open_unix = _close_dt.timestamp() - 900.0
            _k15_post_open: int | None = (
                1 if cached["candle_ts"].timestamp() >= _market_open_unix else 0
            )
            # Record candle-open → ticker mapping so _candle_logger_loop can look up the
            # WS opening snapshot for this contract when it logs the closed candle.
            _open_dt = _close_dt.replace(second=0, microsecond=0) - timedelta(seconds=900)
            self._candle_ticker_map[_open_dt.isoformat()] = ticker
        except Exception:
            _k15_post_open = None
        logger.debug(
            f"KronosBG strike delta: ${abs(cached['strike'] - strike):.2f} "
            f"(cached={cached['strike']:.2f} market={strike:.2f})"
        )
        signal = self._fusion.get_signal(
            timeframe=timeframe, strike=strike,
            kronos_raw=cached["prob"],
            kronos_raw_15min=cached.get("prob_15min"),
        )
        if signal is None:
            logger.debug(f"No signal for {ticker} (gated out)")
            return

        # Write derived context features to Redis so the next DeepSeek refresh
        # can include momentum/trend/range data (one-cycle lag is intentional).
        _DERIVED_CONTEXT_FIELDS = (
            "brti_momentum_5min", "brti_momentum_15min",
            "trend_slope_1h", "trend_r2_1h", "range_breakout_flag",
            "cvd_velocity", "cvd_acceleration", "tape_speed_tpm",
        )
        try:
            derived = {k: signal.regime_features.get(k)
                       for k in _DERIVED_CONTEXT_FIELDS
                       if signal.regime_features.get(k) is not None}
            if derived:
                self._redis.set(
                    "regime:derived_context",
                    json.dumps(derived),
                    ex=120,
                )
        except Exception:
            pass

        # g. Pre-trade checklist
        fill_price_cents = best_ask_cents if signal.direction == 1 else (100 - best_bid_cents)

        side_count = self._monitor.ticker_direction_count(ticker, signal.direction)
        if side_count >= MAX_POSITIONS_PER_TICKER_PER_SIDE:
            logger.info(
                f"Skipping {ticker}: already {side_count} open {'YES' if signal.direction == 1 else 'NO'} positions on this ticker"
            )
            return

        current_exposure = self._monitor.get_current_exposure()
        same_timeframe_open = self._monitor.has_timeframe_position(timeframe)
        _raw_edge_ok = self._edge_tracker.is_above_threshold()
        if config.PAPER_TRADING and not _raw_edge_ok:
            logger.warning(
                f"Gate 4 SHADOW (paper): rolling edge below threshold "
                f"(edge={self._edge_tracker.current_edge():.4f}, "
                f"n={len(self._edge_tracker)}/50) — would block in live mode"
            )
        edge_above_threshold = True if config.PAPER_TRADING else _raw_edge_ok

        fresh_kalshi_mid1 = (best_bid_cents + best_ask_cents) / 200.0
        dir_win_rate = self._dir_tracker.get_win_rate(signal.direction)
        is_bootstrap = self._regime._clf is None
        result = self._checklist.run(
            signal=signal,
            best_ask_cents=best_ask_cents,
            best_bid_cents=best_bid_cents,
            available_contracts=available_contracts,
            current_exposure=current_exposure,
            same_timeframe_open=same_timeframe_open,
            composite_price=composite_price,
            edge_above_threshold=edge_above_threshold,
            fresh_kalshi_mid=fresh_kalshi_mid1,
            is_drifting=self._drift_monitor.is_drifting(),
            direction_win_rate=dir_win_rate,
            is_bootstrap=is_bootstrap,
        )

        # g. Checklist failed
        if not result.passed:
            logger.info(f"Pre-trade checklist failed for {ticker} [gate {result.failed_gate}]: {result.failed_reason}")
            try:
                would_be_fill = best_ask_cents if signal.direction == 1 else (100 - best_bid_cents)
                _k15_raw = cached.get("prob_15min")
                _k15_cal = self._calibrator.transform(_k15_raw, regime=signal.deepseek_regime) if _k15_raw is not None else None
                _candle_prog = (signal.regime_features or {}).get("candle_progress")
                _signal_edge = None
                try:
                    _wp = signal.calibrated_prob if signal.direction == 1 else (1.0 - signal.calibrated_prob)
                    _mp = would_be_fill / 100.0
                    _signal_edge = round(_wp - _mp, 6)
                except Exception:
                    pass
                _rf = signal.regime_features or {}
                self._db.execute(
                    """INSERT OR IGNORE INTO gate_rejections
                       (rejection_id, timestamp, ticker, timeframe, direction,
                        failed_gate, failed_reason, signal_prob, deepseek_regime,
                        kalshi_mid_cents, features, kalshi_mid_at_block, would_be_fill_cents,
                        kronos_raw_15min, kronos_raw, k15_calibrated_prob, candle_progress,
                        k15_post_open, regime_prob, signal_edge,
                        brti_volatility_1h, kalshi_spread_normalized)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        time.time(),
                        ticker,
                        timeframe,
                        signal.direction,
                        result.failed_gate,
                        result.failed_reason,
                        signal.kronos_calibrated,
                        signal.deepseek_regime,
                        round(mid_cents),
                        json.dumps(_rf),
                        result.kalshi_mid_at_block,
                        would_be_fill,
                        _k15_raw,
                        cached.get("prob"),
                        _k15_cal,
                        _candle_prog,
                        _k15_post_open,
                        signal.regime_prob if hasattr(signal, 'regime_prob') else None,
                        _signal_edge,
                        _rf.get("brti_volatility_1h"),
                        _rf.get("kalshi_spread_normalized"),
                    ),
                )
                self._db.commit()
            except sqlite3.Error as exc:
                logger.error(f"SQLite gate_rejections insert failed for {ticker}: {exc}")

            return

        # Gate 7 shadow — log when CVD opposes direction but let trade proceed.
        # Enforcing mode removed; shadow rows accumulate so win-rate can be tracked.
        # Insert is deferred to after fill so would_be_fill_cents stores the real fill price.
        _cvd = (signal.regime_features or {}).get("cvd_normalized", 0.0)
        _g7_reason: str | None = None
        if signal.direction == 1 and _cvd < -config.CVD_GATE_THRESHOLD:
            _g7_reason = f"CVD {_cvd:.3f} opposes YES→UP (threshold -{config.CVD_GATE_THRESHOLD})"
        elif signal.direction == 0 and _cvd > config.CVD_GATE_THRESHOLD:
            _g7_reason = f"CVD {_cvd:.3f} opposes NO→DOWN (threshold +{config.CVD_GATE_THRESHOLD})"

        # h. Second orderbook fetch — get fresh prices right before placing the order.
        #    The first fetch (T+0) was for kalshi_mid/spread feature injection only.
        #    Both paper and live mode abort if the second fetch fails.
        try:
            orderbook_resp2 = self._router.get_orderbook(ticker)
        except Exception as exc:
            logger.warning(
                f"Second orderbook fetch failed for {ticker} — aborting order: {exc}"
            )
            return
        fresh_bid, fresh_ask, fresh_contracts = self._parse_orderbook(orderbook_resp2)
        if fresh_ask == 0:
            logger.warning(f"Second orderbook empty for {ticker} — aborting order")
            return

        fresh_kalshi_mid2 = (fresh_bid + fresh_ask) / 200.0
        result2 = self._checklist.run(
            signal=signal,
            best_ask_cents=fresh_ask,
            best_bid_cents=fresh_bid,
            available_contracts=fresh_contracts,
            current_exposure=current_exposure,
            same_timeframe_open=same_timeframe_open,
            composite_price=composite_price,
            edge_above_threshold=edge_above_threshold,
            fresh_kalshi_mid=fresh_kalshi_mid2,
            is_drifting=self._drift_monitor.is_drifting(),
            direction_win_rate=dir_win_rate,
            is_bootstrap=is_bootstrap,
        )
        if not result2.passed:
            logger.info(
                f"Second-fetch checklist failed for {ticker} [gate {result2.failed_gate}]: "
                f"{result2.failed_reason}"
            )
            return

        fill_price_cents = fresh_ask if signal.direction == 1 else (100 - fresh_bid)

        # i. Place order (or simulate in paper mode)
        trade_id = str(uuid.uuid4())
        side = "yes" if signal.direction == 1 else "no"
        if config.PAPER_TRADING:
            logger.info(
                f"[PAPER] Simulated fill: {ticker} {side} {result2.kelly_contracts}@{fill_price_cents}¢ "
                f"(${result2.kelly_dollars:.2f} kelly) trade_id={trade_id}"
            )
        else:
            try:
                self._router.place_order(
                    ticker=ticker,
                    side=side,
                    count=result2.kelly_contracts,
                    price_cents=fill_price_cents,
                    client_order_id=trade_id,
                )
                logger.info(
                    f"Order placed: {ticker} {side} {result2.kelly_contracts}@{fill_price_cents}¢ "
                    f"(${result2.kelly_dollars:.2f} kelly) trade_id={trade_id}"
                )
            except Exception as exc:
                logger.error(f"Order placement failed for {ticker}: {exc}")
                return

        # j. Record position
        position = OpenPosition(
            trade_id=trade_id,
            ticker=ticker,
            timeframe=timeframe,
            direction=signal.direction,
            strike=strike,
            contracts=result2.kelly_contracts,
            entry_price_cents=fill_price_cents,
            kelly_dollars=result2.kelly_dollars,
            timestamp=time.time(),
            calibrated_prob=signal.calibrated_prob,
            deepseek_regime=signal.deepseek_regime,
        )
        self._monitor.add_position(position)

        # k. Log to SQLite
        _trade_k15_raw = cached.get("prob_15min")
        _trade_k15_cal = self._calibrator.transform(_trade_k15_raw, regime=signal.deepseek_regime) if _trade_k15_raw is not None else None
        self._record_trade_sqlite(trade_id, signal, result2, ticker, fill_price_cents,
                                   kronos_raw_15min=_trade_k15_raw,
                                   k15_calibrated_prob=_trade_k15_cal,
                                   k15_post_open=_k15_post_open)

        # Gate 7 shadow insert — deferred to here so would_be_fill_cents = real fill price.
        if _g7_reason:
            try:
                _g7_k15_raw = cached.get("prob_15min")
                _g7_k15_cal = self._calibrator.transform(_g7_k15_raw, regime=signal.deepseek_regime) if _g7_k15_raw is not None else None
                _g7_candle_prog = (signal.regime_features or {}).get("candle_progress")
                self._db.execute(
                    # shadow=1: trade proceeds; row is observability-only, not training data.
                    """INSERT OR IGNORE INTO gate_rejections
                       (rejection_id, timestamp, ticker, timeframe, direction,
                        failed_gate, failed_reason, signal_prob, deepseek_regime,
                        kalshi_mid_cents, features, shadow, would_be_fill_cents,
                        kronos_raw_15min, kronos_raw, k15_calibrated_prob, candle_progress,
                        k15_post_open)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), time.time(), ticker, timeframe,
                        signal.direction, 7, _g7_reason, signal.kronos_calibrated,
                        signal.deepseek_regime, round(mid_cents),
                        json.dumps(signal.regime_features or {}),
                        fill_price_cents,
                        _g7_k15_raw,
                        cached.get("prob"),
                        _g7_k15_cal,
                        _g7_candle_prog,
                        _k15_post_open,
                    ),
                )
                self._db.commit()
                logger.info(f"Gate 7 shadow (trade proceeds): {ticker} — {_g7_reason}")
            except sqlite3.Error as exc:
                logger.error(f"SQLite gate_rejections shadow insert failed: {exc}")

    # ── Helper methods ────────────────────────────────────────────────────────

    def _check_auth(self) -> None:
        """Verify API key works for authenticated endpoints at startup.

        In paper trading mode the system only calls public endpoints, so a
        broken key is otherwise invisible until WS or live trading fails.
        Logs a loud warning rather than blocking startup.
        """
        try:
            self._router._raw._request("GET", "/trade-api/v2/portfolio/balance")
            logger.info("Auth check passed — API key is valid for authenticated endpoints")
        except Exception as exc:
            logger.error(
                f"AUTH CHECK FAILED: {exc}\n"
                "The API key cannot authenticate against Kalshi. "
                "WebSocket and live order placement will not work. "
                "Regenerate the key at https://kalshi.com and update .env + keys/kalshi_private.key"
            )

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
            import time as _time
            import redis as _redis
            r = _redis.from_url(config.REDIS_URL, decode_responses=True)
            raw_features = r.get("regime:features")
            if raw_features:
                ctx = json.loads(raw_features)
            else:
                # Primary key expired (exchange outage) — try last-known-good fallback.
                # If LKG is fresh (< 15 min) the features are still representative;
                # derivatives update every 5 min so a 15-min-old snapshot is good
                # enough for training. Only mark stale when LKG is ≥ 15 min old.
                _LKG_FRESH_THRESHOLD_S = 900  # 15 minutes
                lkg_raw = r.get("regime:features:lkg")
                if lkg_raw:
                    lkg = json.loads(lkg_raw)
                    age_s = _time.time() - lkg.pop("_lkg_written_at", _time.time())
                    is_stale_lkg = age_s >= _LKG_FRESH_THRESHOLD_S
                    logger.warning(
                        f"regime:features expired — falling back to LKG features "
                        f"({age_s / 60:.0f}min old); "
                        f"row will {'be marked stale' if is_stale_lkg else 'qualify (LKG fresh)'}"
                    )
                    if is_stale_lkg:
                        lkg["_lkg"] = True
                    ctx = lkg
                else:
                    ctx = {}

            # Merge derived context (momentum/trend/range written by _process_market).
            # Existing keys from regime:features take precedence; derived only fills gaps.
            try:
                derived_raw = r.get("regime:derived_context")
                if derived_raw:
                    derived = json.loads(derived_raw)
                    for k, v in derived.items():
                        if k not in ctx:
                            ctx[k] = v
            except Exception:
                pass

            # Merge options features from DeribitOptionsFeed
            try:
                opts_raw = r.get("options:features")
                if opts_raw:
                    opts = json.loads(opts_raw)
                    ctx.update({k: v for k, v in opts.items() if k not in ctx})
                else:
                    opts_lkg_raw = r.get("options:features:lkg")
                    if opts_lkg_raw:
                        opts_lkg = json.loads(opts_lkg_raw)
                        age_s = _time.time() - opts_lkg.pop("_lkg_written_at", _time.time())
                        logger.warning(
                            f"options:features expired — using LKG "
                            f"({age_s / 3600:.1f}h old); row will be deribit_stale"
                        )
                        opts_lkg["_deribit_lkg"] = True
                        ctx.update({k: v for k, v in opts_lkg.items() if k not in ctx})
            except Exception:
                pass

            # Derive iv_rv_spread from merged context (requires both sources present).
            # atm_iv is Deribit annualised implied vol in % (e.g. 31.1).
            # brti_volatility_1h is a dimensionless tick CV (e.g. 0.0009).
            # To compare them we convert brti_volatility_1h to annualised % by
            # treating the 1h tick CV as an hourly vol and scaling to a year:
            #   annualised_rv_pct = brti_vol_cv * sqrt(8760 hours/year) * 100
            # This gives ~8–12% annualised RV, making the spread meaningful
            # (e.g. 31% IV - 10% RV = 21% premium — options are expensive).
            try:
                import math as _math
                atm_iv = ctx.get("atm_iv")
                rv = ctx.get("brti_volatility_1h")
                if atm_iv is not None and rv is not None and rv > 0:
                    annualised_rv_pct = float(rv) * _math.sqrt(8760) * 100
                    ctx["iv_rv_spread"] = float(atm_iv) - annualised_rv_pct
            except Exception:
                pass

            # Build nested fear_greed dict from flat keys written by derivatives_feed.
            if ctx.get("fear_greed_value") is not None:
                ctx["fear_greed"] = {
                    "value": ctx["fear_greed_value"],
                    "label": ctx.get("fear_greed_label", ""),
                }

            # Recent Kalshi outcomes (last 5 resolved trades, chronological order).
            try:
                rows = self._db.execute(
                    """SELECT direction, outcome FROM trades
                       WHERE outcome IS NOT NULL
                       ORDER BY rowid DESC LIMIT 5"""
                ).fetchall()
                ctx["recent_outcomes"] = [
                    1 if row[1] == 1 else 0 for row in rows
                ][::-1]
            except Exception:
                ctx["recent_outcomes"] = []

            return ctx
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
        ticker = market.get("ticker", "")

        # Primary: Kalshi sets floor_strike on KXBTC15M markets to the BRTI average
        # at market open — this is the canonical resolution reference price.
        floor = market.get("floor_strike")
        if floor is not None:
            try:
                v = float(floor)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass

        # Secondary: other explicit strike fields (e.g. KXBTCD strike markets).
        for field in ("cap_strike", "strike_price", "result_at_open"):
            val = market.get(field)
            if val is not None:
                try:
                    v = float(val)
                    if v > 0:
                        return v
                except (TypeError, ValueError):
                    continue

        # Parse from ticker: KXBTC-25JUN-T95000 → 95000.0
        for part in ticker.split("-"):
            if part.startswith("T"):
                try:
                    return float(part[1:])
                except ValueError:
                    continue

        # Fallback for 15-min markets: use last completed BRTI candle close.
        # floor_strike is set by Kalshi at market open; this path is only reached
        # if the market opened before BRTI data was available.
        if market.get("market_type") == "15min":
            price = self._get_15min_reference_price()
            if price > 0.0:
                logger.warning(
                    f"floor_strike missing/zero for {ticker} — "
                    f"falling back to last completed 15-min BRTI close {price:.2f}"
                )
                return price

        price = self._get_composite_price()
        if price > 0.0:
            logger.warning(f"No strike found for {ticker} — using composite price {price:.2f}")
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
        kronos_raw_15min: float | None = None,
        k15_calibrated_prob: float | None = None,
        k15_post_open: int | None = None,
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
                    features_stale,
                    cvd_velocity, cvd_acceleration,
                    brti_momentum_5min, brti_momentum_15min,
                    candle_progress, hour_sin, hour_cos,
                    kalshi_implied_prob, funding_window_proximity,
                    trend_slope_1h, trend_r2_1h,
                    hourly_sr_proximity, range_breakout_flag, tape_speed_tpm,
                    large_print_direction,
                    atm_iv, iv_rv_spread, pcr_oi, term_structure_slope,
                    skew_25d, kalshi_spread_normalized, deribit_stale, okx_stale,
                    btc_24h_return, kronos_raw_15min, k15_calibrated_prob, k15_post_open
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
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
                    feats.get("cvd_velocity"),
                    feats.get("cvd_acceleration"),
                    feats.get("brti_momentum_5min"),
                    feats.get("brti_momentum_15min"),
                    feats.get("candle_progress"),
                    feats.get("hour_sin"),
                    feats.get("hour_cos"),
                    feats.get("kalshi_implied_prob"),
                    feats.get("funding_window_proximity"),
                    feats.get("trend_slope_1h"),
                    feats.get("trend_r2_1h"),
                    feats.get("hourly_sr_proximity"),
                    feats.get("range_breakout_flag"),
                    feats.get("tape_speed_tpm"),
                    feats.get("large_print_direction"),
                    feats.get("atm_iv"),
                    feats.get("iv_rv_spread"),
                    feats.get("pcr_oi"),
                    feats.get("term_structure_slope"),
                    feats.get("skew_25d"),
                    feats.get("kalshi_spread_normalized"),
                    1 if signal.deribit_stale else 0,
                    1 if signal.okx_stale else 0,
                    feats.get("btc_24h_return"),
                    kronos_raw_15min,
                    k15_calibrated_prob,
                    k15_post_open,
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

                LOSS_STREAK_TTL = 86400  # 24 hours
                if outcome == 1:  # win
                    current = int(self._redis.get("trading:loss_streak") or 0)
                    if current > 0:
                        self._redis.set("trading:loss_streak", current - 1, ex=LOSS_STREAK_TTL)
                else:             # loss
                    self._redis.incr("trading:loss_streak")
                    self._redis.expire("trading:loss_streak", LOSS_STREAK_TTL)

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
                y_up_outcome = int(position.direction == outcome)
                self._drift_monitor.record(position.calibrated_prob, y_up_outcome)
                self._dir_tracker.record(position.direction, outcome)
                self._stratified_edge.record(
                    position.deepseek_regime,
                    position.calibrated_prob,
                    outcome,
                    position.entry_price_cents / 100,
                )
                if self._drift_monitor.is_drifting():
                    _baseline = self._drift_monitor.baseline_brier()
                    _baseline_str = f"{_baseline:.4f}" if _baseline is not None else "unknown"
                    logger.warning(
                        f"CalibrationDriftMonitor: drift detected — "
                        f"current_brier={self._drift_monitor.current_brier():.4f} "
                        f"baseline={_baseline_str}"
                    )

                try:
                    self._db.execute(
                        "UPDATE trades SET outcome=?, pnl_dollars=? WHERE trade_id=?",
                        (outcome, trade.pnl_dollars, position.trade_id),
                    )
                    self._db.commit()
                except sqlite3.Error as exc:
                    logger.error(f"SQLite update failed for {position.trade_id}: {exc}")

                # Refit calibrator every 25 resolutions (not every trade)
                try:
                    pending = int(self._redis.incr("calibration_drift:pending_refits") or 0)
                except Exception:
                    pending = 25
                if pending >= 25:
                    self._refit_calibrator()

                logger.info(
                    f"Trade resolved: {position.ticker} trade_id={position.trade_id} "
                    f"outcome={'WIN' if outcome == 1 else 'LOSS'} pnl=${trade.pnl_dollars:.2f}"
                )
            except Exception as exc:
                logger.warning(f"Failed to check resolution for {position.ticker}: {exc}")

    def _refit_calibrator(self) -> None:
        """Phase 3c: refit calibrator on regime_prob + signal_edge (most recent 300 rows).

        Requires 200+ rows where regime_prob IS NOT NULL (i.e., regime v2 has been deployed
        and has been generating predictions). If fewer than 200 regime_prob rows exist, the
        calibrator stays as-is — no fallback to k15, since Phase 3c is the target behaviour.

        signal_edge = abs(regime_prob - kalshi_mid_cents/100) at trade time, already stored
        in both tables. Captures how much edge exists between the model and the market price.
        """
        try:
            self._redis.set("calibration_drift:pending_refits", 0)
            rows = self._db.execute(
                """SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome FROM (
                       SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome, timestamp
                       FROM trades
                       WHERE outcome IS NOT NULL AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL
                       UNION ALL
                       SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome, timestamp
                       FROM gate_rejections
                       WHERE outcome IS NOT NULL AND aged_out=0 AND shadow = 0
                         AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL
                   ) ORDER BY timestamp DESC LIMIT 300"""
            ).fetchall()
            if len(rows) < 500:
                logger.info(
                    f"Calibrator refit skipped: only {len(rows)} regime_prob rows "
                    f"(need 500 — accumulating after regime v2 deploy)"
                )
                return
            regime_probs  = np.array([r[0] for r in rows], dtype=float)
            abs_edges     = np.abs(np.array([r[1] for r in rows], dtype=float))
            refit_regimes = np.array([r[2] for r in rows], dtype=object)
            directions    = np.array([r[3] for r in rows], dtype=float)
            outcomes_arr  = np.array([r[4] for r in rows], dtype=float)
            y_yes = np.where(directions == 1, outcomes_arr, 1.0 - outcomes_arr)
            self._calibrator.fit(regime_probs, y_yes, regimes=refit_regimes, edges=abs_edges)
            os.makedirs("models", exist_ok=True)
            self._calibrator.save(config.CALIBRATOR_MODEL_PATH)
            self._drift_monitor.reset_baseline()
            logger.info(
                f"Calibrator Phase 3c refit: n_samples={self._calibrator.n_samples} "
                f"passthrough={self._calibrator._passthrough} edge_aware={self._calibrator._edge_aware}"
            )
        except Exception as exc:
            logger.warning(f"Calibrator refit failed: {exc}")

    def _resolve_gate_rejections(self) -> None:
        cutoff = time.time() - 900  # only resolve rows ≥15 min old
        age_out_cutoff = time.time() - self._MAX_POSITION_AGE_SECONDS

        pending = self._db.execute(
            """SELECT rejection_id, ticker, direction, timestamp, failed_gate
               FROM gate_rejections
               WHERE outcome IS NULL AND aged_out = 0 AND timestamp < ?
               LIMIT 50""",
            (cutoff,),
        ).fetchall()

        for rejection_id, ticker, direction, ts, failed_gate in pending:
            try:
                if ts < age_out_cutoff:
                    self._db.execute(
                        "UPDATE gate_rejections SET aged_out=1 WHERE rejection_id=?",
                        (rejection_id,),
                    )
                    self._db.commit()
                    logger.info(f"Gate rejection aged out: {ticker} rejection_id={rejection_id}")
                    continue

                resp = self._router._raw._request("GET", f"/trade-api/v2/markets/{ticker}")
                market = resp.get("market", resp)
                status = market.get("status", "")
                if status != "finalized":
                    continue

                result = market.get("result", "")
                if not result:
                    continue

                if result == "yes":
                    outcome = 1 if direction == 1 else 0
                else:
                    outcome = 1 if direction == 0 else 0

                resolved_at = time.time()
                self._db.execute(
                    "UPDATE gate_rejections SET outcome=?, resolved_at=? WHERE rejection_id=?",
                    (outcome, resolved_at, rejection_id),
                )
                self._db.commit()
                logger.info(
                    f"Gate rejection resolved: {ticker} gate={failed_gate} "
                    f"would_have={'WON' if outcome == 1 else 'LOST'}"
                )
                try:
                    pending = int(self._redis.incr("calibration_drift:pending_refits") or 0)
                except Exception:
                    pending = 0
                if pending >= 25:
                    self._refit_calibrator()
            except Exception as exc:
                logger.warning(f"Failed to resolve gate rejection {rejection_id}: {exc}")


# Alias used by tests (KronosV2System is the canonical test-facing name).
KronosV2System = KronosV2

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
