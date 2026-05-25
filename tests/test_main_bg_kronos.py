"""
TDD tests for:
1. KronosV2._kronos_background_loop (new background coroutine)
2. KronosV2._process_market: cache-read guard + second orderbook fetch
"""

import asyncio
import contextlib
import math
import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from loguru import logger as _loguru

import config
import main


@contextlib.contextmanager
def _capture_logs(level: str = "INFO"):
    """Capture loguru messages into a list for assertion."""
    records = []
    sid = _loguru.add(lambda msg: records.append(str(msg)), level=level, format="{level}:{message}")
    try:
        yield records
    finally:
        _loguru.remove(sid)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_df(last_ts: pd.Timestamp, n: int = 12) -> pd.DataFrame:
    prices = [95000.0 + i * 10.0 for i in range(n)]
    idx = pd.date_range(end=last_ts, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": [1.0] * n, "amount": [1.0] * n},
        index=idx,
    )


def _make_system() -> "main.KronosV2":
    with patch.object(main.KronosV2, "__init__", lambda self: None):
        system = main.KronosV2()

    system._running = True
    system._cached_kronos = None
    system._store = MagicMock()
    system._kronos = MagicMock()
    system._fusion = MagicMock()
    system._checklist = MagicMock()
    system._router = MagicMock()
    system._monitor = MagicMock()
    system._redis = MagicMock()
    system._last_deepseek_refresh = 0.0

    db = sqlite3.connect(":memory:")
    db.execute(main._CREATE_TRADES_TABLE)
    for col, defn in main._TRADES_COLUMN_MIGRATIONS:
        try:
            db.execute(f"ALTER TABLE trades ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass
    db.execute(main._CREATE_GATE_REJECTIONS_TABLE)
    for col, defn in main._GATE_REJECTIONS_COLUMN_MIGRATIONS:
        try:
            db.execute(f"ALTER TABLE gate_rejections ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass
    db.commit()
    system._db = db
    return system


def _make_signal(direction: int = 1, prob: float = 0.65) -> MagicMock:
    sig = MagicMock()
    sig.direction = direction
    sig.calibrated_prob = prob
    sig.kronos_raw = prob
    sig.kronos_calibrated = prob
    sig.regime_prob = math.nan
    sig.regime_direction = -1
    sig.deepseek_regime = "trending_up"
    sig.timeframe = "15min"
    sig.strike = 95000.0
    sig.timestamp = datetime.now(timezone.utc)
    sig.regime_features = {}
    sig.features_stale = False
    sig.deribit_stale = False
    sig.okx_stale = False
    return sig


def _make_checklist_result(
    passed: bool = True, gate: int | None = None, reason: str = ""
) -> MagicMock:
    r = MagicMock()
    r.passed = passed
    r.failed_gate = gate
    r.failed_reason = reason
    r.kelly_dollars = 10.0
    r.kelly_contracts = 2
    return r


# Legacy format: yes bids ascending price, no bids ascending price
def _make_ob(yes_price: int = 55, no_price: int = 45, qty: int = 10) -> dict:
    """Build an orderbook in the legacy integer-cents format.

    best_bid_cents = yes_price, best_ask_cents = 100 - no_price
    """
    return {"orderbook": {"yes": [[yes_price, qty]], "no": [[no_price, qty]]}}


def _make_market(ticker: str = "KXBTC15M-25JUN-T95000") -> dict:
    return {
        "ticker": ticker,
        "market_type": "15min",
        "floor_strike": 95000.0,
        "close_time": "2099-01-01T00:00:00Z",
    }


def _setup_passing(system: "main.KronosV2", direction: int = 1) -> None:
    """Configure all dependencies so _process_market would succeed end-to-end."""
    system._cached_kronos = {
        "prob": 0.65,
        "candle_ts": pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
        "computed_at": time.time(),
        "strike": 95000.0,
    }
    ob = _make_ob()
    system._router.get_orderbook.return_value = ob
    system._fusion.get_signal.return_value = _make_signal(direction=direction)
    system._checklist.run.return_value = _make_checklist_result(passed=True)
    system._monitor.ticker_direction_count.return_value = 0
    system._monitor.get_current_exposure.return_value = 0.0
    system._monitor.has_timeframe_position.return_value = False


# ── Tests: _kronos_background_loop ───────────────────────────────────────────


class TestKronosBackgroundLoop:

    def test_populates_cache_on_new_candle(self):
        """Loop stores {prob, candle_ts, computed_at, strike} in _cached_kronos
        when a new candle is detected."""
        system = _make_system()
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        system._store.get_ohlcv.return_value = _make_df(ts, 12)
        system._kronos.run_monte_carlo.return_value = 0.65
        system._get_15min_reference_price = MagicMock(return_value=95000.0)

        async def mock_sleep(n):
            system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._kronos_background_loop())

        assert system._cached_kronos is not None
        assert system._cached_kronos["prob"] == 0.65
        assert system._cached_kronos["candle_ts"] == ts
        assert system._cached_kronos["strike"] == 95000.0
        assert "computed_at" in system._cached_kronos

    def test_does_not_rerun_mc_on_same_candle(self):
        """MC is not re-run when candle timestamp hasn't changed between iterations."""
        system = _make_system()
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        system._store.get_ohlcv.return_value = _make_df(ts, 12)
        system._kronos.run_monte_carlo.return_value = 0.65
        system._get_15min_reference_price = MagicMock(return_value=95000.0)

        sleep_calls = [0]

        async def mock_sleep(n):
            sleep_calls[0] += 1
            if sleep_calls[0] >= 2:
                system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._kronos_background_loop())

        # Two iterations, same candle → MC called exactly once
        assert system._kronos.run_monte_carlo.call_count == 1

    def test_continues_after_mc_exception(self):
        """Loop stays alive after an MC exception and processes the next new candle."""
        system = _make_system()
        ts1 = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        ts2 = pd.Timestamp("2024-01-01 12:05:00", tz="UTC")

        df1 = _make_df(ts1, 12)
        df2 = _make_df(ts2, 12)

        store_calls = [0]

        def get_ohlcv_side(tf):
            if tf == "5min":
                store_calls[0] += 1
                return df1 if store_calls[0] <= 1 else df2
            return None

        system._store.get_ohlcv.side_effect = get_ohlcv_side

        mc_calls = [0]

        def mc_side(*args, **kwargs):
            mc_calls[0] += 1
            if mc_calls[0] == 1:
                raise RuntimeError("MC exploded")
            return 0.70

        system._kronos.run_monte_carlo.side_effect = mc_side
        system._get_15min_reference_price = MagicMock(return_value=95000.0)

        sleep_calls = [0]

        async def mock_sleep(n):
            sleep_calls[0] += 1
            if sleep_calls[0] >= 2:
                system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._kronos_background_loop())

        # Loop survived the exception and populated the cache on the second candle
        assert system._cached_kronos is not None
        assert system._cached_kronos["prob"] == 0.70


# ── Tests: _process_market cache-read guard ──────────────────────────────────


class TestProcessMarketCacheGuard:

    def test_skips_gracefully_when_cache_none(self):
        """Returns with INFO log and no signal call when _cached_kronos is None."""
        system = _make_system()
        system._cached_kronos = None
        system._router.get_orderbook.return_value = _make_ob()

        with _capture_logs("INFO") as records:
            system._process_market(_make_market(), 95000.0)

        system._fusion.get_signal.assert_not_called()
        assert any("not yet populated" in r for r in records)

    def test_skips_when_cache_stale(self):
        """Returns with ERROR log and no signal call when cache is > 600s old."""
        system = _make_system()
        system._router.get_orderbook.return_value = _make_ob()
        system._cached_kronos = {
            "prob": 0.65,
            "candle_ts": pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            "computed_at": time.time() - 700,
            "strike": 95000.0,
        }

        with _capture_logs("ERROR") as records:
            system._process_market(_make_market(), 95000.0)

        system._fusion.get_signal.assert_not_called()
        assert any("too stale" in r for r in records)

    def test_get_signal_receives_kronos_raw_from_cache(self):
        """get_signal() is invoked with kronos_raw equal to the cached prob."""
        system = _make_system()
        _setup_passing(system)

        system._process_market(_make_market(), 95000.0)

        call_kwargs = system._fusion.get_signal.call_args
        provided_raw = (
            call_kwargs.kwargs.get("kronos_raw")
            if call_kwargs.kwargs
            else (call_kwargs.args[2] if len(call_kwargs.args) >= 3 else None)
        )
        assert provided_raw == 0.65


# ── Tests: _process_market second orderbook fetch ────────────────────────────


class TestProcessMarketSecondFetch:

    def test_aborts_when_second_fetch_fails(self):
        """Second fetch exception → WARNING log, no position opened."""
        system = _make_system()
        _setup_passing(system)

        # Second get_orderbook raises
        ob = _make_ob()
        system._router.get_orderbook.side_effect = [ob, Exception("network timeout")]

        with _capture_logs("WARNING") as records:
            system._process_market(_make_market(), 95000.0)

        system._monitor.add_position.assert_not_called()
        assert any(
            "second" in r.lower() or "fetch" in r.lower()
            for r in records
        )

    def test_aborts_when_second_checklist_fails(self):
        """Second checklist failure → INFO log, no position opened."""
        system = _make_system()
        _setup_passing(system)

        # First checklist passes; second fails
        pass_r = _make_checklist_result(passed=True)
        fail_r = _make_checklist_result(passed=False, gate=5, reason="spread too wide")
        system._checklist.run.side_effect = [pass_r, fail_r]

        system._process_market(_make_market(), 95000.0)

        system._monitor.add_position.assert_not_called()

    def test_fill_price_from_second_fetch(self):
        """fill_price_cents passed to _record_trade_sqlite uses fresh second-fetch ask."""
        system = _make_system()
        _setup_passing(system, direction=1)

        # First fetch: ask = 100 - 45 = 55; second fetch: ask = 100 - 42 = 58
        first_ob = _make_ob(yes_price=55, no_price=45)
        second_ob = _make_ob(yes_price=58, no_price=42)
        system._router.get_orderbook.side_effect = [first_ob, second_ob]

        captured_fills = []

        def capture_record(trade_id, signal, checklist_result, ticker, fill_price_cents):
            captured_fills.append(fill_price_cents)

        with patch.object(system, "_record_trade_sqlite", side_effect=capture_record):
            system._process_market(_make_market(), 95000.0)

        assert len(captured_fills) == 1, "Expected exactly one trade recorded"
        assert captured_fills[0] == 58, (
            f"Expected fill_price=58 (fresh ask from second fetch), got {captured_fills[0]}"
        )
