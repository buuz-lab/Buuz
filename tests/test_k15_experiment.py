"""Tests for KronosV2._run_experiment — $100 k15 paper experiment."""
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import main


_CREATE_EXPERIMENT_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS experiment_trades (
    candle_ts          TEXT    PRIMARY KEY,
    entered_at         REAL    NOT NULL,
    direction          INTEGER NOT NULL,
    k15_raw            REAL    NOT NULL,
    entry_price_cents  INTEGER NOT NULL,
    spread_cents       INTEGER NOT NULL,
    candle_progress    REAL    NOT NULL,
    outcome            INTEGER DEFAULT NULL,
    pnl                REAL    DEFAULT NULL,
    budget_after       REAL    DEFAULT NULL,
    resolved_at        REAL    DEFAULT NULL
)
"""


def _make_markets(candle_progress: float = 0.05) -> list[dict]:
    """Build a markets list with one 15-min market at the given candle progress."""
    close_ts = time.time() + (1.0 - candle_progress) * 900.0
    close_dt = datetime.fromtimestamp(close_ts, tz=timezone.utc)
    return [{
        "ticker": "KXBTC15M-TEST",
        "close_time": close_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }]


def _make_df15(candle_open_dt: datetime, close_above_open: bool = True) -> pd.DataFrame:
    """3-row 15-min OHLCV. index[-2] is the just-closed candle (candle_open_dt)."""
    idx = pd.DatetimeIndex([
        candle_open_dt - timedelta(minutes=15),
        candle_open_dt,
        candle_open_dt + timedelta(minutes=15),
    ], tz="UTC")
    close_price = 95100.0 if close_above_open else 94900.0
    return pd.DataFrame(
        {"open": [95000.0] * 3, "high": [96000.0] * 3,
         "low": [94000.0] * 3, "close": [close_price] * 3,
         "volume": [1.0] * 3},
        index=idx,
    )


def _make_system(k15: float = 0.70, bid: int = 48, ask: int = 50,
                 budget: float = 100.0, exp_open: dict | None = None) -> main.KronosV2:
    """Minimal KronosV2 with experiment_trades table and required attrs."""
    with patch.object(main.KronosV2, "__init__", lambda self: None):
        system = main.KronosV2()

    db = sqlite3.connect(":memory:")
    db.execute(_CREATE_EXPERIMENT_TRADES_TABLE)
    db.commit()

    system._db = db
    system._exp_active = True
    system._exp_open = exp_open
    system._exp_budget = budget
    system._cached_kronos = {"prob_15min": k15, "prob": 0.6}
    system._store = MagicMock()
    system._store.get_ohlcv.return_value = None   # no resolved candle by default
    system._router = MagicMock()
    system._router.get_orderbook.return_value = {"yes": [], "no": []}
    system._parse_orderbook = MagicMock(return_value=(bid, ask, 100))
    return system


# ── Entry tests ───────────────────────────────────────────────────────────────

def test_enters_on_strong_bullish_k15():
    """k15=0.70 > 0.55, progress=5%, spread=2¢ → YES entry logged."""
    system = _make_system(k15=0.70)
    system._run_experiment(_make_markets(candle_progress=0.05))
    row = system._db.execute("SELECT direction, k15_raw FROM experiment_trades").fetchone()
    assert row is not None
    assert row[0] == 1   # YES→UP
    assert abs(row[1] - 0.70) < 0.001


def test_enters_on_strong_bearish_k15():
    """k15=0.28 < 0.45, progress=5% → NO entry logged."""
    system = _make_system(k15=0.28)
    system._run_experiment(_make_markets(candle_progress=0.05))
    row = system._db.execute("SELECT direction FROM experiment_trades").fetchone()
    assert row is not None
    assert row[0] == 0   # NO→DOWN


def test_skips_weak_k15():
    """k15=0.52 within ±0.05 of 0.5 → no entry."""
    system = _make_system(k15=0.52)
    system._run_experiment(_make_markets(candle_progress=0.05))
    count = system._db.execute("SELECT COUNT(*) FROM experiment_trades").fetchone()[0]
    assert count == 0


def test_skips_wide_spread():
    """Spread = ask - bid = 50 - 44 = 6¢ > 3¢ → no entry."""
    system = _make_system(k15=0.70, bid=44, ask=50)   # spread = 6¢
    system._run_experiment(_make_markets(candle_progress=0.05))
    count = system._db.execute("SELECT COUNT(*) FROM experiment_trades").fetchone()[0]
    assert count == 0


# ── Resolution tests ──────────────────────────────────────────────────────────

def test_resolves_win_and_updates_budget():
    """Open YES position on candle that closed UP → pnl positive, budget grows."""
    candle_open_dt = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    candle_ts = candle_open_dt.replace(second=0, microsecond=0).isoformat()

    exp_open = {
        "candle_ts": candle_ts,
        "direction": 1,
        "k15": 0.70,
        "entry_price_cents": 45,
    }
    system = _make_system(exp_open=exp_open, budget=100.0)

    system._db.execute(
        "INSERT INTO experiment_trades "
        "(candle_ts, entered_at, direction, k15_raw, entry_price_cents, spread_cents, candle_progress)"
        " VALUES (?, ?, 1, 0.70, 45, 2, 0.05)",
        (candle_ts, time.time()),
    )
    system._db.commit()

    system._store.get_ohlcv.return_value = _make_df15(candle_open_dt, close_above_open=True)

    system._run_experiment(_make_markets(candle_progress=0.40))

    row = system._db.execute(
        "SELECT outcome, pnl, budget_after FROM experiment_trades WHERE candle_ts=?",
        (candle_ts,)
    ).fetchone()
    assert row[0] == 1
    assert row[1] > 0
    assert row[2] > 100.0
    assert system._exp_budget > 100.0
    assert system._exp_open is None


def test_resolves_loss_decrements_budget():
    """Open YES position on candle that closed DOWN → pnl negative, budget shrinks."""
    candle_open_dt = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    candle_ts = candle_open_dt.replace(second=0, microsecond=0).isoformat()

    exp_open = {
        "candle_ts": candle_ts,
        "direction": 1,
        "k15": 0.70,
        "entry_price_cents": 45,
    }
    system = _make_system(exp_open=exp_open, budget=100.0)

    system._db.execute(
        "INSERT INTO experiment_trades "
        "(candle_ts, entered_at, direction, k15_raw, entry_price_cents, spread_cents, candle_progress)"
        " VALUES (?, ?, 1, 0.70, 45, 2, 0.05)",
        (candle_ts, time.time()),
    )
    system._db.commit()

    system._store.get_ohlcv.return_value = _make_df15(candle_open_dt, close_above_open=False)

    system._run_experiment(_make_markets(candle_progress=0.40))

    row = system._db.execute(
        "SELECT outcome, pnl, budget_after FROM experiment_trades WHERE candle_ts=?",
        (candle_ts,)
    ).fetchone()
    assert row[0] == 0
    assert row[1] < 0
    assert row[2] < 100.0
    assert system._exp_budget < 100.0
    assert system._exp_open is None
