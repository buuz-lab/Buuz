import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


def _make_db_with_open_trade(direction: int, fill_price_cents: int) -> sqlite3.Connection:
    """Create an in-memory DB with one unresolved trade keyed by ticker."""
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            direction INTEGER,
            fill_price_cents INTEGER,
            outcome INTEGER
        )
    """)
    db.execute(
        "INSERT INTO trades (ticker, direction, fill_price_cents, outcome) VALUES (?,?,?,?)",
        ["KXBTC-25Jun2026-99500", direction, fill_price_cents, None],
    )
    db.commit()
    return db


def _check_would_exit(db, ticker: str, mid_candle_mid: float) -> tuple[bool, float | None]:
    """Import and call the helper under test."""
    from main import KronosV2System
    system = KronosV2System.__new__(KronosV2System)
    system._db = db
    return system._check_would_exit(ticker, mid_candle_mid)


def test_would_exit_yes_trade_when_market_drops_16_cents():
    """YES trade at 35¢: mid drops to 18¢ (35-16=19 > 18) → would_exit=True."""
    db = _make_db_with_open_trade(direction=1, fill_price_cents=35)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.18)
    assert would_exit is True
    assert abs(price - 18.0) < 0.001


def test_would_not_exit_yes_trade_when_market_drops_14_cents():
    """YES trade at 35¢: mid at 22¢ (35-14=21 < 22) → would_exit=False."""
    db = _make_db_with_open_trade(direction=1, fill_price_cents=35)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.22)
    assert would_exit is False
    assert price is None


def test_would_exit_no_trade_when_market_rises_16_cents():
    """NO trade at 30¢ fill (YES entry = 70¢): mid rises to 87¢ (70+15=85 < 87) → would_exit=True."""
    db = _make_db_with_open_trade(direction=0, fill_price_cents=30)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.87)
    assert would_exit is True


def test_would_not_exit_no_trade_when_market_rises_14_cents():
    """NO trade at 30¢ fill (YES entry = 70¢): mid at 83¢ (70+15=85 > 83) → would_exit=False."""
    db = _make_db_with_open_trade(direction=0, fill_price_cents=30)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.83)
    assert would_exit is False


def test_no_open_trades_returns_false():
    """No unresolved trades → would_exit=False, price=None."""
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            direction INTEGER,
            fill_price_cents INTEGER,
            outcome INTEGER
        )
    """)
    db.commit()
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.18)
    assert would_exit is False
    assert price is None


def test_already_resolved_trade_not_checked():
    """Trade with outcome != NULL is not an open trade → would_exit=False."""
    db = _make_db_with_open_trade(direction=1, fill_price_cents=35)
    db.execute("UPDATE trades SET outcome=0")
    db.commit()
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.18)
    assert would_exit is False
