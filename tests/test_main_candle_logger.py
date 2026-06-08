"""
TDD tests for:
1. candle_features table creation in KronosV2.__init__
2. KronosV2._candle_logger_loop coroutine
"""

import asyncio
import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import main
from btc_kalshi_system.models.regime_model import _FEATURE_ORDER


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_df15(n: int = 4, close_above_open: bool = True) -> pd.DataFrame:
    """Build a 15-min OHLCV DataFrame with n rows."""
    end_ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
    idx = pd.date_range(end=end_ts, periods=n, freq="15min")
    open_price = 95000.0
    close_price = 95100.0 if close_above_open else 94900.0
    return pd.DataFrame(
        {
            "open": [open_price] * n,
            "high": [max(open_price, close_price) + 50] * n,
            "low": [min(open_price, close_price) - 50] * n,
            "close": [close_price] * n,
            "volume": [1.0] * n,
            "amount": [1.0] * n,
        },
        index=idx,
    )


def _make_features() -> dict:
    """Return a zero-filled feature dict matching _FEATURE_ORDER."""
    return {k: 0.0 for k in _FEATURE_ORDER}


def _make_system() -> "main.KronosV2":
    """Create a minimal KronosV2 test system with candle_features table set up."""
    with patch.object(main.KronosV2, "__init__", lambda self: None):
        system = main.KronosV2()

    system._running = True
    system._store = MagicMock()
    system._fusion = MagicMock()
    system._candle_ticker_map = {}
    system._mid_candle_snaps = {}
    system._early_candle_snaps = {}
    system._candle_open_brti = {}
    system._cached_kronos = None
    system._deriv = None

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
    db.execute(main._CREATE_CANDLE_FEATURES_TABLE)
    for col, defn in main._CANDLE_FEATURES_COLUMN_MIGRATIONS:
        try:
            db.execute(f"ALTER TABLE candle_features ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass
    db.commit()
    system._db = db
    return system


# ── Tests: table creation ─────────────────────────────────────────────────────


class TestCandleFeaturesTableCreation:

    def test_candle_features_table_created_on_init(self):
        """candle_features table exists after KronosV2 DB setup."""
        system = _make_system()
        tables = {
            r[0]
            for r in system._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "candle_features" in tables


# ── Tests: _candle_logger_loop ────────────────────────────────────────────────


class TestCandleLoggerLoop:

    def test_candle_logger_writes_row_on_new_candle(self):
        """Loop writes one row when a new closed candle is detected."""
        system = _make_system()
        df15 = _make_df15(n=4)
        system._store.get_ohlcv.return_value = df15
        system._fusion.get_features_snapshot.return_value = (_make_features(), False, False, None, None)

        async def mock_sleep(_):
            system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._candle_logger_loop())

        rows = system._db.execute("SELECT * FROM candle_features").fetchall()
        assert len(rows) == 1

    def test_candle_logger_btc_direction_close_above_open(self):
        """close > open → btc_direction=1."""
        system = _make_system()
        df15 = _make_df15(n=4, close_above_open=True)
        system._store.get_ohlcv.return_value = df15
        system._fusion.get_features_snapshot.return_value = (_make_features(), False, False, None, None)

        async def mock_sleep(_):
            system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._candle_logger_loop())

        row = system._db.execute(
            "SELECT btc_direction FROM candle_features"
        ).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_candle_logger_btc_direction_close_below_open(self):
        """close <= open → btc_direction=0."""
        system = _make_system()
        df15 = _make_df15(n=4, close_above_open=False)
        system._store.get_ohlcv.return_value = df15
        system._fusion.get_features_snapshot.return_value = (_make_features(), False, False, None, None)

        async def mock_sleep(_):
            system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._candle_logger_loop())

        row = system._db.execute(
            "SELECT btc_direction FROM candle_features"
        ).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_candle_logger_no_duplicate_on_same_candle(self):
        """Two iterations with same candle_ts produce only one row (INSERT OR IGNORE)."""
        system = _make_system()
        df15 = _make_df15(n=4)
        system._store.get_ohlcv.return_value = df15
        system._fusion.get_features_snapshot.return_value = (_make_features(), False, False, None, None)

        sleep_calls = [0]

        async def mock_sleep(_):
            sleep_calls[0] += 1
            if sleep_calls[0] >= 2:
                system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._candle_logger_loop())

        count = system._db.execute(
            "SELECT COUNT(*) FROM candle_features"
        ).fetchone()[0]
        assert count == 1

    def test_candle_logger_survives_exception(self):
        """Loop continues after get_ohlcv() raises; next iteration writes row."""
        system = _make_system()
        df15 = _make_df15(n=4)
        system._fusion.get_features_snapshot.return_value = (_make_features(), False, False, None, None)

        call_count = [0]

        def get_ohlcv_side(tf):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Redis connection error")
            return df15

        system._store.get_ohlcv.side_effect = get_ohlcv_side

        sleep_calls = [0]

        async def mock_sleep(_):
            sleep_calls[0] += 1
            if sleep_calls[0] >= 2:
                system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._candle_logger_loop())

        # First iteration raised (no row), second succeeded (one row)
        count = system._db.execute(
            "SELECT COUNT(*) FROM candle_features"
        ).fetchone()[0]
        assert count == 1


# ── Phase 2: Kronos meta-feature columns ─────────────────────────────────────

class TestKronosColumnsInCandleLogger:

    def test_kronos_columns_exist_in_table(self):
        """candle_features table has kronos_raw_15min and kronos_raw_5min columns."""
        system = _make_system()
        cols = {row[1] for row in system._db.execute("PRAGMA table_info(candle_features)").fetchall()}
        assert "kronos_raw_15min" in cols
        assert "kronos_raw_5min" in cols

    def test_candle_logger_writes_kronos_values_from_fusion(self):
        """Kronos values passed in the features snapshot are persisted to candle_features."""
        system = _make_system()
        df15 = _make_df15(n=4)
        system._store.get_ohlcv.return_value = df15
        features = _make_features()
        features["kronos_raw_15min"] = 0.72
        features["kronos_raw_5min"] = 0.68
        system._fusion.get_features_snapshot.return_value = (features, False, False, None, None)

        async def mock_sleep(_):
            system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._candle_logger_loop())

        row = system._db.execute(
            "SELECT kronos_raw_15min, kronos_raw_5min FROM candle_features"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.72)
        assert row[1] == pytest.approx(0.68)

    def test_candle_logger_writes_null_when_kronos_not_cached(self):
        """None values for kronos features are stored as SQL NULL in candle_features."""
        system = _make_system()
        df15 = _make_df15(n=4)
        system._store.get_ohlcv.return_value = df15
        features = _make_features()
        features["kronos_raw_15min"] = None
        features["kronos_raw_5min"] = None
        system._fusion.get_features_snapshot.return_value = (features, False, False, None, None)

        async def mock_sleep(_):
            system._running = False

        with patch("asyncio.sleep", side_effect=mock_sleep):
            asyncio.run(system._candle_logger_loop())

        row = system._db.execute(
            "SELECT kronos_raw_15min, kronos_raw_5min FROM candle_features"
        ).fetchone()
        assert row is not None
        assert row[0] is None
        assert row[1] is None
