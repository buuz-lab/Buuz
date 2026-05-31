"""
Tests for PositionMonitor mid-trade exit logic.

Uses fakeredis; mocks Kalshi API and KronosEngine. No real network calls.
"""
import asyncio
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import numpy as np
import pandas as pd
import pytest

from btc_kalshi_system.execution.position_monitor import PositionMonitor, _parse_orderbook_bbo
from btc_kalshi_system.portfolio.monitor import OpenPosition


def _make_position(
    trade_id: str = "test-trade-1",
    direction: int = 1,
    elapsed: float = 400.0,  # >300s so T5 check fires
) -> OpenPosition:
    return OpenPosition(
        trade_id=trade_id,
        ticker="KXBTC15M-25JUN-T95000",
        timeframe="15min",
        direction=direction,
        strike=95000.0,
        contracts=2,
        entry_price_cents=55,
        kelly_dollars=10.0,
        timestamp=time.time() - elapsed,
        calibrated_prob=0.65,
    )


def _make_db():
    """Create an in-memory SQLite DB with required tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            exit_reason TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_snapshots (
            trade_id TEXT, snapshot_window TEXT,
            snapshot_ts TEXT, funding_rate REAL, funding_rate_trend REAL,
            oi_delta_pct REAL, cvd_normalized REAL, basis_spread_pct REAL,
            brti_volatility_1h REAL, cvd_velocity REAL, cvd_acceleration REAL,
            brti_momentum_5min REAL, brti_momentum_15min REAL, candle_progress REAL,
            hour_sin REAL, hour_cos REAL, kalshi_implied_prob REAL,
            funding_window_proximity REAL, trend_slope_1h REAL, trend_r2_1h REAL,
            hourly_sr_proximity REAL, range_breakout_flag REAL, tape_speed_tpm REAL,
            atm_iv REAL, iv_rv_spread REAL, pcr_oi REAL,
            term_structure_slope REAL, skew_25d REAL, kalshi_spread_normalized REAL,
            kronos_prob REAL, regime_direction INTEGER, exit_triggered INTEGER,
            PRIMARY KEY (trade_id, snapshot_window)
        )
    """)
    conn.commit()
    return conn


def _make_monitor_components(regime_direction: int = 0, kronos_prob: float = 0.3, clf_none: bool = False):
    """Build mocked PositionMonitor dependencies."""
    portfolio_monitor = MagicMock()

    regime_model = MagicMock()
    if clf_none:
        regime_model._clf = None
    else:
        regime_model._clf = MagicMock()  # not None = trained
        regime_model.get_regime.return_value = {
            "prob_up": kronos_prob, "direction": regime_direction, "confidence": 0.7
        }

    kronos_engine = MagicMock()
    kronos_engine.run_monte_carlo.return_value = kronos_prob

    feature_store = MagicMock()
    # Set up feature store so _regime_features doesn't go stale
    prices = np.linspace(95000, 95100, 15).tolist()
    idx = pd.date_range("2024-01-01", periods=15, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [0.0] * 15, "amount": [0.0] * 15,
    }, index=idx)
    # 26 candles needed for btc_24h_return (Feature 28) to not mark stale
    h_prices = np.linspace(94000, 96000, 26).tolist()
    h_idx = pd.date_range("2024-01-01", periods=26, freq="1h", tz="UTC")
    df1h = pd.DataFrame({
        "open": h_prices, "high": h_prices, "low": h_prices, "close": h_prices,
        "volume": [0.0] * 26, "amount": [0.0] * 26,
    }, index=h_idx)
    def ohlcv_side_effect(tf):
        return df1h if tf == "1h" else df5
    feature_store.get_ohlcv.side_effect = ohlcv_side_effect
    feature_store.get_raw_ticks.return_value = None
    now = time.time()
    feature_store._redis = fakeredis.FakeRedis()
    for i, (val, score) in enumerate([
        (0.1, now - 600), (0.2, now - 480), (0.3, now - 360), (0.4, now - 240), (0.5, now - 120),
    ]):
        feature_store._redis.zadd("regime:cvd_history", {str(val): score})

    router = MagicMock()
    router.get_orderbook.return_value = {
        "orderbook_fp": {
            "yes_dollars": [["0.53", "5"]],
            "no_dollars": [["0.45", "5"]],
        }
    }

    from btc_kalshi_system.signal.fusion import SignalFusionEngine
    from btc_kalshi_system.models.calibrator import Calibrator
    from btc_kalshi_system.models.deepseek_parser import DeepSeekContextParser
    from btc_kalshi_system.models.kronos_engine import KronosEngine

    fusion_engine = SignalFusionEngine(
        feature_store=feature_store,
        kronos_engine=MagicMock(),
        calibrator=MagicMock(),
        regime_model=MagicMock(),
        deepseek_parser=MagicMock(),
    )
    fusion_engine.update_market_context({
        "funding_rate": 0.0001, "cvd_normalized": 0.3, "kalshi_mid_cents": 55.0
    })

    return portfolio_monitor, regime_model, kronos_engine, feature_store, router, fusion_engine


def _make_position_monitor(regime_direction=0, kronos_prob=0.3, clf_none=False, db_conn=None):
    """Return a PositionMonitor with a temp DB path that uses db_conn's schema."""
    pm, rm, ke, fs, router, fe = _make_monitor_components(
        regime_direction=regime_direction, kronos_prob=kronos_prob, clf_none=clf_none
    )
    # Write to a temp file DB so we can inspect results
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            exit_reason TEXT,
            outcome INTEGER,
            pnl_dollars REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_snapshots (
            trade_id TEXT, snapshot_window TEXT, snapshot_ts TEXT,
            funding_rate REAL, funding_rate_trend REAL, oi_delta_pct REAL,
            cvd_normalized REAL, basis_spread_pct REAL, brti_volatility_1h REAL,
            cvd_velocity REAL, cvd_acceleration REAL, brti_momentum_5min REAL,
            brti_momentum_15min REAL, candle_progress REAL, hour_sin REAL, hour_cos REAL,
            kalshi_implied_prob REAL, funding_window_proximity REAL, trend_slope_1h REAL,
            trend_r2_1h REAL, hourly_sr_proximity REAL, range_breakout_flag REAL,
            tape_speed_tpm REAL, atm_iv REAL, iv_rv_spread REAL, pcr_oi REAL,
            term_structure_slope REAL, skew_25d REAL, kalshi_spread_normalized REAL,
            kronos_prob REAL, regime_direction INTEGER, exit_triggered INTEGER,
            PRIMARY KEY (trade_id, snapshot_window)
        )
    """)
    conn.commit()
    conn.close()
    monitor = PositionMonitor(
        portfolio_monitor=pm,
        regime_model=rm,
        kronos_engine=ke,
        feature_store=fs,
        router=router,
        fusion_engine=fe,
        db_path=tmp.name,
    )
    return monitor, tmp.name


# ── _parse_orderbook_bbo ───────────────────────────────────────────────────────

def test_parse_bbo_orderbook_fp_format():
    book = {"orderbook_fp": {"yes_dollars": [["0.53", "5"]], "no_dollars": [["0.45", "5"]]}}
    bid, ask = _parse_orderbook_bbo(book)
    assert bid == 53
    assert ask == 55  # round((1 - 0.45) * 100)


def test_parse_bbo_empty_no_bids_returns_zeros():
    book = {"orderbook_fp": {"yes_dollars": [["0.53", "5"]], "no_dollars": []}}
    bid, ask = _parse_orderbook_bbo(book)
    assert bid == 0 and ask == 0


def test_parse_bbo_invalid_returns_zeros():
    assert _parse_orderbook_bbo({}) == (0, 0)
    assert _parse_orderbook_bbo({"orderbook_fp": {}}) == (0, 0)


# ── _evaluate: both models flip → exit ────────────────────────────────────────

def test_evaluate_both_flip_triggers_exit():
    """Regime and Kronos both predict DOWN when entry was UP → exit_triggered=1."""
    monitor, db_path = _make_position_monitor(
        regime_direction=0,   # flipped from entry direction=1
        kronos_prob=0.3,      # < 0.5 → direction=0 (also flipped)
    )
    position = _make_position(direction=1)
    asyncio.run(monitor._evaluate(position, "t5"))

    monitor.portfolio_monitor.remove_position.assert_called_once_with(position.trade_id)
    monitor.router.place_order.assert_called_once()

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT exit_triggered FROM trade_snapshots WHERE trade_id = ?",
        (position.trade_id,)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 1


def test_evaluate_models_agree_with_entry_no_exit():
    """Regime and Kronos both agree with entry direction → no exit."""
    monitor, db_path = _make_position_monitor(
        regime_direction=1,   # agrees with entry direction=1
        kronos_prob=0.7,      # > 0.5 → direction=1 (agrees)
    )
    position = _make_position(direction=1)
    asyncio.run(monitor._evaluate(position, "t5"))

    monitor.portfolio_monitor.remove_position.assert_not_called()
    monitor.router.place_order.assert_not_called()

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT exit_triggered FROM trade_snapshots WHERE trade_id = ?",
        (position.trade_id,)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 0


def test_evaluate_bootstrap_no_exit():
    """When regime._clf is None (bootstrap), snapshots written but no exit attempted."""
    monitor, db_path = _make_position_monitor(clf_none=True)
    position = _make_position(direction=1)
    asyncio.run(monitor._evaluate(position, "t5"))

    monitor.portfolio_monitor.remove_position.assert_not_called()
    monitor.router.place_order.assert_not_called()

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT trade_id FROM trade_snapshots WHERE trade_id = ?",
        (position.trade_id,)
    ).fetchone()
    conn.close()
    assert row is not None  # snapshot was written


# ── Schema migration idempotency ─────────────────────────────────────────────

def test_schema_migration_idempotent():
    """Running all 15 ALTER TABLE migrations twice must not raise."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from main import _TRADES_COLUMN_MIGRATIONS

    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE trades (
            trade_id TEXT PRIMARY KEY,
            timestamp TEXT,
            funding_rate REAL,
            funding_rate_trend REAL,
            oi_delta_pct REAL,
            cvd_normalized REAL,
            basis_spread_pct REAL,
            brti_volatility_1h REAL,
            features_stale INTEGER
        )
    """)
    conn.commit()

    # Run twice — should not raise on second pass
    for _ in range(2):
        for col_name, col_def in _TRADES_COLUMN_MIGRATIONS:
            try:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")
            except Exception:
                pass  # "duplicate column name" is expected on second pass

    # Verify all new columns exist
    cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    for col_name, _ in _TRADES_COLUMN_MIGRATIONS:
        assert col_name in cols, f"Column {col_name} missing after migration"
    conn.close()


# ── _execute_exit: exit order failure recovery ────────────────────────────────

def test_exit_failure_resolves_outcome_from_finalized_market():
    """When place_order raises, outcome is written to DB from Kalshi market result."""
    monitor, db_path = _make_position_monitor(regime_direction=0, kronos_prob=0.3)
    position = _make_position(direction=1, trade_id="fail-exit-trade")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trades (trade_id, exit_reason, outcome) VALUES (?, NULL, NULL)",
        (position.trade_id,),
    )
    conn.commit()
    conn.close()

    monitor.router.place_order.side_effect = Exception("BOTH_FAILED")
    monitor.router._raw = MagicMock()
    monitor.router._raw._request.return_value = {
        "market": {"status": "finalized", "result": "yes"}
    }

    asyncio.run(monitor._execute_exit(position, "t5", 53, 55))

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT outcome, exit_reason FROM trades WHERE trade_id = ?",
        (position.trade_id,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 1, f"Expected outcome=1 (YES settled, direction=1), got {row[0]}"
    assert "market_resolved" in (row[1] or ""), f"exit_reason should indicate market resolution, got {row[1]}"


def test_exit_failure_logs_error_when_market_not_finalized():
    """When place_order raises and market isn't settled yet, logs error without crashing."""
    import contextlib
    import io
    from loguru import logger as _loguru

    monitor, db_path = _make_position_monitor(regime_direction=0, kronos_prob=0.3)
    position = _make_position(direction=1, trade_id="unresolved-trade")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trades (trade_id, exit_reason, outcome) VALUES (?, NULL, NULL)",
        (position.trade_id,),
    )
    conn.commit()
    conn.close()

    monitor.router.place_order.side_effect = Exception("BOTH_FAILED")
    monitor.router._raw = MagicMock()
    monitor.router._raw._request.return_value = {
        "market": {"status": "open", "result": ""}
    }

    records = []
    sid = _loguru.add(lambda msg: records.append(str(msg)), level="ERROR", format="{level}:{message}")
    try:
        asyncio.run(monitor._execute_exit(position, "t5", 53, 55))
    finally:
        _loguru.remove(sid)

    # Outcome must remain NULL — market not settled
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT outcome FROM trades WHERE trade_id = ?", (position.trade_id,)
    ).fetchone()
    conn.close()
    assert row[0] is None, f"Outcome should remain NULL for open market, got {row[0]}"
    assert any("manual" in r.lower() or "not yet" in r.lower() or "finalized" in r.lower() for r in records), \
        f"Expected an error log about manual resolution, got: {records}"
