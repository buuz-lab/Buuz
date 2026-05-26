"""
Tests for gate_rejections table: logging blocked trades and resolving outcomes.

Uses in-memory SQLite; mocks the Kalshi API call. No Redis dependency.
"""
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from btc_kalshi_system.execution.pretrade_checklist import ChecklistResult
from btc_kalshi_system.signal.fusion import TradingSignal


# ── Helpers ───────────────────────────────────────────────────────────────────

_GATE_REJECTIONS_DDL = """
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
    resolved_at      REAL DEFAULT NULL,
    aged_out         INTEGER DEFAULT 0,
    shadow           INTEGER DEFAULT 0,
    kalshi_mid_at_block REAL DEFAULT NULL
)
"""


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(_GATE_REJECTIONS_DDL)
    conn.commit()
    return conn


def _make_signal(
    direction: int = 1,
    prob: float = 0.65,
    regime: str = "TRENDING_BULLISH",
    features: dict | None = None,
) -> TradingSignal:
    return TradingSignal(
        direction=direction,
        calibrated_prob=prob,
        kronos_raw=prob,
        kronos_calibrated=prob,
        regime_prob=prob,
        regime_direction=direction,
        deepseek_regime=regime,
        timeframe="15min",
        strike=96250.0,
        timestamp=datetime.now(timezone.utc),
        regime_features=features or {"cvd_normalized": 0.3, "funding_rate": 0.01},
    )


def _make_trader(db: sqlite3.Connection):
    """Construct a KronosV2 instance bypassing __init__, injecting only needed attrs."""
    from main import KronosV2

    trader = object.__new__(KronosV2)
    trader._db = db
    trader._router = MagicMock()
    trader._fusion = MagicMock()
    trader._checklist = MagicMock()
    trader._monitor = MagicMock()
    trader._edge_tracker = MagicMock()
    trader._redis = MagicMock()
    trader._drift_monitor = MagicMock()
    trader._drift_monitor.is_drifting.return_value = False
    trader._dir_tracker = MagicMock()
    trader._dir_tracker.get_win_rate.return_value = None
    trader._monitor.ticker_direction_count.return_value = 0
    trader._monitor.get_current_exposure.return_value = 0.0
    trader._monitor.has_timeframe_position.return_value = False
    import pandas as pd
    trader._cached_kronos = {
        "prob": 0.65,
        "candle_ts": pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
        "computed_at": time.time(),
        "strike": 96250.0,
    }
    return trader


def _insert_pending_rejection(
    db: sqlite3.Connection,
    ticker: str = "KXBTC15M-25MAY24-T96250",
    direction: int = 1,
    age_seconds: float = 1000.0,
    failed_gate: int = 3,
) -> str:
    """Insert a gate_rejections row and return its rejection_id."""
    rejection_id = str(uuid.uuid4())
    db.execute(
        """INSERT INTO gate_rejections
           (rejection_id, timestamp, ticker, timeframe, direction,
            failed_gate, failed_reason, signal_prob, deepseek_regime,
            kalshi_mid_cents, features)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            rejection_id,
            time.time() - age_seconds,
            ticker,
            "15min",
            direction,
            failed_gate,
            "Exposure limit exceeded",
            0.65,
            "TRENDING_BULLISH",
            55,
            json.dumps({"cvd_normalized": 0.3}),
        ),
    )
    db.commit()
    return rejection_id


# ── Test 1: gate failure writes row ──────────────────────────────────────────

def test_gate_failure_writes_rejection_row():
    """On checklist failure, _process_market inserts a gate_rejections row with
    correct failed_gate, features JSON, direction, and outcome=NULL."""
    db = _make_db()
    trader = _make_trader(db)

    signal = _make_signal(direction=1, features={"cvd_normalized": 0.3, "funding_rate": 0.01})

    # Legacy-format orderbook mock (bid 55¢, ask 45¢ → mid 50¢)
    trader._router.get_orderbook.return_value = {
        "orderbook": {"yes": [[55, 10]], "no": [[45, 10]]}
    }
    trader._fusion.get_signal.return_value = signal
    trader._checklist.run.return_value = ChecklistResult(
        passed=False,
        failed_gate=3,
        failed_reason="Exposure limit exceeded",
        kelly_dollars=0.0,
        kelly_contracts=0,
    )

    market = {
        "ticker": "KXBTC15M-25MAY24-T96250",
        "market_type": "15min",
        "floor_strike": 96250.0,
        "close_time": None,
    }
    trader._process_market(market, 96000.0)

    rows = db.execute("SELECT * FROM gate_rejections").fetchall()
    assert len(rows) == 1, "Expected exactly one gate_rejections row"

    row = db.execute(
        """SELECT ticker, direction, failed_gate, failed_reason, features, outcome, resolved_at
           FROM gate_rejections"""
    ).fetchone()
    ticker_val, direction_val, failed_gate_val, failed_reason_val, features_json, outcome_val, resolved_at_val = row

    assert ticker_val == "KXBTC15M-25MAY24-T96250"
    assert direction_val == 1
    assert failed_gate_val == 3
    assert "Exposure" in failed_reason_val
    assert outcome_val is None
    assert resolved_at_val is None

    features = json.loads(features_json)
    assert "cvd_normalized" in features
    assert features["cvd_normalized"] == pytest.approx(0.3)


# ── Test 2: resolve outcome=1 (would have won) ────────────────────────────────

def test_resolve_gate_rejection_win():
    """When the market finalizes 'yes' and direction=1 (YES→UP),
    outcome is set to 1 (would have won)."""
    db = _make_db()
    trader = _make_trader(db)

    rejection_id = _insert_pending_rejection(db, direction=1, age_seconds=1000.0)

    trader._router._raw._request.return_value = {
        "market": {"status": "finalized", "result": "yes"}
    }

    trader._resolve_gate_rejections()

    row = db.execute(
        "SELECT outcome, resolved_at FROM gate_rejections WHERE rejection_id=?",
        (rejection_id,),
    ).fetchone()
    assert row[0] == 1, "direction=1 + result='yes' → outcome should be 1"
    assert row[1] is not None, "resolved_at should be set"


# ── Test 3: resolve outcome=0 (would have lost) ───────────────────────────────

def test_resolve_gate_rejection_loss():
    """When the market finalizes 'yes' but direction=0 (NO→DOWN),
    outcome is set to 0 (would have lost)."""
    db = _make_db()
    trader = _make_trader(db)

    rejection_id = _insert_pending_rejection(db, direction=0, age_seconds=1000.0)

    trader._router._raw._request.return_value = {
        "market": {"status": "finalized", "result": "yes"}
    }

    trader._resolve_gate_rejections()

    row = db.execute(
        "SELECT outcome FROM gate_rejections WHERE rejection_id=?",
        (rejection_id,),
    ).fetchone()
    assert row[0] == 0, "direction=0 + result='yes' → outcome should be 0"


# ── Test 4: young rows are skipped ────────────────────────────────────────────

def test_resolve_skips_rows_younger_than_900s():
    """Rows inserted less than 900 seconds ago are not yet eligible for resolution."""
    db = _make_db()
    trader = _make_trader(db)

    # Insert a row only 100 seconds old — below the 900s wait window
    rejection_id = _insert_pending_rejection(db, direction=1, age_seconds=100.0)

    # Even if the API would return a finalized result, it should not be called
    trader._router._raw._request.return_value = {
        "market": {"status": "finalized", "result": "yes"}
    }

    trader._resolve_gate_rejections()

    row = db.execute(
        "SELECT outcome, aged_out FROM gate_rejections WHERE rejection_id=?",
        (rejection_id,),
    ).fetchone()
    assert row[0] is None, "Row younger than 900s must remain unresolved (outcome=NULL)"
    assert row[1] == 0, "aged_out must stay 0 for a young unresolved row"


# ── Test 5: age-out sets aged_out=1, outcome stays NULL ──────────────────────

def test_aged_out_row_sets_flag_not_outcome():
    """Rows older than MAX_POSITION_AGE_SECONDS get aged_out=1; outcome stays NULL."""
    db = _make_db()
    trader = _make_trader(db)

    # 90000s > 86400s (_MAX_POSITION_AGE_SECONDS), so age-out fires
    rejection_id = _insert_pending_rejection(db, direction=1, age_seconds=90000.0)

    trader._resolve_gate_rejections()

    row = db.execute(
        "SELECT outcome, aged_out FROM gate_rejections WHERE rejection_id=?",
        (rejection_id,),
    ).fetchone()
    assert row[0] is None, "aged-out row must keep outcome=NULL (not a sentinel value)"
    assert row[1] == 1, "aged_out must be set to 1"
