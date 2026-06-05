"""Tests for scripts/auto_retrain_regime.py — written TDD before implementation."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.auto_retrain_regime import (
    get_qualifying_count,
    load_marker,
    save_marker,
    brier_score,
    evaluate_deployed_model,
    should_retrain,
    should_deploy,
    _MARKER_PATH,
    _ROW_TRIGGER_DELTA,
    _TIME_TRIGGER_DAYS,
    _MIN_ROWS,
    _WINDOW,
    _HOLDOUT_SIZE,
)


# ── helpers ───────────────────────────────────────────────────────────────────

_FEATURE_COLS_FOR_DB = [
    "funding_rate", "funding_rate_trend", "oi_delta_pct", "cvd_normalized",
    "basis_spread_pct", "brti_volatility_1h", "cvd_velocity", "cvd_acceleration",
    "brti_momentum_5min", "brti_momentum_15min", "candle_progress",
    "hour_sin", "hour_cos", "funding_window_proximity",
    "trend_slope_1h", "trend_r2_1h", "hourly_sr_proximity", "range_breakout_flag",
    "tape_speed_tpm", "large_print_direction", "volume_ratio_1h",
    "atm_iv", "iv_rv_spread", "pcr_oi", "term_structure_slope", "skew_25d",
    "btc_24h_return", "kronos_raw_15min", "kronos_raw_5min",
]


def _make_db(rows: list[dict]) -> str:
    """Create a temp SQLite DB with candle_features rows; return path string."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    feat_cols_ddl = " ".join(f", {c} REAL" for c in _FEATURE_COLS_FOR_DB)
    conn.execute(f"""
        CREATE TABLE candle_features (
            id INTEGER PRIMARY KEY,
            candle_ts REAL,
            features_stale INTEGER,
            btc_direction INTEGER
            {feat_cols_ddl}
        )
    """)
    for i, r in enumerate(rows):
        feat_vals = [
            0.001, 0.0001, 0.0001, 0.3,
            0.0005, 0.5, 0.0001, 0.00001,
            0.002, 0.003, 0.5,
            0.5, 0.866, 0.1,
            0.0001, 0.8, 0.5, 0.0,
            5.0, 1.0, 1.0,
            50.0, 0.5, 1.1, 0.01, 0.02,
            0.02,
            r.get("kronos_raw_15min", 0.6),
            r.get("kronos_raw_5min", 0.55),
        ]
        placeholders = ", ".join(["?"] * (4 + len(_FEATURE_COLS_FOR_DB)))
        col_names = "id, candle_ts, features_stale, btc_direction, " + ", ".join(_FEATURE_COLS_FOR_DB)
        conn.execute(
            f"INSERT INTO candle_features ({col_names}) VALUES ({placeholders})",
            (i, r.get("candle_ts", float(i)), r.get("features_stale", 0),
             r.get("btc_direction", 1), *feat_vals),
        )
    conn.commit()
    conn.close()
    return tmp.name


def _qualifying_row(**overrides) -> dict:
    base: dict = {"features_stale": 0, "btc_direction": 1, "candle_ts": 1000.0}
    base.update(overrides)
    return base


# ── get_qualifying_count ──────────────────────────────────────────────────────

def test_get_qualifying_count_empty_db():
    db = _make_db([])
    assert get_qualifying_count(db) == 0


def test_get_qualifying_count_with_qualifying_rows():
    rows = [_qualifying_row() for _ in range(5)]
    db = _make_db(rows)
    assert get_qualifying_count(db) == 5


def test_get_qualifying_count_excludes_stale():
    rows = [_qualifying_row() for _ in range(3)] + [_qualifying_row(features_stale=1) for _ in range(2)]
    db = _make_db(rows)
    assert get_qualifying_count(db) == 3


def test_get_qualifying_count_excludes_null_direction():
    rows = [_qualifying_row() for _ in range(3)] + [_qualifying_row(btc_direction=None) for _ in range(2)]
    db = _make_db(rows)
    assert get_qualifying_count(db) == 3


# ── load_marker / save_marker ─────────────────────────────────────────────────

def test_load_marker_returns_none_when_missing(tmp_path):
    with patch("scripts.auto_retrain_regime._MARKER_PATH", str(tmp_path / "nonexistent.json")):
        assert load_marker() is None


def test_load_marker_returns_none_when_corrupt(tmp_path):
    p = tmp_path / "marker.json"
    p.write_text("not json {{{")
    with patch("scripts.auto_retrain_regime._MARKER_PATH", str(p)):
        assert load_marker() is None


def test_load_marker_returns_none_when_missing_keys(tmp_path):
    p = tmp_path / "marker.json"
    p.write_text(json.dumps({"trained_at_rows": 100}))
    with patch("scripts.auto_retrain_regime._MARKER_PATH", str(p)):
        assert load_marker() is None


def test_load_marker_returns_dict_when_valid(tmp_path):
    p = tmp_path / "marker.json"
    data = {
        "trained_at_rows": 700,
        "trained_at_timestamp": "2026-05-01T03:00:00+00:00",
        "total_rows_at_train": 700,
        "holdout_brier": 0.234,
    }
    p.write_text(json.dumps(data))
    with patch("scripts.auto_retrain_regime._MARKER_PATH", str(p)):
        m = load_marker()
    assert m is not None
    assert m["trained_at_rows"] == 700
    assert m["holdout_brier"] == pytest.approx(0.234)


def test_save_marker_writes_expected_keys(tmp_path):
    marker_path = str(tmp_path / "regime_last_trained.json")
    with patch("scripts.auto_retrain_regime._MARKER_PATH", marker_path):
        save_marker(trained_at_rows=750, total_rows=750, holdout_brier=0.220)
    with open(marker_path) as f:
        data = json.load(f)
    assert data["trained_at_rows"] == 750
    assert "trained_at_timestamp" in data
    assert data["holdout_brier"] == pytest.approx(0.220)
    assert data["total_rows_at_train"] == 750


# ── brier_score ───────────────────────────────────────────────────────────────

def test_brier_score_perfect():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 1.0])
    assert brier_score(y, p) == pytest.approx(0.0)


def test_brier_score_coin_flip():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert brier_score(y, p) == pytest.approx(0.25)


def test_brier_score_worst():
    y = np.array([1.0, 1.0])
    p = np.array([0.0, 0.0])
    assert brier_score(y, p) == pytest.approx(1.0)


# ── evaluate_deployed_model ───────────────────────────────────────────────────

def test_evaluate_deployed_model_returns_none_when_no_model(tmp_path):
    X = np.random.rand(20, 28)
    y = np.random.randint(0, 2, 20)
    result = evaluate_deployed_model(str(tmp_path / "nonexistent.pkl"), X, y)
    assert result is None


def test_evaluate_deployed_model_returns_float_when_trained(tmp_path):
    from btc_kalshi_system.models.regime_model import RegimeModel
    m = RegimeModel()
    X = np.random.rand(100, 28)
    y = np.random.randint(0, 2, 100)
    m.train(X, y)
    model_path = str(tmp_path / "regime.pkl")
    m.save(model_path)

    X_holdout = np.random.rand(20, 28)
    y_holdout = np.random.randint(0, 2, 20)
    result = evaluate_deployed_model(model_path, X_holdout, y_holdout)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ── should_retrain ────────────────────────────────────────────────────────────

def test_should_retrain_force_overrides_all():
    assert should_retrain(count=0, marker=None, force=True) == "FORCE"


def test_should_retrain_row_trigger_fires():
    marker = {
        "trained_at_rows": 700,
        "trained_at_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    count = 700 + _ROW_TRIGGER_DELTA
    assert should_retrain(count=count, marker=marker) == "ROW-BASED"


def test_should_retrain_row_trigger_not_fired():
    marker = {
        "trained_at_rows": 700,
        "trained_at_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    count = 700 + _ROW_TRIGGER_DELTA - 1
    result = should_retrain(count=count, marker=marker)
    assert result != "ROW-BASED"


def test_should_retrain_time_trigger_fires_when_never_trained():
    # No marker → time trigger fires
    assert should_retrain(count=0, marker=None) == "TIME-BASED"


def test_should_retrain_time_trigger_fires_when_elapsed():
    old_ts = (datetime.now(timezone.utc) - timedelta(days=_TIME_TRIGGER_DAYS + 1)).isoformat()
    marker = {"trained_at_rows": 700, "trained_at_timestamp": old_ts}
    result = should_retrain(count=700, marker=marker)
    assert result == "TIME-BASED"


def test_should_retrain_no_trigger_fires():
    recent_ts = datetime.now(timezone.utc).isoformat()
    marker = {"trained_at_rows": 700, "trained_at_timestamp": recent_ts}
    count = 700 + _ROW_TRIGGER_DELTA - 1  # row delta not met
    result = should_retrain(count=count, marker=marker)
    assert result is None


# ── should_deploy ─────────────────────────────────────────────────────────────

def test_should_deploy_when_no_deployed_model():
    assert should_deploy(candidate_brier=0.30, deployed_brier=None) is True


def test_should_deploy_when_candidate_is_better():
    assert should_deploy(candidate_brier=0.22, deployed_brier=0.24) is True


def test_should_not_deploy_when_candidate_is_worse():
    assert should_deploy(candidate_brier=0.26, deployed_brier=0.24) is False


def test_should_not_deploy_when_candidate_equals_deployed():
    assert should_deploy(candidate_brier=0.24, deployed_brier=0.24) is False


# ── constants sanity ──────────────────────────────────────────────────────────

def test_constants_are_sane():
    assert _ROW_TRIGGER_DELTA > 0
    assert _TIME_TRIGGER_DAYS > 0
    assert _MIN_ROWS >= 672
    assert _WINDOW >= _MIN_ROWS
    assert _HOLDOUT_SIZE > 0
    assert _HOLDOUT_SIZE < _MIN_ROWS
