"""
Tests for StratifiedEdgeTracker.

All tests use fakeredis to avoid a live Redis dependency.
"""

import json
from collections import deque
from unittest.mock import patch

import fakeredis
import pytest

from btc_kalshi_system.signal.stratified_edge_tracker import (
    StratifiedEdgeTracker,
    _redis_key,
)


def make_tracker(fake_redis=None) -> StratifiedEdgeTracker:
    """StratifiedEdgeTracker backed by a fresh (or provided) fakeredis instance."""
    if fake_redis is None:
        fake_redis = fakeredis.FakeRedis(decode_responses=True)
    with patch("redis.from_url", return_value=fake_redis):
        return StratifiedEdgeTracker()


# ── test_record_two_regimes_independent ────────────────────────────────────────

def test_record_two_regimes_independent():
    """Recording to one regime must not affect another regime's edge."""
    tracker = make_tracker()

    # trending_up: 3 wins at market 0.40 → realized edge = 0.60
    for _ in range(3):
        tracker.record("trending_up", predicted_prob=0.7, outcome=1, market_price=0.40)

    # ranging: 2 losses at market 0.50 → realized edge = -0.50
    for _ in range(2):
        tracker.record("ranging", predicted_prob=0.55, outcome=0, market_price=0.50)

    assert tracker.current_edge("trending_up") == pytest.approx(0.60)
    assert tracker.current_edge("ranging") == pytest.approx(-0.50)

    # Other regimes untouched — should return 0.0
    assert tracker.current_edge("trending_down") == pytest.approx(0.0)
    assert tracker.current_edge("high_uncertainty") == pytest.approx(0.0)


# ── test_unknown_regime_logs_warning_no_raise ──────────────────────────────────

def test_unknown_regime_logs_warning_no_raise():
    """Recording to an unrecognised regime must log a warning and not raise.

    loguru bypasses pytest's capsys/caplog by writing directly to file
    descriptors. We add a temporary in-memory loguru sink to capture messages.
    """
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        tracker = make_tracker()
        # Must not raise — test would fail automatically if it did
        tracker.record("FAKE_REGIME", predicted_prob=0.6, outcome=1, market_price=0.5)
    finally:
        logger.remove(sink_id)

    assert any("FAKE_REGIME" in m for m in messages), (
        f"Expected WARNING mentioning 'FAKE_REGIME', captured: {messages}"
    )


# ── test_summary_returns_all_four_regimes ──────────────────────────────────────

def test_summary_returns_all_four_regimes():
    """summary() keys must be exactly the four REGIMES, nothing more/less."""
    tracker = make_tracker()

    # Record one trade in two regimes so not everything is zero
    tracker.record("trending_up", predicted_prob=0.7, outcome=1, market_price=0.40)
    tracker.record("high_uncertainty", predicted_prob=0.55, outcome=0, market_price=0.50)

    result = tracker.summary()

    assert set(result.keys()) == set(StratifiedEdgeTracker.REGIMES)
    assert result["trending_up"] == pytest.approx(0.60)
    assert result["high_uncertainty"] == pytest.approx(-0.50)
    assert result["trending_down"] == pytest.approx(0.0)
    assert result["ranging"] == pytest.approx(0.0)


# ── test_unknown_regime_records_into_own_bucket ───────────────────────────────

def test_unknown_regime_records_into_own_bucket():
    """record('unknown', ...) must track into its own bucket, not get dropped."""
    tracker = make_tracker()
    # outcome=1 at market_price=0.55 → realized edge = 1 - 0.55 = 0.45
    tracker.record("unknown", predicted_prob=0.6, outcome=1, market_price=0.55)
    assert tracker.current_edge("unknown") != 0.0, "expected trade to be recorded in 'unknown' bucket"
    assert tracker.is_above_threshold("unknown"), "edge 0.45 >> 0.005 threshold — should be above"


# ── test_unknown_regime_in_summary ────────────────────────────────────────────

def test_unknown_regime_in_summary():
    """'unknown' must appear in summary() and reflect recorded trades."""
    tracker = make_tracker()
    tracker.record("unknown", predicted_prob=0.6, outcome=1, market_price=0.55)
    result = tracker.summary()
    assert "unknown" in result, "summary() must include the 'unknown' bucket"
    assert result["unknown"] != 0.0, "summary()['unknown'] must reflect the recorded trade"


# ── test_redis_persistence_survives_new_instance ───────────────────────────────

def test_redis_persistence_survives_new_instance():
    """A new tracker sharing the same Redis must recover prior history."""
    shared_redis = fakeredis.FakeRedis(decode_responses=True)

    with patch("redis.from_url", return_value=shared_redis):
        tracker1 = StratifiedEdgeTracker()

    # Record 3 wins in trending_up
    for _ in range(3):
        tracker1.record("trending_up", predicted_prob=0.7, outcome=1, market_price=0.40)

    expected_edge = tracker1.current_edge("trending_up")

    # Simulate process restart: new instance, same Redis backing
    with patch("redis.from_url", return_value=shared_redis):
        tracker2 = StratifiedEdgeTracker()

    assert tracker2.current_edge("trending_up") == pytest.approx(expected_edge)

    # Other regimes should still be empty
    assert tracker2.current_edge("ranging") == pytest.approx(0.0)
