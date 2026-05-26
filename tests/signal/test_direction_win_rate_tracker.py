"""
Tests for DirectionWinRateTracker.

Uses fakeredis so no live Redis is needed.
"""
from collections import deque
from unittest.mock import patch

import fakeredis
import pytest

from btc_kalshi_system.signal.direction_win_rate_tracker import (
    DirectionWinRateTracker,
    _WINDOW,
    _KEY_NO,
    _KEY_YES,
)


def make_tracker() -> DirectionWinRateTracker:
    tracker = DirectionWinRateTracker.__new__(DirectionWinRateTracker)
    tracker._redis = fakeredis.FakeRedis(decode_responses=True)
    return tracker


# ── Basic win rate computation ────────────────────────────────────────────────

def test_get_win_rate_returns_none_when_fewer_than_10_records():
    tracker = make_tracker()
    for _ in range(9):
        tracker.record(direction=0, outcome=1)
    assert tracker.get_win_rate(0) is None


def test_get_win_rate_correct_with_10_records():
    tracker = make_tracker()
    # 7 wins + 3 losses = 0.7
    for _ in range(7):
        tracker.record(direction=0, outcome=1)
    for _ in range(3):
        tracker.record(direction=0, outcome=0)
    rate = tracker.get_win_rate(0)
    assert rate == pytest.approx(0.7)


def test_get_win_rate_10_wins_5_losses():
    tracker = make_tracker()
    for _ in range(10):
        tracker.record(direction=1, outcome=1)
    for _ in range(5):
        tracker.record(direction=1, outcome=0)
    rate = tracker.get_win_rate(1)
    assert rate == pytest.approx(10 / 15)


# ── Direction isolation ───────────────────────────────────────────────────────

def test_no_and_yes_tracked_independently():
    tracker = make_tracker()
    # YES: 10 wins
    for _ in range(10):
        tracker.record(direction=1, outcome=1)
    # NO: 10 losses
    for _ in range(10):
        tracker.record(direction=0, outcome=0)

    yes_rate = tracker.get_win_rate(1)
    no_rate = tracker.get_win_rate(0)
    assert yes_rate == pytest.approx(1.0)
    assert no_rate == pytest.approx(0.0)


# ── Eviction: only 30 records retained ───────────────────────────────────────

def test_eviction_retains_at_most_window_records():
    tracker = make_tracker()
    for i in range(31):
        tracker.record(direction=0, outcome=1)
    count = tracker._redis.zcard(_KEY_NO)
    assert count == _WINDOW


def test_eviction_oldest_record_evicted():
    tracker = make_tracker()
    # Record 30 wins then 1 loss — the loss should be the most recent
    for _ in range(30):
        tracker.record(direction=0, outcome=1)
    tracker.record(direction=0, outcome=0)
    # Now 30 records: 29 wins + 1 loss
    rate = tracker.get_win_rate(0)
    assert rate == pytest.approx(29 / 30)
