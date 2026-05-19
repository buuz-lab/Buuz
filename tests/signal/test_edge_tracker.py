import json

import fakeredis
import pytest

from btc_kalshi_system.signal.edge_tracker import EdgeTracker, _REDIS_KEY


def make_tracker(threshold: float = 0.005) -> EdgeTracker:
    """EdgeTracker with fakeredis and a fresh empty history."""
    tracker = EdgeTracker.__new__(EdgeTracker)
    tracker._redis = fakeredis.FakeRedis()
    tracker._threshold = threshold
    from collections import deque
    tracker._history = deque(maxlen=50)
    return tracker


# ── record / current_edge ──────────────────────────────────────────────────────

def test_current_edge_is_zero_when_empty():
    tracker = make_tracker()
    assert tracker.current_edge() == pytest.approx(0.0)


def test_record_appends_to_history():
    tracker = make_tracker()
    tracker.record(predicted_prob=0.6, outcome=1, market_price=0.5)
    assert len(tracker) == 1


def test_current_edge_is_mean_outcome_minus_market_price():
    """Realized edge = mean(outcome - market_price) across recorded trades."""
    tracker = make_tracker()
    tracker.record(predicted_prob=0.6, outcome=1, market_price=0.4)   # +0.6
    tracker.record(predicted_prob=0.55, outcome=0, market_price=0.5)  # -0.5
    tracker.record(predicted_prob=0.7, outcome=1, market_price=0.45)  # +0.55
    expected = (0.6 + (-0.5) + 0.55) / 3
    assert tracker.current_edge() == pytest.approx(expected)


def test_current_edge_all_wins_at_low_market_price_is_positive():
    tracker = make_tracker()
    for _ in range(5):
        tracker.record(predicted_prob=0.6, outcome=1, market_price=0.4)
    # Each trade contributes 1 - 0.4 = 0.6
    assert tracker.current_edge() == pytest.approx(0.6)


def test_current_edge_all_losses_is_negative():
    tracker = make_tracker()
    for _ in range(5):
        tracker.record(predicted_prob=0.6, outcome=0, market_price=0.5)
    # Each trade contributes 0 - 0.5 = -0.5
    assert tracker.current_edge() == pytest.approx(-0.5)


# ── is_above_threshold ─────────────────────────────────────────────────────────

def test_is_above_threshold_false_when_empty():
    tracker = make_tracker()
    assert tracker.is_above_threshold() is False


def test_is_above_threshold_true_when_edge_exceeds_threshold():
    tracker = make_tracker(threshold=0.005)
    # 3 wins at market 0.40 → edge = 0.6 per trade
    for _ in range(3):
        tracker.record(predicted_prob=0.6, outcome=1, market_price=0.4)
    assert tracker.is_above_threshold() is True


def test_is_above_threshold_false_when_edge_below_threshold():
    tracker = make_tracker(threshold=0.005)
    # Edge = 0.001 — below threshold
    tracker.record(predicted_prob=0.51, outcome=1, market_price=0.999)
    assert tracker.is_above_threshold() is False


def test_is_above_threshold_uses_threshold_param():
    tracker = make_tracker(threshold=0.5)
    # Edge = 0.2 per trade, below 0.5
    for _ in range(3):
        tracker.record(predicted_prob=0.55, outcome=1, market_price=0.8)
    assert tracker.current_edge() == pytest.approx(0.2)
    assert tracker.is_above_threshold() is False


# ── deque max length ───────────────────────────────────────────────────────────

def test_history_capped_at_50_entries():
    tracker = make_tracker()
    for i in range(75):
        tracker.record(predicted_prob=0.5, outcome=1, market_price=0.4)
    assert len(tracker) == 50


def test_only_last_50_used_for_edge_calculation():
    tracker = make_tracker()
    # First 50 wins at market 0.9 → edge = 0.1
    for _ in range(50):
        tracker.record(predicted_prob=0.95, outcome=1, market_price=0.9)
    # Then 50 losses at market 0.5 → edge = -0.5
    # After this, deque holds only the last 50: the losses.
    for _ in range(50):
        tracker.record(predicted_prob=0.55, outcome=0, market_price=0.5)
    assert tracker.current_edge() == pytest.approx(-0.5)


# ── Redis persistence ──────────────────────────────────────────────────────────

def test_record_writes_history_to_redis():
    tracker = make_tracker()
    tracker.record(predicted_prob=0.6, outcome=1, market_price=0.4)
    raw = tracker._redis.get(_REDIS_KEY)
    assert raw is not None
    loaded = json.loads(raw)
    assert isinstance(loaded, list)
    assert len(loaded) == 1
    assert loaded[0]["predicted_prob"] == pytest.approx(0.6)
    assert loaded[0]["outcome"] == 1
    assert loaded[0]["market_price"] == pytest.approx(0.4)


def test_load_from_redis_restores_history():
    """A new tracker with the same Redis backing should see prior history."""
    tracker = make_tracker()
    tracker.record(predicted_prob=0.6, outcome=1, market_price=0.4)
    tracker.record(predicted_prob=0.55, outcome=0, market_price=0.5)
    redis_client = tracker._redis

    # Simulate restart: new tracker reads from same Redis
    new_tracker = EdgeTracker.__new__(EdgeTracker)
    new_tracker._redis = redis_client
    new_tracker._threshold = 0.005
    from collections import deque
    new_tracker._history = deque(maxlen=50)
    new_tracker._load_from_redis()

    assert len(new_tracker) == 2
    assert new_tracker.current_edge() == pytest.approx(tracker.current_edge())


def test_load_from_empty_redis_leaves_history_empty():
    tracker = EdgeTracker.__new__(EdgeTracker)
    tracker._redis = fakeredis.FakeRedis()
    tracker._threshold = 0.005
    from collections import deque
    tracker._history = deque(maxlen=50)
    tracker._load_from_redis()
    assert len(tracker) == 0


def test_load_from_redis_with_corrupt_data_starts_empty():
    """Corrupt JSON should not crash; tracker silently starts fresh."""
    tracker = EdgeTracker.__new__(EdgeTracker)
    tracker._redis = fakeredis.FakeRedis()
    tracker._threshold = 0.005
    tracker._redis.set(_REDIS_KEY, "this is not valid json")
    from collections import deque
    tracker._history = deque(maxlen=50)
    tracker._load_from_redis()
    assert len(tracker) == 0


# ── __len__ convenience ────────────────────────────────────────────────────────

def test_len_returns_history_size():
    tracker = make_tracker()
    assert len(tracker) == 0
    tracker.record(predicted_prob=0.6, outcome=1, market_price=0.4)
    tracker.record(predicted_prob=0.55, outcome=0, market_price=0.5)
    assert len(tracker) == 2
