import time
from collections import deque

import pandas as pd
import pytest
import fakeredis

from btc_kalshi_system.data.feature_store import FeatureStore


def make_store() -> FeatureStore:
    """FeatureStore backed by an in-memory FakeRedis — no real Redis needed."""
    store = FeatureStore.__new__(FeatureStore)
    store._tick_buffer = deque(maxlen=7200)
    store._redis = fakeredis.FakeRedis()
    return store


# ── _resolution_estimate ───────────────────────────────────────────────────

def test_resolution_estimate_averages_last_60s_only():
    store = make_store()
    now = time.time()
    for i in range(60):
        store._tick_buffer.append((now - 120 + i, 100.0))  # old (>60s ago)
    for i in range(60):
        store._tick_buffer.append((now - 59 + i, 200.0))   # fresh (≤60s ago)
    assert store._resolution_estimate() == pytest.approx(200.0)


def test_resolution_estimate_returns_none_when_buffer_empty():
    store = make_store()
    assert store._resolution_estimate() is None


def test_resolution_estimate_includes_all_ticks_within_60s():
    store = make_store()
    now = time.time()
    store._tick_buffer.extend([
        (now - 59, 100.0),
        (now - 30, 200.0),
        (now - 1,  300.0),
    ])
    assert store._resolution_estimate() == pytest.approx(200.0)  # mean(100, 200, 300)


# ── _resample ──────────────────────────────────────────────────────────────

def _populate_two_candles(store: FeatureStore) -> None:
    """Fill buffer with 10 ticks (1/min) forming 2 complete 5-min candles."""
    base_ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp()
    prices = [100.0, 101.0, 99.0, 102.0, 103.0,   # candle 1: min 0-4
              200.0, 201.0, 199.0, 202.0, 203.0]  # candle 2: min 5-9
    for i, price in enumerate(prices):
        store._tick_buffer.append((base_ts + i * 60, price))


def test_resample_returns_correct_columns():
    store = make_store()
    _populate_two_candles(store)
    df = store._resample("5min")
    assert df is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]


def test_resample_ohlcv_values_candle_1():
    store = make_store()
    _populate_two_candles(store)
    df = store._resample("5min")
    assert df is not None
    assert len(df) == 2
    c1 = df.iloc[0]
    assert c1["open"]   == pytest.approx(100.0)
    assert c1["high"]   == pytest.approx(103.0)
    assert c1["low"]    == pytest.approx(99.0)
    assert c1["close"]  == pytest.approx(103.0)
    assert c1["volume"] == pytest.approx(0.0)
    assert c1["amount"] == pytest.approx(0.0)


def test_resample_returns_none_when_fewer_than_2_ticks():
    store = make_store()
    store._tick_buffer.append((time.time(), 100.0))
    assert store._resample("5min") is None


def test_resample_returns_none_for_empty_buffer():
    store = make_store()
    assert store._resample("5min") is None


# ── get_raw_ticks ──────────────────────────────────────────────────────────

def test_get_raw_ticks_returns_ticks_within_window():
    store = make_store()
    now = time.time()
    store._tick_buffer.extend([
        (now - 120, 100.0),  # old — excluded
        (now - 30,  200.0),  # within 60s
        (now - 1,   300.0),  # within 60s
    ])
    result = store.get_raw_ticks(60)
    assert result is not None
    assert len(result) == 2
    assert list(result.values) == pytest.approx([200.0, 300.0])


def test_get_raw_ticks_returns_none_when_no_data_in_window():
    store = make_store()
    store._tick_buffer.append((time.time() - 120, 100.0))
    assert store.get_raw_ticks(60) is None


# ── _flush_to_redis → read API ─────────────────────────────────────────────

def test_flush_writes_resolution_estimate_to_redis():
    store = make_store()
    now = time.time()
    for i in range(60):
        store._tick_buffer.append((now - 59 + i, 150.0))
    store._flush_to_redis()
    assert store.get_resolution_estimate() == pytest.approx(150.0, abs=1.0)


def test_flush_writes_ohlcv_to_redis():
    store = make_store()
    _populate_two_candles(store)
    store._flush_to_redis()
    df = store.get_ohlcv("5min")
    assert df is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert len(df) >= 1


def test_get_ohlcv_preserves_utc_index():
    store = make_store()
    _populate_two_candles(store)
    store._flush_to_redis()
    df = store.get_ohlcv("5min")
    assert df is not None
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_get_resolution_estimate_returns_none_before_flush():
    store = make_store()
    assert store.get_resolution_estimate() is None


def test_get_ohlcv_returns_none_before_flush():
    store = make_store()
    assert store.get_ohlcv("5min") is None
