import time
from collections import deque

import fakeredis
import pytest

from btc_kalshi_system.data.feature_store import FeatureStore
from btc_kalshi_system.models.kronos_engine import KronosEngine


def make_store_with_candles(n_candles: int = 420) -> FeatureStore:
    """FeatureStore with synthetic 5-min candles: linear price ramp 100→102."""
    store = FeatureStore.__new__(FeatureStore)
    store._tick_buffer = deque(maxlen=7200)
    store._redis = fakeredis.FakeRedis()
    # 5 ticks per candle at 60s intervals so _resample produces clean 5-min OHLCV
    base_ts = time.time() - n_candles * 300
    for i in range(n_candles * 5):
        price = 100.0 + (i / (n_candles * 5)) * 2.0
        store._tick_buffer.append((base_ts + i * 60, price))
    store._flush_to_redis()
    return store


def test_kronos_engine_raises_when_insufficient_data():
    store = make_store_with_candles(3)
    engine = KronosEngine()
    with pytest.raises(ValueError, match="Insufficient OHLCV data"):
        engine.run_monte_carlo(store, n_paths=5, threshold=101.0)
