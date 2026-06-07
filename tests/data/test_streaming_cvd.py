import json
import time
import pytest
from btc_kalshi_system.data.derivatives_feed import StreamingCVDAccumulator


def _tick(side: str, size: float, price: float, age_s: float = 0.0):
    """Helper: build a (ts_ms, side, size, price) tuple."""
    ts_ms = int((time.time() - age_s) * 1000)
    return (ts_ms, side, size, price)


def test_cvd_all_buys():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._ingest_tick(_tick("buy", 2.0, 95001.0))
    assert acc.cvd_normalized == pytest.approx(1.0)


def test_cvd_equal_buys_sells():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    assert acc.cvd_normalized == pytest.approx(0.0)


def test_cvd_mixed():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 3.0, 95000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    # (3-1)/(3+1) = 0.5
    assert acc.cvd_normalized == pytest.approx(0.5)


def test_old_ticks_pruned():
    """Tick older than 15 min is dropped; only newer tick counts."""
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 5.0, 95000.0, age_s=960))  # 16 min old
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    assert len(acc._trades) == 1
    assert acc.cvd_normalized == pytest.approx(-1.0)


def test_is_stale_with_no_ticks():
    acc = StreamingCVDAccumulator()
    assert acc.is_stale is True


def test_is_stale_clears_after_5_ticks():
    acc = StreamingCVDAccumulator()
    for _ in range(5):
        acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    assert acc.is_stale is False


def test_is_stale_after_silence():
    acc = StreamingCVDAccumulator()
    for _ in range(5):
        acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._last_tick_at = time.time() - 121  # backdate to simulate silence
    assert acc.is_stale is True


def test_last_price_tracks_most_recent_tick():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 94000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    assert acc.last_price == pytest.approx(95000.0)


def test_large_print_direction_buy_dominated():
    acc = StreamingCVDAccumulator()
    # avg_size = (4*1 + 6)/5 = 2.0, threshold = 4.0, only the buy-6 is large
    for _ in range(4):
        acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    acc._ingest_tick(_tick("buy", 6.0, 95000.0))
    assert acc.large_print_direction == pytest.approx(1.0)


def test_large_print_direction_no_large_trades():
    acc = StreamingCVDAccumulator()
    acc._ingest_tick(_tick("buy", 1.0, 95000.0))
    acc._ingest_tick(_tick("sell", 1.0, 95000.0))
    # avg=1.0, threshold=2.0, no trades exceed it → 0.0
    assert acc.large_print_direction == pytest.approx(0.0)


def test_cvd_empty_deque():
    acc = StreamingCVDAccumulator()
    assert acc.cvd_normalized == 0.0
    assert acc.large_print_direction == 0.0
    assert acc.last_price == 0.0
