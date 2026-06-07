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


def test_parse_okx_message_single_tick():
    import json
    acc = StreamingCVDAccumulator()
    msg = json.dumps({
        "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
        "data": [{"px": "95000.5", "sz": "0.25", "side": "buy", "ts": "1700000000000"}]
    })
    ticks = acc._parse_okx_message(msg)
    assert len(ticks) == 1
    ts_ms, side, size, price = ticks[0]
    assert ts_ms == 1700000000000
    assert side == "buy"
    assert size == pytest.approx(0.25)
    assert price == pytest.approx(95000.5)


def test_parse_okx_message_ignores_non_trade_events():
    import json
    acc = StreamingCVDAccumulator()
    msg = json.dumps({"event": "subscribe", "arg": {"channel": "trades"}})
    ticks = acc._parse_okx_message(msg)
    assert ticks == []


def test_parse_kraken_message_single_tick():
    import json
    acc = StreamingCVDAccumulator()
    msg = json.dumps({
        "channel": "trade",
        "type": "update",
        "data": [{"side": "sell", "qty": 0.1, "price": 94500.0,
                  "timestamp": "2023-11-14T12:00:00.000000Z"}]
    })
    ticks = acc._parse_kraken_message(msg)
    assert len(ticks) == 1
    ts_ms, side, size, price = ticks[0]
    assert side == "sell"
    assert size == pytest.approx(0.1)
    assert price == pytest.approx(94500.0)
    assert ts_ms > 0


def test_parse_kraken_message_ignores_non_update_events():
    import json
    acc = StreamingCVDAccumulator()
    msg = json.dumps({"channel": "trade", "type": "snapshot", "data": []})
    ticks = acc._parse_kraken_message(msg)
    assert ticks == []
