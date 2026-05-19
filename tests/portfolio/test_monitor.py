import time

import fakeredis
import pytest

from btc_kalshi_system.portfolio.monitor import (
    DAILY_PNL_DATE_KEY,
    DAILY_PNL_KEY,
    OpenPosition,
    PortfolioMonitor,
    ResolvedTrade,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_monitor() -> PortfolioMonitor:
    mon = PortfolioMonitor.__new__(PortfolioMonitor)
    mon._redis = fakeredis.FakeRedis(decode_responses=True)
    mon._positions = {}
    mon._load_state()
    return mon


def make_position(
    trade_id: str = "trade-1",
    timeframe: str = "same_day",
    kelly_dollars: float = 20.0,
    entry_price_cents: int = 50,
    contracts: int = 4,
) -> OpenPosition:
    return OpenPosition(
        trade_id=trade_id,
        ticker="KXBTC-25JUN-T95000",
        timeframe=timeframe,
        direction=1,
        strike=95000.0,
        contracts=contracts,
        entry_price_cents=entry_price_cents,
        kelly_dollars=kelly_dollars,
        timestamp=time.time(),
    )


# ------------------------------------------------------------------
# add_position / get_open_positions
# ------------------------------------------------------------------

def test_add_position_visible_in_get_open_positions():
    mon = make_monitor()
    pos = make_position()
    mon.add_position(pos)
    positions = mon.get_open_positions()
    assert len(positions) == 1
    assert positions[0].trade_id == "trade-1"


def test_add_position_persists_to_redis():
    mon = make_monitor()
    mon.add_position(make_position("trade-1"))
    # Reload from same redis — should see position
    mon2 = PortfolioMonitor.__new__(PortfolioMonitor)
    mon2._redis = mon._redis
    mon2._positions = {}
    mon2._load_state()
    assert any(p.trade_id == "trade-1" for p in mon2.get_open_positions())


# ------------------------------------------------------------------
# get_current_exposure
# ------------------------------------------------------------------

def test_get_current_exposure_sums_kelly_dollars():
    mon = make_monitor()
    mon.add_position(make_position("t1", kelly_dollars=15.0))
    mon.add_position(make_position("t2", kelly_dollars=25.0))
    assert mon.get_current_exposure() == pytest.approx(40.0)


def test_get_current_exposure_zero_when_no_positions():
    mon = make_monitor()
    assert mon.get_current_exposure() == 0.0


# ------------------------------------------------------------------
# has_timeframe_position
# ------------------------------------------------------------------

def test_has_timeframe_position_true():
    mon = make_monitor()
    mon.add_position(make_position(timeframe="1h"))
    assert mon.has_timeframe_position("1h") is True


def test_has_timeframe_position_false():
    mon = make_monitor()
    mon.add_position(make_position(timeframe="1h"))
    assert mon.has_timeframe_position("4h") is False


def test_has_timeframe_position_false_when_empty():
    mon = make_monitor()
    assert mon.has_timeframe_position("same_day") is False


# ------------------------------------------------------------------
# resolve_trade — PnL
# ------------------------------------------------------------------

def test_resolve_trade_win_pnl():
    mon = make_monitor()
    # 4 contracts at 50 cents → win pays (1 - 0.50) * 4 = $2.00
    mon.add_position(make_position(entry_price_cents=50, contracts=4))
    trade = mon.resolve_trade("trade-1", outcome=1)
    assert trade is not None
    assert trade.pnl_dollars == pytest.approx(2.00)
    assert trade.outcome == 1


def test_resolve_trade_loss_pnl():
    mon = make_monitor()
    # 4 contracts at 50 cents → loss costs 0.50 * 4 = $2.00
    mon.add_position(make_position(entry_price_cents=50, contracts=4))
    trade = mon.resolve_trade("trade-1", outcome=0)
    assert trade is not None
    assert trade.pnl_dollars == pytest.approx(-2.00)
    assert trade.outcome == 0


def test_resolve_trade_removes_from_open_positions():
    mon = make_monitor()
    mon.add_position(make_position())
    mon.resolve_trade("trade-1", outcome=1)
    assert len(mon.get_open_positions()) == 0


def test_resolve_trade_returns_none_for_unknown_id():
    mon = make_monitor()
    result = mon.resolve_trade("nonexistent", outcome=1)
    assert result is None


def test_resolve_trade_sets_resolved_at():
    mon = make_monitor()
    mon.add_position(make_position())
    ts = 1_700_000_000.0
    trade = mon.resolve_trade("trade-1", outcome=1, resolved_at=ts)
    assert trade.resolved_at == ts


# ------------------------------------------------------------------
# get_daily_pnl — reset on date change
# ------------------------------------------------------------------

def test_get_daily_pnl_resets_on_date_change():
    mon = make_monitor()
    # Write a fake yesterday date with some PnL
    mon._redis.set(DAILY_PNL_KEY, "99.99")
    mon._redis.set(DAILY_PNL_DATE_KEY, "2000-01-01")
    pnl = mon.get_daily_pnl()
    assert pnl == pytest.approx(0.0)


def test_get_daily_pnl_accumulates_from_resolve():
    mon = make_monitor()
    mon.add_position(make_position("t1", entry_price_cents=50, contracts=4))
    mon.add_position(make_position("t2", entry_price_cents=50, contracts=2))
    mon.resolve_trade("t1", outcome=1)  # +$2.00
    mon.resolve_trade("t2", outcome=0)  # -$1.00
    assert mon.get_daily_pnl() == pytest.approx(1.00)


# ------------------------------------------------------------------
# get_resolved_trades / get_trade_count
# ------------------------------------------------------------------

def test_get_resolved_trades_respects_limit():
    mon = make_monitor()
    for i in range(5):
        mon.add_position(make_position(f"t{i}"))
        mon.resolve_trade(f"t{i}", outcome=1)

    trades = mon.get_resolved_trades(limit=3)
    assert len(trades) == 3


def test_get_resolved_trades_newest_first():
    mon = make_monitor()
    mon.add_position(make_position("first"))
    mon.resolve_trade("first", outcome=1)
    mon.add_position(make_position("second"))
    mon.resolve_trade("second", outcome=1)

    trades = mon.get_resolved_trades(limit=2)
    assert trades[0].trade_id == "second"
    assert trades[1].trade_id == "first"


def test_get_trade_count():
    mon = make_monitor()
    for i in range(3):
        mon.add_position(make_position(f"t{i}"))
        mon.resolve_trade(f"t{i}", outcome=1)
    assert mon.get_trade_count() == 3


# ------------------------------------------------------------------
# State survives restart (new instance, same Redis)
# ------------------------------------------------------------------

def test_state_survives_restart():
    mon = make_monitor()
    mon.add_position(make_position("trade-persist", timeframe="4h", kelly_dollars=30.0))

    # Simulate restart: new instance sharing same Redis
    mon2 = PortfolioMonitor.__new__(PortfolioMonitor)
    mon2._redis = mon._redis
    mon2._positions = {}
    mon2._load_state()

    assert mon2.has_timeframe_position("4h")
    assert mon2.get_current_exposure() == pytest.approx(30.0)
    positions = mon2.get_open_positions()
    assert len(positions) == 1
    assert positions[0].trade_id == "trade-persist"
