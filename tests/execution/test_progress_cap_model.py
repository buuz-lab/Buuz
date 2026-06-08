from btc_kalshi_system.execution.pretrade_checklist import RuleBasedProgressCap


def test_quiet_market_cap_is_10_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.001, spread=0.02, volume_ratio=1.0)
    assert result == 0.10


def test_high_vol_wide_spread_requires_5_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.005, spread=0.06, volume_ratio=1.0)
    assert result == 0.05


def test_high_vol_tight_spread_gives_10_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.005, spread=0.02, volume_ratio=1.0)
    assert result == 0.10


def test_low_vol_wide_spread_gives_10_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.001, spread=0.06, volume_ratio=1.0)
    assert result == 0.10


def test_boundary_volatility_not_high_vol():
    """Exactly at the threshold is not high vol (> not >=)."""
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.003, spread=0.06, volume_ratio=1.0)
    assert result == 0.10  # wide spread only → 0.10, not 0.05


def test_returns_float():
    cap = RuleBasedProgressCap()
    assert isinstance(cap.get_cap(0.001, 0.02, 1.0), float)
