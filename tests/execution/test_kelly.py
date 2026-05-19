import math
import pytest
from btc_kalshi_system.execution.kelly import (
    KellySizer,
    KELLY_FRACTION,
    MAX_SINGLE_TRADE_DOLLARS,
    MAX_TOTAL_EXPOSURE_DOLLARS,
    CORRELATION_DISCOUNT,
)


@pytest.fixture
def sizer():
    return KellySizer()


# --- compute_size ---

def test_zero_when_no_edge(sizer):
    assert sizer.compute_size(prob=0.50, market_price=0.50, current_exposure=0.0, same_timeframe_open=False) == 0.0


def test_zero_when_negative_edge(sizer):
    assert sizer.compute_size(prob=0.40, market_price=0.55, current_exposure=0.0, same_timeframe_open=False) == 0.0


def test_zero_when_at_max_exposure(sizer):
    assert sizer.compute_size(prob=0.70, market_price=0.50, current_exposure=150.0, same_timeframe_open=False) == 0.0


def test_zero_when_over_max_exposure(sizer):
    assert sizer.compute_size(prob=0.70, market_price=0.50, current_exposure=200.0, same_timeframe_open=False) == 0.0


def test_correlation_discount_reduces_size(sizer):
    without = sizer.compute_size(prob=0.65, market_price=0.50, current_exposure=0.0, same_timeframe_open=False)
    with_ = sizer.compute_size(prob=0.65, market_price=0.50, current_exposure=0.0, same_timeframe_open=True)
    assert with_ < without
    assert math.isclose(with_, without * CORRELATION_DISCOUNT, rel_tol=1e-9)


def test_hard_cap_single_trade(sizer):
    # Large edge, zero exposure — should hit MAX_SINGLE_TRADE_DOLLARS
    size = sizer.compute_size(prob=0.99, market_price=0.01, current_exposure=0.0, same_timeframe_open=False)
    assert size <= MAX_SINGLE_TRADE_DOLLARS


def test_hard_cap_remaining_capacity(sizer):
    # Only $10 headroom left
    size = sizer.compute_size(prob=0.99, market_price=0.01, current_exposure=140.0, same_timeframe_open=False)
    assert size <= 10.0


def test_positive_size_with_edge(sizer):
    size = sizer.compute_size(prob=0.65, market_price=0.50, current_exposure=0.0, same_timeframe_open=False)
    assert size > 0.0


def test_size_never_negative(sizer):
    # Remaining capacity nearly zero
    size = sizer.compute_size(prob=0.60, market_price=0.50, current_exposure=149.99, same_timeframe_open=False)
    assert size >= 0.0


def test_kelly_formula_correctness(sizer):
    prob, price = 0.60, 0.50
    edge = prob - price
    full_kelly = edge / (1 - price)
    expected = full_kelly * KELLY_FRACTION * MAX_TOTAL_EXPOSURE_DOLLARS
    size = sizer.compute_size(prob=prob, market_price=price, current_exposure=0.0, same_timeframe_open=False)
    assert math.isclose(size, min(expected, MAX_SINGLE_TRADE_DOLLARS), rel_tol=1e-9)


# --- dollars_to_contracts ---

def test_contracts_basic_math(sizer):
    # $10 at 50 cents per contract → 20 contracts
    assert sizer.dollars_to_contracts(10.0, 50) == 20


def test_contracts_floors_result(sizer):
    # $10 at 30 cents → 33.33 → floor to 33
    assert sizer.dollars_to_contracts(10.0, 30) == 33


def test_contracts_zero_on_zero_dollars(sizer):
    assert sizer.dollars_to_contracts(0.0, 50) == 0


def test_contracts_zero_on_negative_dollars(sizer):
    assert sizer.dollars_to_contracts(-5.0, 50) == 0


def test_contracts_zero_on_zero_price(sizer):
    assert sizer.dollars_to_contracts(10.0, 0) == 0


def test_contracts_zero_on_negative_price(sizer):
    assert sizer.dollars_to_contracts(10.0, -10) == 0
