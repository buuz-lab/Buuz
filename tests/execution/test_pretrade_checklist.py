import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from btc_kalshi_system.execution.kelly import KellySizer
from btc_kalshi_system.execution.pretrade_checklist import PreTradeChecklist
from btc_kalshi_system.signal.fusion import TradingSignal


def make_signal(
    calibrated_prob: float = 0.65,
    deepseek_regime: str = "neutral",
    strike: float = 95000.0,
) -> TradingSignal:
    return TradingSignal(
        direction=1,
        calibrated_prob=calibrated_prob,
        kronos_raw=calibrated_prob,
        kronos_calibrated=calibrated_prob,
        regime_prob=calibrated_prob,
        regime_direction=1,
        deepseek_regime=deepseek_regime,
        timeframe="5min",
        strike=strike,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def checklist():
    with patch("btc_kalshi_system.execution.pretrade_checklist.redis") as mock_redis:
        mock_client = MagicMock()
        mock_client.get.return_value = None  # loss_streak = 0
        mock_redis.from_url.return_value = mock_client
        yield PreTradeChecklist(KellySizer())


def base_kwargs(signal: TradingSignal | None = None) -> dict:
    """Passing kwargs that clear all 6 gates."""
    return dict(
        signal=signal or make_signal(),
        best_ask_cents=50,
        best_bid_cents=48,        # spread = $0.02, under limit
        available_contracts=100,
        current_exposure=0.0,
        same_timeframe_open=False,
        composite_price=96000.0,  # distance from strike=95000 → $1000, >= 150
        edge_above_threshold=True,
    )


# ── Gate failures ───────────────────────────────────────────────────────────

def test_gate1_spread_too_wide(checklist):
    kw = base_kwargs()
    kw["best_bid_cents"] = 45  # spread = $0.05
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 1
    assert r.kelly_dollars == 0.0
    assert r.kelly_contracts == 0


def test_gate2_kelly_rounds_to_zero(checklist):
    # Near-zero edge → Kelly produces 0 contracts
    kw = base_kwargs(make_signal(calibrated_prob=0.501))
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 49
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 2
    assert r.kelly_dollars == 0.0
    assert r.kelly_contracts == 0


def test_gate2_insufficient_depth(checklist):
    kw = base_kwargs(make_signal(calibrated_prob=0.99))
    kw["best_ask_cents"] = 1   # very cheap → many contracts needed
    kw["best_bid_cents"] = 0   # spread = $0.00, passes gate 1
    kw["available_contracts"] = 1
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 2


def test_gate3_high_uncertainty_thin_edge(checklist):
    signal = make_signal(calibrated_prob=0.52, deepseek_regime="high_uncertainty")
    kw = base_kwargs(signal)
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 3
    assert r.kelly_dollars == 0.0
    assert r.kelly_contracts == 0


def test_gate4_edge_below_threshold(checklist):
    kw = base_kwargs()
    kw["edge_above_threshold"] = False
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 4
    assert r.kelly_dollars == 0.0
    assert r.kelly_contracts == 0


def test_gate5_signal_edge_too_small(checklist):
    # calibrated_prob=0.52, ask=50 cents → signal_edge=0.02
    # spread=0.02, min_required=0.025 → 0.02 <= 0.025 → fail
    signal = make_signal(calibrated_prob=0.52)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48  # spread=$0.02, min_required=$0.025
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 5
    assert r.kelly_dollars == 0.0
    assert r.kelly_contracts == 0


def test_gate6_too_close_to_strike(checklist):
    signal = make_signal(strike=95000.0)
    kw = base_kwargs(signal)
    kw["composite_price"] = 95100.0  # distance = $100 < $150
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 6
    assert r.kelly_dollars == 0.0
    assert r.kelly_contracts == 0


# ── All gates pass ───────────────────────────────────────────────────────────

def test_all_gates_pass(checklist):
    r = checklist.run(**base_kwargs())
    assert r.passed
    assert r.failed_gate is None
    assert r.failed_reason is None
    assert r.kelly_dollars > 0.0
    assert r.kelly_contracts > 0


def test_passing_result_has_correct_kelly_values(checklist):
    signal = make_signal(calibrated_prob=0.65)
    kw = base_kwargs(signal)
    r = checklist.run(**kw)
    sizer = KellySizer()
    expected_dollars = sizer.compute_size(
        prob=0.65,
        market_price=kw["best_ask_cents"] / 100,
        current_exposure=kw["current_exposure"],
        same_timeframe_open=kw["same_timeframe_open"],
    )
    expected_contracts = sizer.dollars_to_contracts(expected_dollars, kw["best_ask_cents"])
    assert math.isclose(r.kelly_dollars, expected_dollars, rel_tol=1e-9)
    assert r.kelly_contracts == expected_contracts


# ── Gate 3 boundary: thick edge should NOT trigger gate 3 ───────────────────

def test_gate3_does_not_fire_on_thick_edge(checklist):
    signal = make_signal(calibrated_prob=0.60, deepseek_regime="high_uncertainty")
    kw = base_kwargs(signal)
    r = checklist.run(**kw)
    # Gate 3 should not fire (edge_from_center = 0.10 >= 0.05)
    assert r.failed_gate != 3


# ── Gate 6 boundary: exactly 150 should PASS ────────────────────────────────

def test_gate6_boundary_exactly_150_passes(checklist):
    signal = make_signal(strike=95000.0)
    kw = base_kwargs(signal)
    kw["composite_price"] = 95150.0  # distance exactly 150 → should pass
    r = checklist.run(**kw)
    assert r.failed_gate != 6


def test_gate6_boundary_149_fails(checklist):
    signal = make_signal(strike=95000.0)
    kw = base_kwargs(signal)
    kw["composite_price"] = 95149.0  # distance = 149 → should fail
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 6
