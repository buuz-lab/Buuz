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
    direction: int = 1,
    regime_features: dict | None = None,
) -> TradingSignal:
    return TradingSignal(
        direction=direction,
        calibrated_prob=calibrated_prob,
        kronos_raw=calibrated_prob,
        kronos_calibrated=calibrated_prob,
        regime_prob=calibrated_prob,
        regime_direction=direction,
        deepseek_regime=deepseek_regime,
        timeframe="5min",
        strike=strike,
        timestamp=datetime.now(timezone.utc),
        regime_features=regime_features or {},
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


# ── Gate 2: practical Kelly 1-contract floor ─────────────────────────────────

def test_gate2_passes_with_1_contract_floor_at_boundary(checklist):
    """Kelly=$0.23 on a 45¢ market (>= half of $0.45) → floor to 1 contract, passes."""
    kw = base_kwargs()
    kw["best_ask_cents"] = 45
    with patch.object(checklist._kelly, "compute_size", return_value=0.23), \
         patch.object(checklist._kelly, "dollars_to_contracts", return_value=0):
        r = checklist.run(**kw)
    assert r.passed
    assert r.kelly_contracts == 1


def test_gate2_fails_below_half_contract_cost(checklist):
    """Kelly=$0.22 on a 45¢ market (< half of $0.45) → still fails Gate 2."""
    kw = base_kwargs()
    kw["best_ask_cents"] = 45
    with patch.object(checklist._kelly, "compute_size", return_value=0.22), \
         patch.object(checklist._kelly, "dollars_to_contracts", return_value=0):
        r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 2
    assert "rounds to 0" in r.failed_reason


# ── Gate 8 tests ─────────────────────────────────────────────────────────────

def test_gate8_blocks_no_down_when_kalshi_mid_high(checklist):
    """kalshi_mid=0.60 → opposing=0.10 > threshold=0.08 → Gate 8 blocks NO→DOWN."""
    signal = make_signal(direction=0, calibrated_prob=0.35)
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.60
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8
    assert r.kalshi_mid_at_block == pytest.approx(0.60)


def test_gate8_passes_no_down_when_kalshi_mid_close(checklist):
    """kalshi_mid=0.55 → opposing=0.05 < threshold=0.08 → Gate 8 passes."""
    signal = make_signal(direction=0, calibrated_prob=0.35)
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.55
    r = checklist.run(**kw)
    assert r.failed_gate != 8


def test_gate8_blocks_yes_up_when_kalshi_mid_low(checklist):
    """kalshi_mid=0.40 → opposing=0.10 > threshold=0.08 → Gate 8 blocks YES→UP."""
    signal = make_signal(direction=1, calibrated_prob=0.65)
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.40
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8


def test_gate8_oi_squeeze_compound(checklist):
    """OI squeeze: oi_delta_pct=0.002 AND NO→DOWN → effective_threshold=0.02. kalshi_mid=0.53 → opposing=0.03 > 0.02."""
    signal = make_signal(direction=0, calibrated_prob=0.35, regime_features={"oi_delta_pct": 0.002})
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.53
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8


def test_gate8_kelly_multiplier_reduces_dollars(checklist):
    """kalshi_mid=0.60 for NO bet: opposing=0.10, mult=1-0.10/0.20=0.50 → kelly_dollars*0.50."""
    signal = make_signal(direction=0, calibrated_prob=0.35)
    kw = base_kwargs(signal)
    # Use a kalshi_mid that passes the hard gate but triggers the multiplier
    kw["fresh_kalshi_mid"] = 0.55  # opposing=0.05, mult=1-0.05/0.20=0.75 < 1.0; passes hard gate (0.05 < 0.08)
    r_no_mult = checklist.run(**{**base_kwargs(signal), "fresh_kalshi_mid": 0.50})
    r_with_mult = checklist.run(**kw)
    # With opposing margin, kelly_dollars should be less
    assert r_with_mult.kelly_dollars < r_no_mult.kelly_dollars


def test_gate8_drift_shrink_halves_kelly(checklist):
    """is_drifting=True → kelly_dollars halved (before contract rounding)."""
    signal = make_signal(direction=1, calibrated_prob=0.65)
    r_no_drift = checklist.run(**base_kwargs(signal))
    kw = base_kwargs(signal)
    kw["is_drifting"] = True
    r_drifting = checklist.run(**kw)
    assert r_drifting.kelly_dollars < r_no_drift.kelly_dollars


def test_direction_win_rate_passed_to_kelly(checklist):
    """direction_win_rate param flows through checklist to kelly.compute_size."""
    signal = make_signal(direction=1, calibrated_prob=0.65)
    r_no_wr = checklist.run(**base_kwargs(signal))
    kw = base_kwargs(signal)
    kw["direction_win_rate"] = 0.40  # below 0.45 threshold → 40% shrink
    r_with_wr = checklist.run(**kw)
    assert r_with_wr.kelly_dollars < r_no_wr.kelly_dollars


def test_gate8_both_shrinks_stack(checklist):
    """Kalshi mult AND drift shrink both active → kelly_dollars = base * mult * 0.5."""
    signal = make_signal(direction=0, calibrated_prob=0.35)
    r_base = checklist.run(**base_kwargs(signal))
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.55  # opposing=0.05, mult=0.75
    kw["is_drifting"] = True
    r_both = checklist.run(**kw)
    expected = r_base.kelly_dollars * 0.75 * 0.5
    assert r_both.kelly_dollars == pytest.approx(expected, rel=0.01)


# ── Bootstrap floor tests ─────────────────────────────────────────────────────

def test_bootstrap_floor_allows_1_contract_on_thin_edge(checklist):
    """is_bootstrap=True + positive edge + price 25-75¢ → 1 contract instead of gate 2 fail.

    prob=0.507, ask=bid=50¢ (zero spread), plus chop+tape+direction_win_rate shrinks stack
    kelly_dollars to ~0.176 — below the 0.5x heuristic (0.25) but still positive.
    Gate 5 passes (edge=0.007 > 0.005). Bootstrap floor gives 1 contract.
    """
    # chop shrink (×0.70) + tape shrink (×0.80) + direction_win_rate (×0.60)
    # → kelly_dollars ≈ 75 * 0.007 * 0.70 * 0.80 * 0.60 = 0.176 < 0.25 (0.5×price)
    rf = {"range_breakout_flag": 0.10, "tape_speed_tpm": 0.10}
    signal = make_signal(direction=1, calibrated_prob=0.507, regime_features=rf)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 50
    kw["direction_win_rate"] = 0.40

    r_normal = checklist.run(**kw)
    assert not r_normal.passed and r_normal.failed_gate == 2

    kw["is_bootstrap"] = True
    r_bootstrap = checklist.run(**kw)
    assert r_bootstrap.passed
    assert r_bootstrap.kelly_contracts == 1


def test_bootstrap_floor_blocked_outside_price_range(checklist):
    """is_bootstrap=True but NO trade_price > 75¢ → still fails gate 2 (bad risk/reward).
    Mock kelly_dollars=0.35 < 0.40 (0.5×80¢ threshold) so heuristic also fails."""
    signal = make_signal(direction=0, calibrated_prob=0.193)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 20
    kw["best_bid_cents"] = 20  # NO costs 100-20=80¢ > 75¢
    kw["is_bootstrap"] = True
    with patch.object(checklist._kelly, "compute_size", return_value=0.35), \
         patch.object(checklist._kelly, "dollars_to_contracts", return_value=0):
        r = checklist.run(**kw)
    assert not r.passed and r.failed_gate == 2


def test_bootstrap_floor_not_active_when_regime_trained(checklist):
    """is_bootstrap=False (regime trained) → thin-edge trade still fails gate 2."""
    rf = {"range_breakout_flag": 0.10, "tape_speed_tpm": 0.10}
    signal = make_signal(direction=1, calibrated_prob=0.507, regime_features=rf)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 50
    kw["direction_win_rate"] = 0.40
    kw["is_bootstrap"] = False
    r = checklist.run(**kw)
    assert not r.passed and r.failed_gate == 2


# ── Gate 2a: minimum price filter ────────────────────────────────────────────

def test_min_price_blocks_yes_at_low_cents(checklist):
    """YES trade at 9¢ ask is rejected before Kelly runs (extreme/illiquid market)."""
    signal = make_signal(direction=1, calibrated_prob=0.52)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 9
    kw["best_bid_cents"] = 7
    r = checklist.run(**kw)
    assert not r.passed and r.failed_gate == 2
    assert "below minimum" in r.failed_reason

def test_min_price_blocks_no_at_low_cents(checklist):
    """NO trade where 100-bid=15¢ is also rejected (direction=0, bid=85)."""
    signal = make_signal(direction=0, calibrated_prob=0.52)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 87
    kw["best_bid_cents"] = 85   # NO price = 100-85 = 15¢
    r = checklist.run(**kw)
    assert not r.passed and r.failed_gate == 2
    assert "below minimum" in r.failed_reason

def test_min_price_allows_trade_at_boundary(checklist):
    """Trade at exactly 20¢ is allowed through the price filter."""
    signal = make_signal(direction=1, calibrated_prob=0.65)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 20
    kw["best_bid_cents"] = 18
    kw["available_contracts"] = 200  # Kelly at 20¢ requests ~105 contracts
    r = checklist.run(**kw)
    assert r.passed
