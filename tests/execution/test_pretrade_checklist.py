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


def test_gate2_depth_capped_to_available(checklist):
    """Kelly wants N contracts but only M available (M>0) → trade with M, don't block.

    Regression: previously hard-failed on depth, leaving edge on the table when
    the orderbook had fewer contracts than Kelly requested (e.g. 10 available, 27 wanted).
    Uses 50¢ fill (above Gate 11's 45¢ floor) so Gate 11 does not fire.
    """
    signal = make_signal(direction=1, calibrated_prob=0.85)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 49
    kw["available_contracts"] = 5   # Kelly will want far more than 5 at 85%/50¢
    r = checklist.run(**kw)
    assert r.passed
    assert r.kelly_contracts == 5
    assert r.kelly_dollars == pytest.approx(5 * 0.50, rel=0.01)


def test_gate2_zero_depth_still_fails(checklist):
    """available_contracts=0 → still fails Gate 2 (nothing to buy)."""
    signal = make_signal(direction=1, calibrated_prob=0.85)
    kw = base_kwargs(signal)
    kw["available_contracts"] = 0
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
    """kalshi_mid=0.77 → opposing=0.27 > threshold=0.25 → Gate 8 blocks NO→DOWN.
    opposing=0.27 < Gate8b zero-point (0.30) so Kelly survives the multiplier and Gate 8 fires."""
    signal = make_signal(direction=0, calibrated_prob=0.35)
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.77
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8
    assert r.kalshi_mid_at_block == pytest.approx(0.77)


def test_gate8_passes_no_down_when_kalshi_mid_close(checklist):
    """kalshi_mid=0.55 → opposing=0.05 < threshold=0.25 → Gate 8 passes."""
    signal = make_signal(direction=0, calibrated_prob=0.35)
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.55
    r = checklist.run(**kw)
    assert r.failed_gate != 8


def test_gate8_blocks_yes_up_when_kalshi_mid_low(checklist):
    """kalshi_mid=0.23 → opposing=0.27 > threshold=0.25 → Gate 8 blocks YES→UP.
    opposing=0.27 < Gate8b zero-point (0.30) so Kelly survives the multiplier and Gate 8 fires."""
    signal = make_signal(direction=1, calibrated_prob=0.65)
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.23
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8


def test_gate8_oi_squeeze_compound(checklist):
    """OI squeeze: oi_delta_pct=0.002 AND NO→DOWN → effective_threshold=0.0625. kalshi_mid=0.57 → opposing=0.07 > 0.0625."""
    signal = make_signal(direction=0, calibrated_prob=0.35, regime_features={"oi_delta_pct": 0.002})
    kw = base_kwargs(signal)
    kw["fresh_kalshi_mid"] = 0.57
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8


def test_gate8_kelly_multiplier_reduces_dollars(checklist):
    """kalshi_mid=0.55 for NO bet: opposing=0.05, mult=1-0.05/0.30≈0.83 → kelly_dollars reduced."""
    signal = make_signal(direction=0, calibrated_prob=0.35)
    kw = base_kwargs(signal)
    # Use a kalshi_mid that passes the hard gate but triggers the multiplier
    kw["fresh_kalshi_mid"] = 0.55  # opposing=0.05, mult≈0.83; passes hard gate (0.05 < 0.25)
    r_no_mult = checklist.run(**{**base_kwargs(signal), "fresh_kalshi_mid": 0.50})
    r_with_mult = checklist.run(**kw)
    # With opposing margin, kelly_dollars should be less
    assert r_with_mult.kelly_dollars < r_no_mult.kelly_dollars


def test_high_confidence_k15_passes_when_kalshi_disagrees_moderately(checklist):
    """k15=0.89 YES@50¢, Kalshi=0.29 → opposing=0.21 < threshold=0.25 → passes Gate 8.

    Regression: old Gate 8 threshold (0.08) and Gate 8b denominator (0.20) blocked
    high-confidence YES calls with moderate Kalshi disagreement. Uses 50¢ fill to avoid
    Gate 11 (which blocks YES fills < 45¢ at k_cal > 0.75 — different issue).
    """
    signal = make_signal(direction=1, calibrated_prob=0.89)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 49
    kw["fresh_kalshi_mid"] = 0.29
    kw["available_contracts"] = 200
    r = checklist.run(**kw)
    assert r.passed
    assert r.kelly_contracts >= 1
    assert r.kelly_dollars > 0


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
    kw["fresh_kalshi_mid"] = 0.55  # opposing=0.05, mult=1-0.05/0.30≈0.833
    kw["is_drifting"] = True
    r_both = checklist.run(**kw)
    expected = r_base.kelly_dollars * (1.0 - 0.05 / 0.30) * 0.5
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


# ── Gate 8: confidence-aware threshold ───────────────────────────────────────

def test_gate8_low_confidence_signal_blocked_at_mild_kalshi_disagreement(checklist):
    """Low k15 confidence (prob=0.60, distance=0.10 < 0.15) → threshold=0.10.
    kalshi_mid=0.38 → opposing=0.12 > 0.10 → Gate 8 blocks even at mild disagreement."""
    signal = make_signal(direction=1, calibrated_prob=0.60)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 38
    kw["best_bid_cents"] = 37
    kw["available_contracts"] = 200
    kw["fresh_kalshi_mid"] = 0.38
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8


def test_gate8_medium_confidence_signal_blocked_at_moderate_kalshi_disagreement(checklist):
    """Medium k15 confidence (prob=0.70, distance=0.20 in 0.15–0.29) → threshold=0.15.
    kalshi_mid=0.30 → opposing=0.20 > 0.15 → Gate 8 blocks."""
    signal = make_signal(direction=1, calibrated_prob=0.70)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 30
    kw["best_bid_cents"] = 29
    kw["available_contracts"] = 200
    kw["fresh_kalshi_mid"] = 0.30
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8


def test_gate8_medium_confidence_passes_when_kalshi_barely_aligned(checklist):
    """Medium k15 confidence (prob=0.65, distance=0.15) → threshold=0.15.
    kalshi_mid=0.36 → opposing=0.14 < 0.15 → passes Gate 8."""
    signal = make_signal(direction=1, calibrated_prob=0.65)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 36
    kw["best_bid_cents"] = 35
    kw["available_contracts"] = 200
    kw["fresh_kalshi_mid"] = 0.36
    r = checklist.run(**kw)
    assert r.failed_gate != 8


def test_gate8_high_confidence_still_passes_moderate_kalshi_disagreement(checklist):
    """High k15 confidence (prob=0.89, distance=0.39 ≥ 0.30) → threshold=0.25.
    kalshi_mid=0.29 → opposing=0.21 < 0.25 → passes Gate 8. Uses 50¢ fill to avoid
    Gate 11 (Gate 11 blocks YES < 45¢ at k_cal > 0.75; Gate 8 behavior is separate)."""
    signal = make_signal(direction=1, calibrated_prob=0.89)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 49
    kw["available_contracts"] = 200
    kw["fresh_kalshi_mid"] = 0.29
    r = checklist.run(**kw)
    assert r.passed


# ── Gate 10 removed ───────────────────────────────────────────────────────────

def test_gate10_removed_trending_down_yes_now_passes(checklist):
    """Gate 10 removed — trending_down + YES→UP is no longer blocked."""
    signal = make_signal(direction=1, calibrated_prob=0.55, deepseek_regime="trending_down")
    r = checklist.run(**base_kwargs(signal))
    assert r.passed


def test_gate10_removed_trending_up_no_now_passes(checklist):
    """Gate 10 removed — trending_up + NO→DOWN is no longer blocked."""
    signal = make_signal(direction=0, calibrated_prob=0.45, deepseek_regime="trending_up")
    r = checklist.run(**base_kwargs(signal))
    assert r.passed


# ── Gate 11: Overconfidence guard ────────────────────────────────────────────

def test_gate11_fires_high_kcal_low_fill_yes(checklist):
    """direction=YES, k_cal=0.85, fill=40¢ → Gate 11 blocks.
    High Kronos confidence + market pricing YES below 45¢ = market strongly disagrees.
    Post-May-26 data shows 15% win rate in this zone."""
    signal = make_signal(direction=1, calibrated_prob=0.85)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 40
    kw["best_bid_cents"] = 38
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 11


def test_gate11_does_not_fire_high_fill(checklist):
    """direction=YES, k_cal=0.85, fill=50¢ → Gate 11 must not fire (fill >= 45¢ threshold)."""
    signal = make_signal(direction=1, calibrated_prob=0.85)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48
    r = checklist.run(**kw)
    assert r.failed_gate != 11


def test_gate11_does_not_fire_low_kcal(checklist):
    """direction=YES, k_cal=0.60, fill=35¢ → Gate 11 must not fire (k_cal <= 0.75 floor)."""
    signal = make_signal(direction=1, calibrated_prob=0.60)
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 35
    kw["best_bid_cents"] = 33
    kw["available_contracts"] = 200
    r = checklist.run(**kw)
    assert r.failed_gate != 11


def test_gate11_does_not_fire_no_direction(checklist):
    """direction=NO, k_cal=0.85, NO fill=35¢ (= YES at 65¢). Gate 11 only guards YES direction."""
    signal = TradingSignal(
        direction=0,
        calibrated_prob=0.35,
        kronos_raw=0.85,
        kronos_calibrated=0.85,
        regime_prob=0.50,
        regime_direction=0,
        deepseek_regime="neutral",
        timeframe="5min",
        strike=95000.0,
        timestamp=datetime.now(timezone.utc),
        regime_features={},
    )
    kw = base_kwargs(signal)
    kw["best_bid_cents"] = 65  # NO price = 100 - 65 = 35¢
    kw["best_ask_cents"] = 67
    r = checklist.run(**kw)
    assert r.failed_gate != 11


# ── High-uncertainty: no Kelly shrink in checklist ────────────────────────────
# The 50% shrink in fusion.py already compresses `combined` toward 0.5, reducing
# stated edge and feeding Gates 3 and 5. The checklist shrink was a redundant
# sizing filter uncorrelated with signal quality (68.4% WR blocked, 45.2% taken).

def test_high_uncertainty_kelly_same_as_neutral(checklist):
    """high_uncertainty regime no longer shrinks kelly — fusion handles it."""
    uncertain = make_signal(calibrated_prob=0.75, deepseek_regime="high_uncertainty")
    normal    = make_signal(calibrated_prob=0.75, deepseek_regime="neutral")
    r_u = checklist.run(**base_kwargs(uncertain))
    r_n = checklist.run(**base_kwargs(normal))
    assert r_u.passed
    assert abs(r_u.kelly_dollars - r_n.kelly_dollars) < 0.01


def test_high_uncertainty_kelly_contracts_same_as_neutral(checklist):
    """high_uncertainty no longer reduces contracts vs neutral at same prob."""
    uncertain = make_signal(calibrated_prob=0.80, deepseek_regime="high_uncertainty")
    normal    = make_signal(calibrated_prob=0.80, deepseek_regime="neutral")
    r_u = checklist.run(**base_kwargs(uncertain))
    r_n = checklist.run(**base_kwargs(normal))
    assert r_u.passed
    assert r_u.kelly_contracts == r_n.kelly_contracts


def test_high_uncertainty_does_not_block(checklist):
    """high_uncertainty should not hard-block on its own."""
    signal = make_signal(calibrated_prob=0.65, deepseek_regime="high_uncertainty")
    r = checklist.run(**base_kwargs(signal))
    assert r.passed
    assert r.kelly_contracts > 0


def test_other_regimes_kelly_unchanged(checklist):
    """ranging, trending_up, neutral regimes all produce the same kelly."""
    normal_kelly = None
    for regime in ("neutral", "ranging", "trending_up"):
        signal = make_signal(calibrated_prob=0.75, deepseek_regime=regime)
        r = checklist.run(**base_kwargs(signal))
        assert r.passed
        if normal_kelly is None:
            normal_kelly = r.kelly_dollars
        assert abs(r.kelly_dollars - normal_kelly) < 0.01, (
            f"Regime '{regime}' unexpectedly changed kelly: {r.kelly_dollars} vs {normal_kelly}"
        )


def test_high_uncertainty_strong_k15_no_longer_blocked_by_shrink(checklist):
    """Regression: high_uncertainty + strong k15 (0.75) that old shrink would kill.

    Old path: compute_size=0.26, halved to 0.13, 0.13 < 0.25 (0.5×50¢) →
    fail Gate 2 'rounds to 0 after high_uncertainty shrink'.
    New path: no shrink → kelly_dollars=0.26 ≥ 0.25 → 1 contract → passes.
    """
    signal = make_signal(calibrated_prob=0.75, deepseek_regime="high_uncertainty")
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48
    with patch.object(checklist._kelly, "compute_size", return_value=0.26), \
         patch.object(checklist._kelly, "dollars_to_contracts", return_value=0):
        r = checklist.run(**kw)
    assert r.passed
    assert r.kelly_contracts == 1


# ── Gate 5 regime-aware edge floor ──────────────────────────────────────────

def test_gate5_ranging_edge_below_015_fails(checklist):
    # ranging requires edge >= 0.15; edge=0.10 should fail Gate 5
    signal = make_signal(calibrated_prob=0.60, deepseek_regime="ranging")
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48  # spread=$0.02, base_min=$0.025, ranging floor=0.15
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 5


def test_gate5_ranging_edge_above_015_passes(checklist):
    # ranging requires edge >= 0.15; edge=0.16 should pass Gate 5
    signal = make_signal(calibrated_prob=0.66, deepseek_regime="ranging")
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48  # spread=$0.02, ranging min_required=0.15
    r = checklist.run(**kw)
    assert r.passed


def test_gate5_high_uncertainty_edge_below_008_fails(checklist):
    # high_uncertainty requires edge >= 0.08; edge=0.07 should fail Gate 5
    signal = make_signal(calibrated_prob=0.57, deepseek_regime="high_uncertainty")
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48  # spread=$0.02, high_uncertainty min_required=0.08
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 5


def test_gate5_trending_up_passes_with_small_edge(checklist):
    # trending_up uses base threshold only; edge=0.006 > spread+0.005=0.005 → passes
    signal = make_signal(calibrated_prob=0.506, deepseek_regime="trending_up")
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 50  # spread=$0, min_required=$0.005
    r = checklist.run(**kw)
    assert r.passed


# ── Gate 8: high_uncertainty flat threshold ───────────────────────────────────

def test_gate8_high_uncertainty_uses_flat_005_threshold(checklist):
    """high_uncertainty uses flat 0.05 Gate 8 threshold regardless of signal_confidence.

    prob=0.89 → signal_confidence=0.39 ≥ 0.30 → would be gate8_base=0.25 in neutral.
    In high_uncertainty: gate8_base=0.05.
    fresh_kalshi_mid=0.44 → opposing=0.06 > 0.05 → Gate 8 blocks.
    In neutral at same opposing=0.06 < 0.25 → Gate 8 would pass.
    """
    signal = make_signal(direction=1, calibrated_prob=0.89, deepseek_regime="high_uncertainty")
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48
    kw["available_contracts"] = 200
    kw["fresh_kalshi_mid"] = 0.44  # opposing=0.06 > 0.05 flat threshold
    r = checklist.run(**kw)
    assert not r.passed
    assert r.failed_gate == 8


def test_gate8_high_uncertainty_same_signal_passes_in_neutral(checklist):
    """Control: same signal in neutral regime uses threshold=0.25, passes Gate 8 at opposing=0.06."""
    signal = make_signal(direction=1, calibrated_prob=0.89, deepseek_regime="neutral")
    kw = base_kwargs(signal)
    kw["best_ask_cents"] = 50
    kw["best_bid_cents"] = 48
    kw["available_contracts"] = 200
    kw["fresh_kalshi_mid"] = 0.44  # opposing=0.06 < 0.25 → passes in neutral
    r = checklist.run(**kw)
    assert r.failed_gate != 8
