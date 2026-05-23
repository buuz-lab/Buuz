"""
Tests for Gate 7: CVD soft gate in PreTradeChecklist.
"""
import pytest
from unittest.mock import MagicMock

import config as _config
from btc_kalshi_system.execution.pretrade_checklist import PreTradeChecklist
from btc_kalshi_system.execution.kelly import KellySizer
from btc_kalshi_system.signal.fusion import TradingSignal
from datetime import datetime, timezone
import math


def _make_signal(direction: int, cvd: float, calibrated_prob: float = 0.65) -> TradingSignal:
    return TradingSignal(
        direction=direction,
        calibrated_prob=calibrated_prob,
        kronos_raw=0.65,
        kronos_calibrated=calibrated_prob,
        regime_prob=float("nan"),
        regime_direction=-1,
        deepseek_regime="trending_up",
        timeframe="15min",
        strike=95000.0,
        timestamp=datetime.now(timezone.utc),
        regime_features={"cvd_normalized": cvd},
        features_stale=False,
    )


def _make_checklist() -> PreTradeChecklist:
    kelly = MagicMock(spec=KellySizer)
    kelly.compute_size.return_value = 10.0
    kelly.dollars_to_contracts.return_value = 2
    return PreTradeChecklist(kelly)


GOOD_KWARGS = dict(
    best_ask_cents=65,
    best_bid_cents=63,
    available_contracts=10,
    current_exposure=50.0,
    same_timeframe_open=False,
    composite_price=95000.0,
    edge_above_threshold=True,
)


def test_gate7_blocks_yes_up_with_negative_cvd():
    """direction=1 (YES→UP), cvd=-0.4 → Gate 7 blocks."""
    checklist = _make_checklist()
    signal = _make_signal(direction=1, cvd=-0.4)
    # Use ask=60 so edge = 0.65 - 0.60 = 0.05 > spread(0.02) + 0.005 → Gate 5 passes
    kwargs = {**GOOD_KWARGS, "best_ask_cents": 60, "best_bid_cents": 58}
    result = checklist.run(signal=signal, **kwargs)
    assert not result.passed
    assert result.failed_gate == 7
    assert "YES→UP" in result.failed_reason


def test_gate7_passes_yes_up_with_mild_negative_cvd():
    """direction=1, cvd=-0.2 (below threshold) → passes Gate 7."""
    checklist = _make_checklist()
    signal = _make_signal(direction=1, cvd=-0.2)
    result = checklist.run(signal=signal, **GOOD_KWARGS)
    # Gate 7 passes; may fail on other gates but not gate 7
    assert result.failed_gate != 7


def test_gate7_blocks_no_down_with_positive_cvd():
    """direction=0 (NO→DOWN), cvd=+0.4 → Gate 7 blocks."""
    checklist = _make_checklist()
    signal = _make_signal(direction=0, cvd=0.4, calibrated_prob=0.35)
    result = checklist.run(signal=signal, **GOOD_KWARGS)
    assert not result.passed
    assert result.failed_gate == 7
    assert "NO→DOWN" in result.failed_reason


def test_gate7_passes_no_down_with_mild_positive_cvd():
    """direction=0, cvd=+0.2 → passes Gate 7."""
    checklist = _make_checklist()
    signal = _make_signal(direction=0, cvd=0.2, calibrated_prob=0.35)
    result = checklist.run(signal=signal, **GOOD_KWARGS)
    assert result.failed_gate != 7


def test_gate7_passes_yes_up_with_positive_cvd():
    """direction=1, positive CVD (aligned) → passes Gate 7."""
    checklist = _make_checklist()
    signal = _make_signal(direction=1, cvd=0.5)
    result = checklist.run(signal=signal, **GOOD_KWARGS)
    assert result.failed_gate != 7


def test_gate7_passes_no_down_with_negative_cvd():
    """direction=0, negative CVD (aligned) → passes Gate 7."""
    checklist = _make_checklist()
    signal = _make_signal(direction=0, cvd=-0.5, calibrated_prob=0.35)
    result = checklist.run(signal=signal, **GOOD_KWARGS)
    assert result.failed_gate != 7


def test_gate7_threshold_boundary_exactly_at_threshold_passes():
    """cvd = -CVD_GATE_THRESHOLD exactly → does NOT block (gate uses strict <)."""
    checklist = _make_checklist()
    signal = _make_signal(direction=1, cvd=-_config.CVD_GATE_THRESHOLD)
    result = checklist.run(signal=signal, **GOOD_KWARGS)
    assert result.failed_gate != 7


def test_gate7_missing_cvd_in_features_defaults_to_zero_and_passes():
    """regime_features without cvd_normalized → defaults to 0.0 → passes Gate 7."""
    checklist = _make_checklist()
    signal = _make_signal(direction=1, cvd=0.0)
    signal.regime_features.pop("cvd_normalized", None)
    result = checklist.run(signal=signal, **GOOD_KWARGS)
    assert result.failed_gate != 7
