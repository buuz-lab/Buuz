from dataclasses import dataclass
from typing import Optional

import redis
import config
from btc_kalshi_system.execution.kelly import KellySizer
from btc_kalshi_system.signal.fusion import TradingSignal


@dataclass
class ChecklistResult:
    passed: bool
    failed_gate: Optional[int]
    failed_reason: Optional[str]
    kelly_dollars: float
    kelly_contracts: int
    kalshi_mid_at_block: Optional[float] = None


class PreTradeChecklist:
    def __init__(self, kelly_sizer: KellySizer) -> None:
        self._kelly = kelly_sizer
        self._redis = redis.from_url(config.REDIS_URL)

    def run(
        self,
        signal: TradingSignal,
        best_ask_cents: int,
        best_bid_cents: int,
        available_contracts: int,
        current_exposure: float,
        same_timeframe_open: bool,
        composite_price: float,
        edge_above_threshold: bool,
        fresh_kalshi_mid: float = 0.5,
        is_drifting: bool = False,
        direction_win_rate: Optional[float] = None,
        is_bootstrap: bool = False,
    ) -> ChecklistResult:
        def fail(gate: int, reason: str, kalshi_mid: Optional[float] = None) -> ChecklistResult:
            return ChecklistResult(
                passed=False,
                failed_gate=gate,
                failed_reason=reason,
                kelly_dollars=0.0,
                kelly_contracts=0,
                kalshi_mid_at_block=kalshi_mid,
            )

        # Gate 1 — Spread check
        spread_cents = best_ask_cents - best_bid_cents
        spread_dollars = spread_cents / 100
        if spread_dollars > 0.03:
            return fail(1, f"Spread ${spread_dollars:.3f} exceeds $0.03 limit")

        # Gate 2 — Depth check (also computes kelly for final result)
        # "yes" trades pay ask_cents; "no" trades pay (100 - bid_cents).
        # Kelly and contract sizing must use the actual price being paid and the
        # correct win probability for each direction.
        if signal.direction == 1:
            win_prob = signal.calibrated_prob
            trade_price_cents = best_ask_cents
        else:
            win_prob = 1.0 - signal.calibrated_prob
            trade_price_cents = 100 - best_bid_cents
        market_price = trade_price_cents / 100

        # Gate 2a — Minimum price filter: reject extreme-priced markets.
        # Sub-20¢ contracts require 100-400+ contracts for any meaningful dollar
        # size, exhausting orderbook depth every time. Historically 0W/10L at ≤18¢.
        _MIN_TRADE_PRICE_CENTS = 20
        if trade_price_cents < _MIN_TRADE_PRICE_CENTS:
            return fail(2, f"Trade price {trade_price_cents}¢ below minimum {_MIN_TRADE_PRICE_CENTS}¢ (extreme/illiquid market)")
        loss_streak = int(self._redis.get("trading:loss_streak") or 0)
        kelly_dollars = self._kelly.compute_size(
            prob=win_prob,
            market_price=market_price,
            current_exposure=current_exposure,
            same_timeframe_open=same_timeframe_open,
            regime_features=signal.regime_features,
            loss_streak=loss_streak,
            direction_win_rate=direction_win_rate,
        )
        kelly_contracts = self._kelly.dollars_to_contracts(kelly_dollars, trade_price_cents)

        if kelly_contracts == 0:
            # Bootstrap floor: regime model untrained, positive edge, price 25–75¢.
            # Breaks the chicken-and-egg deadlock where Kelly rounds to 0 in bootstrap
            # mode, starving the system of training data.
            if is_bootstrap and kelly_dollars > 0 and 25 <= trade_price_cents <= 75:
                kelly_contracts = 1
            elif kelly_dollars >= (trade_price_cents / 100) * 0.5:
                kelly_contracts = 1
            else:
                return fail(2, "Kelly size rounds to 0 contracts")
        if kelly_contracts > available_contracts:
            if available_contracts == 0:
                return fail(2, "Insufficient depth: 0 contracts available")
            kelly_contracts = available_contracts
            kelly_dollars = kelly_contracts * (trade_price_cents / 100)

        # Gate 8b — Kalshi Kelly multiplier (continuous gradient reduction before hard block)
        opposing_margin = max(0.0, (fresh_kalshi_mid - 0.5) if signal.direction == 0 else (0.5 - fresh_kalshi_mid))
        _pre_mult_kelly_dollars = kelly_dollars
        kalshi_kelly_mult = max(0.0, 1.0 - opposing_margin / 0.30)
        kelly_dollars *= kalshi_kelly_mult
        kelly_contracts = self._kelly.dollars_to_contracts(kelly_dollars, trade_price_cents)
        if kelly_contracts == 0:
            if is_bootstrap and _pre_mult_kelly_dollars > 0 and 25 <= trade_price_cents <= 75:
                kelly_contracts = 1
            elif kelly_dollars >= (trade_price_cents / 100) * 0.5:
                kelly_contracts = 1
            else:
                return fail(2, "Kelly size rounds to 0 contracts after Kalshi Kelly multiplier")

        # Drift Kelly shrink — 50% additional shrink when calibration drift detected
        if is_drifting:
            kelly_dollars *= 0.5
            kelly_contracts = self._kelly.dollars_to_contracts(kelly_dollars, trade_price_cents)
            if kelly_contracts == 0:
                if is_bootstrap and kelly_dollars > 0 and 25 <= trade_price_cents <= 75:
                    kelly_contracts = 1
                elif kelly_dollars >= (trade_price_cents / 100) * 0.5:
                    kelly_contracts = 1
                else:
                    return fail(2, "Kelly size rounds to 0 contracts after drift shrink")

        # Gate 3 — High uncertainty + thin edge
        edge_from_center = abs(signal.calibrated_prob - 0.5)
        if signal.deepseek_regime == "high_uncertainty" and edge_from_center < 0.05:
            return fail(3, f"High uncertainty regime with thin edge ({edge_from_center:.3f} from center)")

        # Gate 4 — Rolling edge check
        if not edge_above_threshold:
            return fail(4, "Rolling realized edge below threshold")

        # Gate 5 — Signal edge vs spread check
        # For "yes": edge = P(up) - ask_price
        # For "no":  edge = P(down) - no_price = (1 - P(up)) - (1 - bid_price) = bid_price - P(up)
        signal_edge = win_prob - market_price
        min_required = spread_dollars + 0.005
        if signal_edge <= min_required:
            return fail(5, f"Signal edge {signal_edge:.4f} does not exceed spread + 0.005 ({min_required:.4f})")

        # Gate 6 — Strike proximity check (KXBTCD / strike markets only)
        # For KXBTC15M up/down markets _extract_strike uses the last completed
        # 15-min BRTI close as the threshold, not composite_price.  Applying a
        # $150 proximity gate would reject every 15-min market unconditionally.
        # Skip Gate 6 for the 15min timeframe.
        if signal.timeframe != "15min":
            distance = abs(composite_price - signal.strike)
            if distance < 150:
                return fail(6, f"Composite price ${composite_price:,.0f} within $150 of strike ${signal.strike:,.0f} (distance ${distance:.0f})")

        # Gate 10 — DeepSeek trend-direction conflict
        # When the regime classifier identifies a confirmed trend that directly
        # opposes our trade direction, the market is systematically moving against
        # us and Kronos is likely seeing noise/bounces. Block unconditionally.
        if (signal.deepseek_regime == "trending_down" and signal.direction == 1) or \
                (signal.deepseek_regime == "trending_up" and signal.direction == 0):
            side = "YES→UP" if signal.direction == 1 else "NO→DOWN"
            return fail(10, f"DeepSeek regime '{signal.deepseek_regime}' contradicts {side}")

        # Gate 8 — Kalshi consensus hard block (confidence-aware threshold)
        # High-conviction signals (k15_cal far from 0.5) tolerate more Kalshi
        # disagreement. Low-conviction signals must respect the market more.
        signal_confidence = abs(signal.calibrated_prob - 0.5)
        if signal_confidence >= 0.30:     # k15_cal ≥ 0.80 or ≤ 0.20
            gate8_base = 0.25
        elif signal_confidence >= 0.15:   # k15_cal ≥ 0.65 or ≤ 0.35
            gate8_base = 0.15
        else:                             # k15_cal between 0.35 and 0.65
            gate8_base = 0.10
        oi_delta = signal.regime_features.get("oi_delta_pct", 0.0) if signal.regime_features else 0.0
        oi_squeeze = (oi_delta > 0.001) and (signal.direction == 0)
        effective_threshold = gate8_base / 4.0 if oi_squeeze else gate8_base
        opposing = (fresh_kalshi_mid - 0.5) if signal.direction == 0 else (0.5 - fresh_kalshi_mid)
        if opposing > effective_threshold:
            side = "NO→DOWN" if signal.direction == 0 else "YES→UP"
            return fail(8, f"Kalshi consensus {fresh_kalshi_mid:.3f} opposes {side} (threshold {effective_threshold:.3f})", kalshi_mid=fresh_kalshi_mid)

        return ChecklistResult(
            passed=True,
            failed_gate=None,
            failed_reason=None,
            kelly_dollars=kelly_dollars,
            kelly_contracts=kelly_contracts,
        )
