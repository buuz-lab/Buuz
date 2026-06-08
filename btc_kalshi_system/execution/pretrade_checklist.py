from dataclasses import dataclass
from typing import Optional

import redis
import config
from btc_kalshi_system.execution.kelly import KellySizer
from btc_kalshi_system.signal.fusion import TradingSignal


class ProgressCapModel:
    """Interface for dynamic candle-progress entry cap. Swap RuleBasedProgressCap
    for a LogisticProgressCap once 200+ candle_features rows under regime v2 exist."""
    def get_cap(self, volatility: float, spread: float, volume_ratio: float) -> float:
        raise NotImplementedError


class RuleBasedProgressCap(ProgressCapModel):
    """Rule-based entry window cap based on BRTI volatility and Kalshi spread.

    Thresholds calibrated after 100+ candle_features rows under regime v2.
    Query: SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY brti_volatility_1h)
           FROM candle_features WHERE features_stale=0 AND brti_volatility_1h IS NOT NULL;
    """
    _HIGH_VOL = 0.003     # ~0.3% per 5min — active market
    _WIDE_SPREAD = 0.04   # >4¢ spread — thin or rapidly repricing

    def get_cap(self, volatility: float, spread: float, volume_ratio: float) -> float:
        # Data: 10-15% progress = -$2.62/trade, 15-20% = -$5.74/trade — edge gone by 90s.
        # Upper bound is 10% in all conditions; 5% only when both volatile AND wide spread.
        high_vol    = volatility > self._HIGH_VOL
        wide_spread = spread    > self._WIDE_SPREAD
        if high_vol and wide_spread:
            return 0.05
        else:
            return 0.10


_PROGRESS_CAP_MODEL = RuleBasedProgressCap()


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

        # Gate 2a — Minimum price filter (YES direction only).
        # YES at sub-20¢: 0W/10L historically — market pricing UP >80¢ is too
        # expensive for meaningful Kronos edge. NOT applied to NO direction:
        # NO at sub-20¢ fill means Kalshi prices YES at 80-98¢ (extreme bull).
        # When k15 disagrees in these setups, NO has gone 32W/0L historically —
        # Kalshi extreme-bull mispricing that k15 correctly fades.
        _MIN_TRADE_PRICE_CENTS = 20
        if signal.direction == 1 and trade_price_cents < _MIN_TRADE_PRICE_CENTS:
            return fail(2, f"Trade price {trade_price_cents}¢ below minimum {_MIN_TRADE_PRICE_CENTS}¢ (YES at extreme price, 0W/10L historically)")

        # Gate 2b — NO maximum price filter.
        # NO fill > 55¢ means YES bid < 45¢ — the market is already pricing DOWN >55%.
        # In this zone NO has gone 30W/120L (25% WR), losing $740 historically.
        # When the market is already strongly bearish, k15's NO signal adds no edge —
        # BTC bounces or doesn't fall further within the 15-min window 75% of the time.
        # Sub-20¢ NO (the mispricing edge) is exempt — handled above.
        _MAX_NO_TRADE_PRICE_CENTS = 55
        if signal.direction == 0 and trade_price_cents > _MAX_NO_TRADE_PRICE_CENTS:
            return fail(2, f"NO fill {trade_price_cents}¢ exceeds {_MAX_NO_TRADE_PRICE_CENTS}¢ max (market already bearish, 25% WR historically)")

        # Gate 12 — Dynamic candle progress window (floor 3%, ceiling 5-10%)
        # Floor: wait for T+27s so Kalshi can reprice to the candle open (avg 2.71¢ move in 30s).
        # Ceiling: edge decays rapidly after 90s (10-15% = -$2.62/trade, 15-20% = -$5.74/trade).
        # Thresholds: _HIGH_VOL=0.3%/5min, _WIDE_SPREAD=4¢. Rules: both→5%, else→10%.
        _PROGRESS_FLOOR = 0.03
        candle_progress = (signal.regime_features or {}).get("candle_progress", 0.0) or 0.0
        _volatility  = (signal.regime_features or {}).get("brti_volatility_1h", 0.0) or 0.0
        _spread      = (signal.market_context  or {}).get("kalshi_spread_normalized", 0.0) or 0.0
        _vol_ratio   = (signal.regime_features or {}).get("volume_ratio_1h", 1.0) or 1.0
        _cap = _PROGRESS_CAP_MODEL.get_cap(_volatility, _spread, _vol_ratio)
        if candle_progress < _PROGRESS_FLOOR:
            return fail(12, f"Candle progress {candle_progress:.3f} below {_PROGRESS_FLOOR} floor — waiting for T+27s Kalshi reaction")
        if candle_progress > _cap:
            return fail(12, (
                f"Candle progress {candle_progress:.2f} exceeds dynamic cap {_cap:.2f} "
                f"(vol={_volatility:.4f} spread={_spread:.3f})"
            ))

        # NOTE: Gate 11 uses signal.kronos_calibrated which equals kronos_raw while the
        # calibrator is in passthrough mode. Once the calibrator activates and compresses
        # strong signals (e.g. k15_raw=0.80 → k15_cal≈0.55), this gate will silently
        # deactivate — calibrated values will not reach the 0.75 threshold. This is
        # intentional: the calibrator's compression is the correct fix for overconfident
        # signals, making this gate redundant. Monitor gate_rejections failed_gate=11
        # counts after calibrator deploys to confirm.
        # Gate 11 — Overconfidence guard
        # Block YES trades where Kronos is at high confidence (k_cal > 0.75) but the
        # market prices strongly against us (YES fill < 45¢). In this zone, the market's
        # disagreement is informative: post-May-26 data shows 15% win rate on 13 trades.
        # The calibrator compresses k_raw=1.0 to ~0.56 but keeps direction YES, so this
        # gate is still needed after calibrator activates.
        # Only applies to YES direction — NO direction at low prices has different dynamics.
        _OVERCONFIDENCE_K_CAL_FLOOR = 0.75
        _OVERCONFIDENCE_MAX_FILL_CENTS = 45
        if (signal.direction == 1
                and signal.kronos_calibrated > _OVERCONFIDENCE_K_CAL_FLOOR
                and trade_price_cents < _OVERCONFIDENCE_MAX_FILL_CENTS):
            return fail(
                11,
                f"Overconfidence guard: k_cal={signal.kronos_calibrated:.2f} but "
                f"YES fill {trade_price_cents}¢ < {_OVERCONFIDENCE_MAX_FILL_CENTS}¢ "
                f"(market disagrees strongly; 15% historical win rate in this zone)",
            )

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

        # Gate 13 — Kelly hard cap at $8
        # Regime v2 at 100% weight can produce extreme probabilities (0.02–0.98) that
        # yield uncalibrated Kelly sizes. Until Phase 3c calibrator is built, cap at $8.
        # Backtest: kelly≥$10 had 35% WR and -$3.30/trade; kelly<$5 had 52% WR and -$0.05/trade.
        # This is a SCALE-DOWN not a reject — trade still proceeds at capped size.
        _KELLY_HARD_CAP = 8.0
        if kelly_dollars > _KELLY_HARD_CAP:
            kelly_dollars = _KELLY_HARD_CAP
            kelly_contracts = self._kelly.dollars_to_contracts(kelly_dollars, trade_price_cents)
            if kelly_contracts == 0:
                kelly_contracts = 1

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
        base_min = spread_dollars + 0.005
        if signal.deepseek_regime == "ranging":
            min_required = max(base_min, 0.15)
        elif signal.deepseek_regime == "high_uncertainty":
            min_required = max(base_min, 0.08)
        else:
            min_required = base_min
        if signal_edge <= min_required:
            return fail(5, f"Signal edge {signal_edge:.4f} does not exceed min required {min_required:.4f} (regime={signal.deepseek_regime})")

        # Gate 14 — Edge upper bound at 20¢
        # Regime v2 at 100% weight produces extreme probabilities in the current bear
        # market, yielding computed edges of 40–50¢. These are uncalibrated — XGBoost
        # at 357 training rows is overconfident at the extremes. Backtest: >20¢ edge
        # bucket had 46% WR and -$1.30/trade. Cap until Phase 3c calibrator is deployed.
        _EDGE_CEILING = 0.20
        if signal_edge > _EDGE_CEILING:
            return fail(14, f"Edge {signal_edge:.4f} exceeds {_EDGE_CEILING:.2f} ceiling — uncalibrated regime v2 overconfidence")

        # Gate 6 — Strike proximity check (KXBTCD / strike markets only)
        # For KXBTC15M up/down markets _extract_strike uses the last completed
        # 15-min BRTI close as the threshold, not composite_price.  Applying a
        # $150 proximity gate would reject every 15-min market unconditionally.
        # Skip Gate 6 for the 15min timeframe.
        if signal.timeframe != "15min":
            distance = abs(composite_price - signal.strike)
            if distance < 150:
                return fail(6, f"Composite price ${composite_price:,.0f} within $150 of strike ${signal.strike:,.0f} (distance ${distance:.0f})")

        # Gate 8 — Kalshi consensus hard block (confidence-aware threshold)
        # High-conviction signals (k15_cal far from 0.5) tolerate more Kalshi
        # disagreement. Low-conviction signals must respect the market more.
        if signal.deepseek_regime == "high_uncertainty":
            # Flat threshold: confidence tiers are meaningless here because calibrator
            # compression collapses calibrated_prob toward 0.5 in this regime.
            # Kalshi accuracy > Kronos accuracy in high_uncertainty (54.9% vs 18.4%
            # in losing periods) — tighter threshold weights Kalshi's view more heavily.
            # Gate 5 already requires 8% Kronos edge; Kalshi opposing by nearly as much
            # cancels that edge. 5% threshold is approximately half Gate 5's floor.
            # Revisit confidence tiers after regime v2 + calibrator give reliable
            # signal_confidence values.
            gate8_base = 0.05
        else:
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

        # Trending-regime DOWN-direction shrink — bootstrap only.
        # Data from Kronos-era trades (before regime v2 trained): DOWN bets in trending_up
        # = -$1.17/trade (bear-bias) and trending_down = -$3.03/trade (chasing moves).
        # Guard is removed once regime v2 deploys — at that point the model's own
        # calibrated probabilities reflect current market context. Re-evaluate with
        # 50+ regime-v2 trending trades before deciding whether to re-enable.
        if is_bootstrap and signal.deepseek_regime in ("trending_up", "trending_down") and signal.direction == 0:
            kelly_dollars = kelly_dollars * 0.5
            kelly_contracts = max(1, kelly_contracts // 2)

        return ChecklistResult(
            passed=True,
            failed_gate=None,
            failed_reason=None,
            kelly_dollars=kelly_dollars,
            kelly_contracts=kelly_contracts,
        )
