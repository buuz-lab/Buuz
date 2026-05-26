import math

KELLY_FRACTION = 0.25
MAX_SINGLE_TRADE_DOLLARS = 50.0
MAX_TOTAL_EXPOSURE_DOLLARS = 150.0
CORRELATION_DISCOUNT = 0.7

KELLY_CHOP_THRESHOLD  = 0.15
KELLY_CHOP_SHRINK     = 0.70
KELLY_TAPE_THRESHOLD  = 0.20
KELLY_TAPE_SHRINK     = 0.80
KELLY_STREAK_FLOOR    = 0.60
KELLY_STREAK_STEP     = 0.08


class KellySizer:
    def compute_size(
        self,
        prob: float,
        market_price: float,
        current_exposure: float,
        same_timeframe_open: bool,
        regime_features: dict | None = None,
        loss_streak: int = 0,
        direction_win_rate: float | None = None,
    ) -> float:
        edge = prob - market_price
        if edge <= 0:
            return 0.0
        if current_exposure >= MAX_TOTAL_EXPOSURE_DOLLARS:
            return 0.0

        full_kelly = edge / (1 - market_price)
        fractional = full_kelly * KELLY_FRACTION
        raw_dollars = fractional * MAX_TOTAL_EXPOSURE_DOLLARS

        if same_timeframe_open:
            raw_dollars *= CORRELATION_DISCOUNT

        remaining_capacity = MAX_TOTAL_EXPOSURE_DOLLARS - current_exposure
        size = min(raw_dollars, MAX_SINGLE_TRADE_DOLLARS, remaining_capacity)

        if regime_features:
            if abs(regime_features.get("range_breakout_flag", 1.0)) < KELLY_CHOP_THRESHOLD:
                size *= KELLY_CHOP_SHRINK
            if regime_features.get("tape_speed_tpm", 1.0) < KELLY_TAPE_THRESHOLD:
                size *= KELLY_TAPE_SHRINK
        if loss_streak > 0:
            size *= max(KELLY_STREAK_FLOOR, 1.0 - max(0, loss_streak - 1) * KELLY_STREAK_STEP)

        if direction_win_rate is not None and direction_win_rate < 0.45:
            size *= 0.60

        return max(size, 0.0)

    def dollars_to_contracts(self, dollars: float, price_cents: int) -> int:
        if price_cents <= 0 or dollars <= 0:
            return 0
        return math.floor(dollars / (price_cents / 100))
