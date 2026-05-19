import math

KELLY_FRACTION = 0.25
MAX_SINGLE_TRADE_DOLLARS = 50.0
MAX_TOTAL_EXPOSURE_DOLLARS = 150.0
CORRELATION_DISCOUNT = 0.7


class KellySizer:
    def compute_size(
        self,
        prob: float,
        market_price: float,
        current_exposure: float,
        same_timeframe_open: bool,
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
        return max(size, 0.0)

    def dollars_to_contracts(self, dollars: float, price_cents: int) -> int:
        if price_cents <= 0 or dollars <= 0:
            return 0
        return math.floor(dollars / (price_cents / 100))
