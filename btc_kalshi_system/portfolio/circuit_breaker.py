from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

import config
from btc_kalshi_system.execution.router import ClientState, KalshiClientRouter
from btc_kalshi_system.models.calibrator import Calibrator
from btc_kalshi_system.portfolio.monitor import PortfolioMonitor
from btc_kalshi_system.signal.edge_tracker import EdgeTracker

MAX_DAILY_DRAWDOWN_DOLLARS = 200.0
MIN_CALIBRATOR_SAMPLES = 500
ROLLING_EDGE_WINDOW = 30


class TripReason(Enum):
    NEGATIVE_ROLLING_EDGE = "negative_rolling_edge"
    DAILY_DRAWDOWN = "daily_drawdown"
    BOTH_CLIENTS_FAILED = "both_clients_failed"
    CALIBRATOR_INSUFFICIENT = "calibrator_insufficient"


@dataclass
class BreakerStatus:
    tripped: bool
    reason: Optional[TripReason]
    message: Optional[str]


_CLEAR = BreakerStatus(tripped=False, reason=None, message=None)


class CircuitBreaker:
    def __init__(
        self,
        monitor: PortfolioMonitor,
        edge_tracker: EdgeTracker,
        router: KalshiClientRouter,
        calibrator: Calibrator,
    ) -> None:
        self._monitor = monitor
        self._edge_tracker = edge_tracker
        self._router = router
        self._calibrator = calibrator

    def check(self) -> BreakerStatus:
        checks = [self._check_clients, self._check_drawdown]
        if not config.PAPER_TRADING:
            checks += [self._check_rolling_edge, self._check_calibrator]
        for fn in checks:
            result = fn()
            if result is not None:
                logger.error(f"Circuit breaker tripped: [{result.reason.value}] {result.message}")
                return result

        logger.debug("Circuit breaker: all checks clear")
        return _CLEAR

    def is_tripped(self) -> bool:
        return self.check().tripped

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_clients(self) -> Optional[BreakerStatus]:
        if self._router.state is ClientState.BOTH_FAILED:
            return BreakerStatus(
                tripped=True,
                reason=TripReason.BOTH_CLIENTS_FAILED,
                message="Both pykalshi and raw HTTP clients have failed",
            )
        return None

    def _check_drawdown(self) -> Optional[BreakerStatus]:
        daily_pnl = self._monitor.get_daily_pnl()
        if daily_pnl < -MAX_DAILY_DRAWDOWN_DOLLARS:
            return BreakerStatus(
                tripped=True,
                reason=TripReason.DAILY_DRAWDOWN,
                message=f"Daily drawdown ${abs(daily_pnl):.2f} exceeds ${MAX_DAILY_DRAWDOWN_DOLLARS:.0f} limit",
            )
        return None

    def _check_rolling_edge(self) -> Optional[BreakerStatus]:
        n = len(self._edge_tracker)
        if n >= ROLLING_EDGE_WINDOW:
            edge = self._edge_tracker.current_edge()
            if edge < 0:
                return BreakerStatus(
                    tripped=True,
                    reason=TripReason.NEGATIVE_ROLLING_EDGE,
                    message=f"Rolling realized edge {edge:.4f} is negative over last {n} trades",
                )
        return None

    def _check_calibrator(self) -> Optional[BreakerStatus]:
        n = self._calibrator.n_samples
        if n < MIN_CALIBRATOR_SAMPLES:
            return BreakerStatus(
                tripped=True,
                reason=TripReason.CALIBRATOR_INSUFFICIENT,
                message=f"Calibrator has {n} samples, need {MIN_CALIBRATOR_SAMPLES} before using Kelly sizing",
            )
        return None
