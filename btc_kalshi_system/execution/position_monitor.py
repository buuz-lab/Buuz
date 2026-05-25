"""
PositionMonitor — mid-trade exit signal.

Polls open positions every 60 seconds. At T+5 and T+10 checkpoints evaluates
whether to close early: if both RegimeModel and Kronos flip direction versus
the entry direction, places an offsetting order and removes the position from
the tracker.

In bootstrap mode (regime_model._clf is None) collects feature snapshots for
future exit-classifier training but does NOT execute exits.
"""

import asyncio
import sqlite3
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from btc_kalshi_system.data.feature_store import FeatureStore
    from btc_kalshi_system.execution.router import KalshiClientRouter
    from btc_kalshi_system.models.kronos_engine import KronosEngine
    from btc_kalshi_system.models.regime_model import RegimeModel
    from btc_kalshi_system.portfolio.monitor import OpenPosition, PortfolioMonitor
    from btc_kalshi_system.signal.fusion import SignalFusionEngine


def _parse_orderbook_bbo(book: dict) -> tuple[int, int]:
    """Parse best bid/ask in cents from a Kalshi orderbook response.

    Returns (best_bid_cents, best_ask_cents). Returns (0, 0) on any parse error.
    Mirrors the parsing logic in main.py _parse_orderbook().
    """
    try:
        book_fp = book.get("orderbook_fp")
        if book_fp:
            yes_bids = book_fp.get("yes_dollars", [])
            no_bids = book_fp.get("no_dollars", [])
            if not no_bids:
                return (0, 0)
            best_yes_bid = float(yes_bids[-1][0]) if yes_bids else 0.0
            best_no_bid = float(no_bids[-1][0])
            best_bid_cents = round(best_yes_bid * 100)
            best_ask_cents = round((1.0 - best_no_bid) * 100)
            if best_ask_cents <= 0 or best_ask_cents >= 100:
                return (0, 0)
            return (best_bid_cents, best_ask_cents)

        # Legacy format
        legacy = book.get("orderbook", book)
        yes_bids = legacy.get("yes", [])
        no_bids = legacy.get("no", [])
        best_bid_cents = int(yes_bids[0][0]) if yes_bids else 0
        if not no_bids:
            return (0, 0)
        best_ask_cents = 100 - int(no_bids[0][0])
        return (best_bid_cents, best_ask_cents)
    except (IndexError, KeyError, TypeError, ValueError):
        return (0, 0)


class PositionMonitor:
    """Monitors open positions and executes mid-trade exits when warranted."""

    def __init__(
        self,
        portfolio_monitor: "PortfolioMonitor",
        regime_model: "RegimeModel",
        kronos_engine: "KronosEngine",
        feature_store: "FeatureStore",
        router: "KalshiClientRouter",
        fusion_engine: "SignalFusionEngine",
        db_path: str,
    ) -> None:
        self.portfolio_monitor = portfolio_monitor
        self.regime_model = regime_model
        self.kronos_engine = kronos_engine
        self.feature_store = feature_store
        self.router = router
        self.fusion_engine = fusion_engine
        self.db_path = db_path
        self._checked: set[tuple[str, str]] = set()

    async def run(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self._check_positions()

    async def _check_positions(self) -> None:
        positions = self.portfolio_monitor.get_open_positions()
        open_ids = {p.trade_id for p in positions}
        # Purge checked set of positions that have already closed
        self._checked = {(tid, w) for tid, w in self._checked if tid in open_ids}
        for position in positions:
            elapsed = time.time() - position.timestamp
            if elapsed >= 300 and (position.trade_id, "t5") not in self._checked:
                await self._evaluate(position, "t5")
            if elapsed >= 600 and (position.trade_id, "t10") not in self._checked:
                await self._evaluate(position, "t10")

    async def _evaluate(self, position: "OpenPosition", window: str) -> None:
        self._checked.add((position.trade_id, window))

        best_bid_cents: int | None = None
        best_ask_cents: int | None = None
        try:
            book = self.router.get_orderbook(position.ticker)
            bid, ask = _parse_orderbook_bbo(book)
            if bid > 0 and ask > 0:
                best_bid_cents = bid
                best_ask_cents = ask
                mid_cents = (bid + ask) / 2.0
                self.fusion_engine.update_kalshi_mid(mid_cents)
        except Exception as exc:
            logger.warning(
                f"PositionMonitor: orderbook fetch failed for {position.ticker}: {exc}"
            )

        features, stale, _, _ = self.fusion_engine._regime_features()

        snapshot: dict = {
            "trade_id":       position.trade_id,
            "snapshot_window": window,
            "snapshot_ts":    datetime.now(timezone.utc).isoformat(),
            **features,
            "kronos_prob":        None,
            "regime_direction":   None,
            "exit_triggered":     0,
        }

        if stale:
            self._write_snapshot(snapshot)
            return

        # Bootstrap: regime not trained yet — collect snapshot only, no exit.
        if self.regime_model._clf is None:
            loop = asyncio.get_event_loop()
            try:
                kp = await loop.run_in_executor(
                    None,
                    lambda: self.kronos_engine.run_monte_carlo(self.feature_store),
                )
                snapshot["kronos_prob"] = kp
            except Exception as exc:
                logger.warning(
                    f"PositionMonitor: Kronos failed during bootstrap snapshot: {exc}"
                )
            self._write_snapshot(snapshot)
            return

        # Regime inference
        try:
            regime_result = self.regime_model.get_regime(features)
            regime_direction = regime_result["direction"]
        except Exception as exc:
            logger.warning(f"PositionMonitor: regime inference failed: {exc}")
            self._write_snapshot(snapshot)
            return

        # Kronos inference (blocking — must run in executor)
        loop = asyncio.get_event_loop()
        try:
            kronos_prob = await loop.run_in_executor(
                None,
                lambda: self.kronos_engine.run_monte_carlo(self.feature_store),
            )
        except Exception as exc:
            logger.warning(f"PositionMonitor: Kronos inference failed: {exc}")
            self._write_snapshot(snapshot)
            return

        kronos_direction = 1 if kronos_prob >= 0.5 else 0
        should_exit = (
            regime_direction != position.direction
            and kronos_direction != position.direction
        )

        snapshot.update({
            "kronos_prob":      kronos_prob,
            "regime_direction": regime_direction,
            "exit_triggered":   int(should_exit),
        })
        self._write_snapshot(snapshot)

        if should_exit:
            await self._execute_exit(position, window, best_bid_cents, best_ask_cents)

    async def _execute_exit(
        self,
        position: "OpenPosition",
        window: str,
        best_bid_cents: int | None,
        best_ask_cents: int | None,
    ) -> None:
        # CRITICAL: remove_position() FIRST — decrements per-side count in Redis.
        # The offsetting order is a raw API call and must NEVER call add_position().
        self.portfolio_monitor.remove_position(position.trade_id)

        if best_bid_cents is None or best_ask_cents is None:
            logger.warning(
                f"PositionMonitor: no orderbook BBO for {position.ticker}, "
                f"cannot execute exit for {position.trade_id}. "
                f"Position already removed from tracker."
            )
            return

        opposite_side = "no" if position.direction == 1 else "yes"
        mid_cents = int((best_bid_cents + best_ask_cents) / 2)

        try:
            self.router.place_order(
                ticker=position.ticker,
                side=opposite_side,
                count=position.contracts,
                price_cents=mid_cents,
                client_order_id=f"exit_{position.trade_id}_{window}",
            )
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE trades SET exit_reason = ? WHERE trade_id = ?",
                (f"mid_trade_exit_{window}", position.trade_id),
            )
            conn.commit()
            conn.close()
            logger.info(
                f"PositionMonitor: closed {position.trade_id} at {window} — "
                f"{opposite_side} IOC {position.contracts}x at {mid_cents}¢ "
                f"(entry_direction={position.direction})"
            )
        except Exception as exc:
            logger.warning(
                f"PositionMonitor: exit order failed for {position.trade_id}: {exc}. "
                f"Position already removed from tracker — investigate Kalshi state manually."
            )

    def _write_snapshot(self, data: dict) -> None:
        cols = [
            "trade_id", "snapshot_window", "snapshot_ts",
            "funding_rate", "funding_rate_trend", "oi_delta_pct", "cvd_normalized",
            "basis_spread_pct", "brti_volatility_1h", "cvd_velocity", "cvd_acceleration",
            "brti_momentum_5min", "brti_momentum_15min", "candle_progress",
            "hour_sin", "hour_cos", "kalshi_implied_prob", "funding_window_proximity",
            "trend_slope_1h", "trend_r2_1h", "hourly_sr_proximity",
            "range_breakout_flag", "tape_speed_tpm",
            "atm_iv", "iv_rv_spread", "pcr_oi",
            "term_structure_slope", "skew_25d", "kalshi_spread_normalized",
            "kronos_prob", "regime_direction", "exit_triggered",
        ]
        try:
            conn = sqlite3.connect(self.db_path)
            placeholders = ", ".join("?" * len(cols))
            conn.execute(
                f"INSERT OR REPLACE INTO trade_snapshots "
                f"({', '.join(cols)}) VALUES ({placeholders})",
                [data.get(c) for c in cols],
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning(f"PositionMonitor: snapshot write failed: {exc}")
