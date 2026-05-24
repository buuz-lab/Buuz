import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import redis
from loguru import logger

import config

OPEN_POSITIONS_KEY = "portfolio:open_positions"
RESOLVED_TRADES_KEY = "portfolio:resolved_trades"
DAILY_PNL_KEY = "portfolio:daily_pnl"
DAILY_PNL_DATE_KEY = "portfolio:daily_pnl_date"


@dataclass
class OpenPosition:
    trade_id: str
    ticker: str
    timeframe: str
    direction: int
    strike: float
    contracts: int
    entry_price_cents: int
    kelly_dollars: float
    timestamp: float
    calibrated_prob: float = 0.0
    deepseek_regime: str = "unknown"


@dataclass
class ResolvedTrade:
    trade_id: str
    ticker: str
    timeframe: str
    direction: int
    strike: float
    contracts: int
    entry_price_cents: int
    kelly_dollars: float
    outcome: int
    pnl_dollars: float
    timestamp: float
    resolved_at: float


class PortfolioMonitor:
    def __init__(self, redis_url: str = config.REDIS_URL) -> None:
        self._redis: redis.Redis = redis.from_url(redis_url, decode_responses=True)
        self._positions: dict[str, OpenPosition] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Init / persistence helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        try:
            raw = self._redis.hgetall(OPEN_POSITIONS_KEY)
            for trade_id, payload in raw.items():
                self._positions[trade_id] = OpenPosition(**json.loads(payload))
        except redis.RedisError as exc:
            logger.warning(f"Failed to load positions from Redis: {exc}")

    def _persist_position(self, position: OpenPosition) -> None:
        try:
            self._redis.hset(OPEN_POSITIONS_KEY, position.trade_id, json.dumps(asdict(position)))
        except redis.RedisError as exc:
            logger.warning(f"Failed to persist position {position.trade_id}: {exc}")

    def _delete_position(self, trade_id: str) -> None:
        try:
            self._redis.hdel(OPEN_POSITIONS_KEY, trade_id)
        except redis.RedisError as exc:
            logger.warning(f"Failed to delete position {trade_id} from Redis: {exc}")

    def _persist_resolved(self, trade: ResolvedTrade) -> None:
        try:
            self._redis.lpush(RESOLVED_TRADES_KEY, json.dumps(asdict(trade)))
        except redis.RedisError as exc:
            logger.warning(f"Failed to persist resolved trade {trade.trade_id}: {exc}")

    def _update_daily_pnl(self, delta: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            stored_date = self._redis.get(DAILY_PNL_DATE_KEY)
            if stored_date != today:
                self._redis.set(DAILY_PNL_KEY, str(delta))
                self._redis.set(DAILY_PNL_DATE_KEY, today)
            else:
                current = float(self._redis.get(DAILY_PNL_KEY) or 0.0)
                self._redis.set(DAILY_PNL_KEY, str(current + delta))
        except redis.RedisError as exc:
            logger.warning(f"Failed to update daily PnL: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_position(self, position: OpenPosition) -> None:
        self._positions[position.trade_id] = position
        self._persist_position(position)

    def remove_position(self, trade_id: str) -> Optional[OpenPosition]:
        position = self._positions.pop(trade_id, None)
        if position is not None:
            self._delete_position(trade_id)
        return position

    def resolve_trade(
        self,
        trade_id: str,
        outcome: int,
        resolved_at: float | None = None,
    ) -> Optional[ResolvedTrade]:
        position = self.remove_position(trade_id)
        if position is None:
            return None

        if position.direction == 1:
            # YES contract: paid fill_price_cents, collects 100¢ on win
            if outcome == 1:
                pnl = position.contracts * (100 - position.entry_price_cents) / 100
            else:
                pnl = -position.contracts * position.entry_price_cents / 100
        else:
            # NO contract: paid fill_price_cents (= 100 - YES_bid), collects 100¢ on win
            if outcome == 1:
                pnl = position.contracts * (100 - position.entry_price_cents) / 100
            else:
                pnl = -position.contracts * position.entry_price_cents / 100

        trade = ResolvedTrade(
            trade_id=position.trade_id,
            ticker=position.ticker,
            timeframe=position.timeframe,
            direction=position.direction,
            strike=position.strike,
            contracts=position.contracts,
            entry_price_cents=position.entry_price_cents,
            kelly_dollars=position.kelly_dollars,
            outcome=outcome,
            pnl_dollars=pnl,
            timestamp=position.timestamp,
            resolved_at=resolved_at if resolved_at is not None else time.time(),
        )

        self._persist_resolved(trade)
        self._update_daily_pnl(pnl)
        return trade

    def get_open_positions(self) -> list[OpenPosition]:
        return list(self._positions.values())

    def get_current_exposure(self) -> float:
        return sum(p.kelly_dollars for p in self._positions.values())

    def has_timeframe_position(self, timeframe: str) -> bool:
        return any(p.timeframe == timeframe for p in self._positions.values())

    def ticker_position_count(self, ticker: str) -> int:
        try:
            raw = self._redis.hgetall(OPEN_POSITIONS_KEY)
            return sum(1 for v in raw.values() if json.loads(v).get("ticker") == ticker)
        except redis.RedisError:
            return sum(1 for p in self._positions.values() if p.ticker == ticker)

    def ticker_direction_count(self, ticker: str, direction: int) -> int:
        try:
            raw = self._redis.hgetall(OPEN_POSITIONS_KEY)
            return sum(
                1 for v in raw.values()
                if (d := json.loads(v)).get("ticker") == ticker and d.get("direction") == direction
            )
        except redis.RedisError:
            return sum(
                1 for p in self._positions.values()
                if p.ticker == ticker and p.direction == direction
            )

    def get_daily_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            stored_date = self._redis.get(DAILY_PNL_DATE_KEY)
            if stored_date != today:
                self._redis.set(DAILY_PNL_KEY, "0.0")
                self._redis.set(DAILY_PNL_DATE_KEY, today)
                return 0.0
            return float(self._redis.get(DAILY_PNL_KEY) or 0.0)
        except redis.RedisError as exc:
            logger.warning(f"Failed to read daily PnL from Redis: {exc}")
            return 0.0

    def get_resolved_trades(self, limit: int = 50) -> list[ResolvedTrade]:
        try:
            raw = self._redis.lrange(RESOLVED_TRADES_KEY, 0, limit - 1)
            return [ResolvedTrade(**json.loads(r)) for r in raw]
        except redis.RedisError as exc:
            logger.warning(f"Failed to read resolved trades from Redis: {exc}")
            return []

    def get_trade_count(self) -> int:
        try:
            return self._redis.llen(RESOLVED_TRADES_KEY)
        except redis.RedisError as exc:
            logger.warning(f"Failed to read trade count from Redis: {exc}")
            return 0
