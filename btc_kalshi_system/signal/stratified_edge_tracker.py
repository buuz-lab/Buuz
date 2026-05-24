"""
StratifiedEdgeTracker — per-regime rolling edge tracker.

Separate 50-trade deques keyed by deepseek_regime label. Allows detecting
regime-conditional edge collapse without replacing the global EdgeTracker.

Migration: build alongside EdgeTracker and confirm parity on identical data
before wiring into CircuitBreaker or Gate 4.

Redis keys: "stratified_edge:{regime}:history" — JSON list per regime.
Schema per entry: {"predicted_prob": float, "outcome": int, "market_price": float}

Realized edge per trade: outcome - market_price (same convention as EdgeTracker).
"""

import json
from collections import deque
from typing import Deque

import redis
from loguru import logger

from config import REDIS_URL

_MAX_HISTORY = 50
_DEFAULT_THRESHOLD = 0.005  # 0.5 cents per $1 contract notional


def _redis_key(regime: str) -> str:
    return f"stratified_edge:{regime}:history"


class StratifiedEdgeTracker:
    """
    Per-regime rolling edge tracker. Separate 50-trade deques keyed by
    deepseek_regime label. Allows detecting regime-conditional edge collapse.
    """

    REGIMES = ("trending_up", "trending_down", "ranging", "high_uncertainty")

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._threshold = threshold
        self._histories: dict[str, Deque[tuple[float, int, float]]] = {
            regime: deque(maxlen=_MAX_HISTORY) for regime in self.REGIMES
        }
        self._load_from_redis()

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(
        self,
        regime: str,
        predicted_prob: float,
        outcome: int,
        market_price: float,
    ) -> None:
        """Append a resolved trade for the given regime and persist to Redis.

        Unknown regime labels: log warning and skip (don't crash).
        """
        if regime not in self._histories:
            logger.warning(
                f"StratifiedEdgeTracker: unknown regime '{regime}' — skipping record"
            )
            return
        self._histories[regime].append(
            (float(predicted_prob), int(outcome), float(market_price))
        )
        self._persist_regime(regime)

    def current_edge(self, regime: str) -> float:
        """Mean realized edge for the given regime. 0.0 when empty."""
        history = self._histories.get(regime)
        if not history:
            return 0.0
        return float(
            sum(outcome - market_price for _, outcome, market_price in history)
            / len(history)
        )

    def is_above_threshold(self, regime: str) -> bool:
        """True iff current_edge(regime) >= threshold and len >= 1."""
        history = self._histories.get(regime)
        if not history:
            return False
        return self.current_edge(regime) >= self._threshold

    def summary(self) -> dict[str, float]:
        """Returns {regime: current_edge(regime)} for all REGIMES."""
        return {regime: self.current_edge(regime) for regime in self.REGIMES}

    # ── Redis I/O ──────────────────────────────────────────────────────────────

    def _persist_regime(self, regime: str) -> None:
        key = _redis_key(regime)
        payload = [
            {
                "predicted_prob": predicted_prob,
                "outcome": outcome,
                "market_price": market_price,
            }
            for (predicted_prob, outcome, market_price) in self._histories[regime]
        ]
        try:
            self._redis.set(key, json.dumps(payload))
        except redis.RedisError as exc:
            logger.warning(
                f"StratifiedEdgeTracker: failed to persist history for '{regime}' — {exc}"
            )

    def _load_from_redis(self) -> None:
        """Best-effort load of prior history for all regimes; silently start empty on failure."""
        for regime in self.REGIMES:
            key = _redis_key(regime)
            try:
                raw = self._redis.get(key)
            except redis.RedisError as exc:
                logger.warning(
                    f"StratifiedEdgeTracker: Redis unreachable loading '{regime}' — {exc}"
                )
                continue
            if raw is None:
                continue
            try:
                entries = json.loads(raw)
                for entry in entries:
                    self._histories[regime].append(
                        (
                            float(entry["predicted_prob"]),
                            int(entry["outcome"]),
                            float(entry["market_price"]),
                        )
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    f"StratifiedEdgeTracker: corrupt history for '{regime}' in Redis, "
                    f"starting empty — {exc}"
                )
                self._histories[regime].clear()
