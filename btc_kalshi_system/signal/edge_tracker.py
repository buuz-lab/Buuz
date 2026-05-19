"""
EdgeTracker — rolling realized-edge tracker for circuit-breaker logic.

For each resolved Kalshi trade we log (predicted_prob, outcome, market_price).
Realized edge per trade is `outcome - market_price` (dollar EV per $1 of
contract notional). Rolling realized edge = mean over the last 50 trades.

This is *not* the predicted edge (`predicted_prob - market_price`). It's the
realized one — used to detect model failure. The pre-trade checklist gates
new orders on `is_above_threshold()`; the circuit breaker halts trading when
this goes negative.

State is persisted to Redis key "edge_tracker:history" as a JSON list so the
history survives process restarts.
"""

import json
from collections import deque
from typing import Deque

import redis
from loguru import logger

from config import REDIS_URL

_REDIS_KEY = "edge_tracker:history"
_MAX_HISTORY = 50
_DEFAULT_THRESHOLD = 0.005  # 0.5 cents per $1 contract notional


class EdgeTracker:
    """Rolling deque of (predicted_prob, outcome, market_price) tuples."""

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._redis = redis.from_url(redis_url)
        self._threshold = threshold
        self._history: Deque[tuple[float, int, float]] = deque(maxlen=_MAX_HISTORY)
        self._load_from_redis()

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self, predicted_prob: float, outcome: int, market_price: float) -> None:
        """Append a resolved trade to history and persist to Redis."""
        self._history.append((float(predicted_prob), int(outcome), float(market_price)))
        self._persist_to_redis()

    def current_edge(self) -> float:
        """Mean realized edge across recorded trades. 0.0 when empty."""
        if not self._history:
            return 0.0
        return float(
            sum(outcome - market_price for _, outcome, market_price in self._history)
            / len(self._history)
        )

    def is_above_threshold(self) -> bool:
        """True iff current_edge() >= threshold. False when history is empty."""
        if not self._history:
            return False
        return self.current_edge() >= self._threshold

    def __len__(self) -> int:
        return len(self._history)

    # ── Redis I/O ──────────────────────────────────────────────────────────────

    def _persist_to_redis(self) -> None:
        payload = [
            {
                "predicted_prob": predicted_prob,
                "outcome": outcome,
                "market_price": market_price,
            }
            for (predicted_prob, outcome, market_price) in self._history
        ]
        try:
            self._redis.set(_REDIS_KEY, json.dumps(payload))
        except redis.RedisError as exc:
            logger.warning(f"EdgeTracker: failed to persist history — {exc}")

    def _load_from_redis(self) -> None:
        """Best-effort load of prior history; silently start empty on any failure."""
        try:
            raw = self._redis.get(_REDIS_KEY)
        except redis.RedisError as exc:
            logger.warning(f"EdgeTracker: Redis unreachable on load — {exc}")
            return
        if raw is None:
            return
        try:
            entries = json.loads(raw)
            for entry in entries:
                self._history.append(
                    (
                        float(entry["predicted_prob"]),
                        int(entry["outcome"]),
                        float(entry["market_price"]),
                    )
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"EdgeTracker: corrupt history in Redis, starting empty — {exc}")
            self._history.clear()
