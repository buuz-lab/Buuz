import time

import redis
from loguru import logger

from config import REDIS_URL

_WINDOW = 30
_KEY_NO  = "trading:win_history_no"
_KEY_YES = "trading:win_history_yes"


class DirectionWinRateTracker:
    """Per-direction rolling 30-trade win rate tracker backed by Redis sorted sets."""

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)

    def record(self, direction: int, outcome: int) -> None:
        key = _KEY_NO if direction == 0 else _KEY_YES
        ts = time.time()
        try:
            pipe = self._redis.pipeline()
            pipe.zadd(key, {f"{ts}:{outcome}": ts})
            pipe.zremrangebyrank(key, 0, -(_WINDOW + 1))
            pipe.execute()
        except redis.RedisError as exc:
            logger.warning(f"DirectionWinRateTracker: record failed — {exc}")

    def get_win_rate(self, direction: int) -> float | None:
        key = _KEY_NO if direction == 0 else _KEY_YES
        try:
            members = self._redis.zrange(key, 0, -1)
            if len(members) < 10:
                return None
            outcomes = [int(m.split(":")[-1]) for m in members]
            return sum(outcomes) / len(outcomes)
        except (redis.RedisError, ValueError) as exc:
            logger.warning(f"DirectionWinRateTracker: get failed — {exc}")
            return None
