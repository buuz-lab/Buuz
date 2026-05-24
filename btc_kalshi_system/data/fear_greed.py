"""Fear & Greed Index fetcher with Redis caching."""
import json

import requests
import redis
from loguru import logger

from config import REDIS_URL

_FNG_URL = "https://api.alternative.me/fng/?limit=1"
_REDIS_KEY = "market:fear_greed"
_CACHE_TTL = 3600        # 1 hour — index updates once daily
_HTTP_TIMEOUT = 10


def fetch_fear_greed(redis_client: redis.Redis) -> dict | None:
    """Return {value: int, label: str} from cache or live fetch. None on failure."""
    try:
        cached = redis_client.get(_REDIS_KEY)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        resp = requests.get(_FNG_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        result = {
            "value": int(data["value"]),
            "label": data["value_classification"],
        }
        try:
            redis_client.set(_REDIS_KEY, json.dumps(result), ex=_CACHE_TTL)
        except Exception:
            pass
        return result
    except Exception as exc:
        logger.warning(f"FearGreedFetcher: fetch failed — {exc}")
        return None
