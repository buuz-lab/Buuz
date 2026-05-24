import json
from unittest.mock import patch, MagicMock

import fakeredis
import pytest

from btc_kalshi_system.data.fear_greed import fetch_fear_greed, _REDIS_KEY


def _make_redis():
    return fakeredis.FakeRedis()


def test_fetch_returns_cached_value():
    r = _make_redis()
    cached = {"value": 55, "label": "Greed"}
    r.set(_REDIS_KEY, json.dumps(cached))

    with patch("requests.get") as mock_get:
        result = fetch_fear_greed(r)

    assert result == cached
    mock_get.assert_not_called()


def test_fetch_from_api_and_caches():
    r = _make_redis()
    api_payload = {
        "data": [{"value": "72", "value_classification": "Greed"}]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_payload

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = fetch_fear_greed(r)

    assert result == {"value": 72, "label": "Greed"}
    mock_get.assert_called_once()

    # Verify cached in Redis
    cached_raw = r.get(_REDIS_KEY)
    assert cached_raw is not None
    assert json.loads(cached_raw) == {"value": 72, "label": "Greed"}


def test_fetch_returns_none_on_failure():
    r = _make_redis()

    with patch("requests.get", side_effect=RuntimeError("network down")):
        result = fetch_fear_greed(r)

    assert result is None
