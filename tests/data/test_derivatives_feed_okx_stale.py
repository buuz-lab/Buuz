"""Tests for OKX partial-flag behaviour in DerivativesFeed._write_features()."""
import json
import time

import fakeredis
import pytest
from unittest.mock import AsyncMock, MagicMock

from btc_kalshi_system.data.derivatives_feed import DerivativesFeed


def make_feed() -> DerivativesFeed:
    feed = DerivativesFeed.__new__(DerivativesFeed)
    feed._redis = fakeredis.FakeRedis()
    feed._exchange = MagicMock()
    feed._kraken_exchange = None
    feed._ccxt_async = MagicMock()
    feed._prev_oi = 0.0
    return feed


def _base_features() -> dict:
    return {
        "funding_rate": 0.0001,
        "funding_rate_trend": 0.00002,
        "oi_delta_pct": 0.001,
        "cvd_normalized": 0.3,
        "basis_spread_pct": 0.0005,
        "brti_volatility_1h": 0.001,
        "large_print_direction": 0.5,
        "volume_ratio_1h": 1.0,
        "fear_greed_value": 30,
        "fear_greed_label": "Fear",
    }


# ── test_lkg_not_updated_on_partial ──────────────────────────────────────────

def test_lkg_not_updated_on_partial():
    """When okx_partial=True, regime:features:lkg must not be overwritten."""
    feed = make_feed()
    sentinel = json.dumps({"funding_rate": 9.99, "_lkg_written_at": 1234567890.0})
    feed._redis.set("regime:features:lkg", sentinel, ex=86400)

    feed._write_features(_base_features(), okx_partial=True)

    assert feed._redis.get("regime:features:lkg").decode() == sentinel


def test_partial_flag_embedded_in_primary_key():
    """When okx_partial=True, regime:features must contain _okx_partial=true."""
    feed = make_feed()
    feed._write_features(_base_features(), okx_partial=True)

    raw = feed._redis.get("regime:features")
    assert raw is not None
    payload = json.loads(raw)
    assert payload.get("_okx_partial") is True


# ── test_lkg_updated_on_success ───────────────────────────────────────────────

def test_lkg_updated_on_success():
    """When okx_partial=False, regime:features:lkg IS written and has no _okx_partial."""
    feed = make_feed()
    feed._write_features(_base_features(), okx_partial=False)

    raw = feed._redis.get("regime:features:lkg")
    assert raw is not None
    payload = json.loads(raw)
    assert "_okx_partial" not in payload
    assert "_lkg_written_at" in payload
