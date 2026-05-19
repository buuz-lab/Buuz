import json
from unittest.mock import patch

import pytest

from btc_kalshi_system.models.deepseek_parser import (
    DeepSeekContextParser,
    SAFE_DEFAULT,
)


def _good_context() -> dict:
    """Synthetic market context the parser would receive."""
    return {
        "funding_rate": 0.012,
        "funding_trend": 0.002,
        "oi_delta": 0.05,
        "liquidations_usd": 1.2,
        "basis_spread": -0.001,
        "headlines": ["BTC ETF inflow $300M", "Fed minutes due Wed"],
        "macro_events": ["FOMC 2pm Wednesday"],
    }


def _good_response() -> str:
    return json.dumps({
        "regime": "trending_up",
        "confidence": 0.72,
        "suppress_trading": False,
        "suppress_reason": None,
        "notes": "Strong ETF inflow with positive funding.",
    })


# ── Successful parsing ─────────────────────────────────────────────────────────

def test_get_current_context_returns_parsed_dict():
    parser = DeepSeekContextParser(api_key="test-key")
    with patch.object(parser, "_call_api", return_value=_good_response()):
        result = parser.get_current_context(_good_context())
    assert result["regime"] == "trending_up"
    assert result["confidence"] == pytest.approx(0.72)
    assert result["suppress_trading"] is False


def test_returned_dict_has_all_required_keys():
    parser = DeepSeekContextParser(api_key="test-key")
    with patch.object(parser, "_call_api", return_value=_good_response()):
        result = parser.get_current_context(_good_context())
    for key in ("regime", "confidence", "suppress_trading", "suppress_reason", "notes"):
        assert key in result


# ── Caching (15-minute window) ────────────────────────────────────────────────

def test_second_call_within_window_uses_cache_not_api():
    parser = DeepSeekContextParser(api_key="test-key", cache_minutes=15)
    with patch.object(parser, "_call_api", return_value=_good_response()) as mock_api:
        parser.get_current_context(_good_context())
        parser.get_current_context(_good_context())
        parser.get_current_context(_good_context())
    assert mock_api.call_count == 1  # Only first call hit the API


def test_call_after_cache_expiry_refreshes():
    """When cache_minutes elapses, parser should re-call the API."""
    parser = DeepSeekContextParser(api_key="test-key", cache_minutes=15)
    with patch.object(parser, "_call_api", return_value=_good_response()) as mock_api:
        parser.get_current_context(_good_context())
        # Force cache expiry by rewinding the stored timestamp
        parser._cache_time -= 60 * 16  # 16 minutes ago
        parser.get_current_context(_good_context())
    assert mock_api.call_count == 2


def test_cache_disabled_when_zero_minutes():
    """cache_minutes=0 means every call hits the API."""
    parser = DeepSeekContextParser(api_key="test-key", cache_minutes=0)
    with patch.object(parser, "_call_api", return_value=_good_response()) as mock_api:
        parser.get_current_context(_good_context())
        parser.get_current_context(_good_context())
    assert mock_api.call_count == 2


# ── Failure handling ───────────────────────────────────────────────────────────

def test_api_exception_returns_safe_default():
    parser = DeepSeekContextParser(api_key="test-key")
    with patch.object(parser, "_call_api", side_effect=RuntimeError("network down")):
        result = parser.get_current_context(_good_context())
    assert result == SAFE_DEFAULT
    assert result["suppress_trading"] is False
    assert result["regime"] == "high_uncertainty"
    assert result["confidence"] == pytest.approx(0.0)


def test_malformed_json_response_returns_safe_default():
    parser = DeepSeekContextParser(api_key="test-key")
    with patch.object(parser, "_call_api", return_value="this is not valid json"):
        result = parser.get_current_context(_good_context())
    assert result == SAFE_DEFAULT


def test_partial_json_response_returns_safe_default():
    """Missing required keys → safe default."""
    parser = DeepSeekContextParser(api_key="test-key")
    incomplete = json.dumps({"regime": "trending_up"})  # missing confidence, etc.
    with patch.object(parser, "_call_api", return_value=incomplete):
        result = parser.get_current_context(_good_context())
    assert result == SAFE_DEFAULT


def test_invalid_regime_label_returns_safe_default():
    """Regime must be one of the four allowed labels."""
    parser = DeepSeekContextParser(api_key="test-key")
    bad = json.dumps({
        "regime": "moonshot",  # not a valid label
        "confidence": 0.5,
        "suppress_trading": False,
        "suppress_reason": None,
        "notes": "x",
    })
    with patch.object(parser, "_call_api", return_value=bad):
        result = parser.get_current_context(_good_context())
    assert result == SAFE_DEFAULT


def test_safe_default_not_cached():
    """A failed call should not poison the cache — next call should retry."""
    parser = DeepSeekContextParser(api_key="test-key", cache_minutes=15)
    with patch.object(parser, "_call_api", side_effect=RuntimeError("fail")) as mock_api:
        parser.get_current_context(_good_context())
        parser.get_current_context(_good_context())
    assert mock_api.call_count == 2  # Both calls retried — failure not cached


# ── Prompt building ───────────────────────────────────────────────────────────

def test_prompt_includes_market_context_values():
    parser = DeepSeekContextParser(api_key="test-key")
    prompt = parser._build_prompt(_good_context())
    assert "0.012" in prompt or "1.2" in prompt  # at least one numeric value passed through
    assert "BTC ETF inflow $300M" in prompt
    assert "FOMC 2pm Wednesday" in prompt


# ── Construction ──────────────────────────────────────────────────────────────

def test_missing_api_key_falls_back_to_safe_default():
    """Without an API key, every call returns safe default (cannot authenticate)."""
    parser = DeepSeekContextParser(api_key="")
    result = parser.get_current_context(_good_context())
    assert result == SAFE_DEFAULT
