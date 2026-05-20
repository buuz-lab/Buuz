"""
DeepSeekContextParser — calls DeepSeek R1 to produce a structured market
regime tag every 15 minutes.

The parser takes a dict of market context (funding rate, OI delta,
liquidations, headlines, macro events) and returns a categorical regime
classification plus a `suppress_trading` flag. The downstream Signal Fusion
engine uses this as a hard gate, not as a probability — LLM outputs are
poorly calibrated for numeric prediction.

On any failure (network error, malformed response, missing API key) the
parser returns SAFE_DEFAULT: `suppress_trading=False`, regime
"high_uncertainty", confidence 0.0. Failures are not cached, so the next
call will retry the API.
"""

import json
import os
import time
from typing import Any

import requests
from loguru import logger

_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
_DEEPSEEK_MODEL = "deepseek-reasoner"
_DEFAULT_CACHE_MINUTES = 15
_HTTP_TIMEOUT_SECONDS = 30

_VALID_REGIMES = {"trending_up", "trending_down", "ranging", "high_uncertainty"}
_REQUIRED_KEYS = ("regime", "confidence", "suppress_trading", "suppress_reason", "notes")

# Used when DeepSeek is reachable and returns a valid regime with no special conditions.
NEUTRAL_DEFAULT: dict[str, Any] = {
    "regime": "ranging",
    "confidence": 0.0,
    "suppress_trading": False,
    "suppress_reason": None,
    "notes": "DeepSeek unavailable — using neutral fallback so signals are not shrunk.",
}

# Used when DeepSeek call fails in a way that suggests we should be cautious
# (e.g. a partial/garbled response, not just a network/billing error).
SAFE_DEFAULT: dict[str, Any] = {
    "regime": "high_uncertainty",
    "confidence": 0.0,
    "suppress_trading": False,
    "suppress_reason": "deepseek_unavailable",
    "notes": "Falling back to safe default — DeepSeek call failed or returned malformed data.",
}

_PROMPT_TEMPLATE = """You are a BTC market regime classifier. Given the following market context, output ONLY a JSON object with no preamble, no explanation, and no markdown fencing.

Context:
- Funding rate: {funding_rate}% (trend: {funding_trend})
- Open interest change (4h): {oi_delta}%
- Recent liquidations (1h): ${liquidations_usd}M
- Basis spread: {basis_spread}%
- Recent headlines: {headlines}
- Upcoming macro events (2h): {macro_events}

Output exactly this JSON structure:
{{
  "regime": "trending_up" | "trending_down" | "ranging" | "high_uncertainty",
  "confidence": 0.0-1.0,
  "suppress_trading": true | false,
  "suppress_reason": "string or null",
  "notes": "one sentence max"
}}"""


class DeepSeekContextParser:
    """Calls DeepSeek R1 with a structured prompt and caches the result."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_minutes: float = _DEFAULT_CACHE_MINUTES,
        model: str = _DEEPSEEK_MODEL,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", "")
        self._cache_minutes = cache_minutes
        self._model = model
        self._cache: dict | None = None
        self._cache_time: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_current_context(self, market_context: dict) -> dict:
        """Return a structured regime dict — cached for `cache_minutes` minutes."""
        if not self._api_key:
            # No key configured = intentional, not an error — use neutral so signals aren't shrunk.
            logger.debug("DeepSeekContextParser: no API key configured, using neutral fallback")
            return dict(NEUTRAL_DEFAULT)

        if self._is_cache_valid():
            return dict(self._cache)  # defensive copy

        try:
            prompt = self._build_prompt(market_context)
            raw_response = self._call_api(prompt)
        except requests.HTTPError as exc:
            # 402 = no credits.  Log once clearly; use neutral fallback (not high_uncertainty)
            # so signals are not shrunk during bootstrap while credits are topped up.
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 402:
                logger.warning(
                    "DeepSeekContextParser: 402 Payment Required — DeepSeek account has no credits. "
                    "Using neutral fallback (regime=ranging, suppress=False). "
                    "Top up at https://platform.deepseek.com to enable LLM context gating."
                )
            else:
                logger.warning(f"DeepSeekContextParser: HTTP error — {exc}")
                return dict(SAFE_DEFAULT)
            return dict(NEUTRAL_DEFAULT)
        except Exception as exc:
            # Unknown failure (network error, timeout, etc.) — stay conservative.
            logger.warning(f"DeepSeekContextParser: API call failed — {exc}")
            return dict(SAFE_DEFAULT)

        parsed = self._parse_response(raw_response)
        if parsed is None:
            logger.warning("DeepSeekContextParser: response failed validation, returning safe default")
            return dict(SAFE_DEFAULT)

        # Only cache successful parses — never cache safe defaults.
        self._cache = parsed
        self._cache_time = time.time()
        return dict(parsed)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _is_cache_valid(self) -> bool:
        if self._cache is None or self._cache_minutes <= 0:
            return False
        return (time.time() - self._cache_time) < (self._cache_minutes * 60)

    def _build_prompt(self, market_context: dict) -> str:
        return _PROMPT_TEMPLATE.format(
            funding_rate=market_context.get("funding_rate", "n/a"),
            funding_trend=market_context.get("funding_trend", "n/a"),
            oi_delta=market_context.get("oi_delta", "n/a"),
            liquidations_usd=market_context.get("liquidations_usd", "n/a"),
            basis_spread=market_context.get("basis_spread", "n/a"),
            headlines=market_context.get("headlines", []),
            macro_events=market_context.get("macro_events", []),
        )

    def _call_api(self, prompt: str) -> str:
        """POST to DeepSeek chat completions endpoint. Returns raw assistant content."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        response = requests.post(
            _DEEPSEEK_URL,
            headers=headers,
            json=payload,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str) -> dict | None:
        """Parse + validate response. Returns None if anything is off-spec."""
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(parsed, dict):
            return None

        for key in _REQUIRED_KEYS:
            if key not in parsed:
                return None

        if parsed["regime"] not in _VALID_REGIMES:
            return None

        try:
            confidence = float(parsed["confidence"])
        except (TypeError, ValueError):
            return None
        if not (0.0 <= confidence <= 1.0):
            return None

        if not isinstance(parsed["suppress_trading"], bool):
            return None

        return {
            "regime": parsed["regime"],
            "confidence": confidence,
            "suppress_trading": parsed["suppress_trading"],
            "suppress_reason": parsed.get("suppress_reason"),
            "notes": str(parsed.get("notes", "")),
        }
