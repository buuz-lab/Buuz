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
_DEEPSEEK_MODEL = "deepseek-chat"
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

_PROMPT_TEMPLATE = """You are a BTC market regime classifier for an automated
15-minute prediction market trading system. Given the market snapshot below,
output ONLY a JSON object with no preamble, no explanation, no markdown.

Market snapshot — {utc_time} UTC ({session}):

PRICE & MOMENTUM
- BTC price: {btc_price}
- 5-min momentum: {momentum_5min}
- 15-min momentum: {momentum_15min}
- 1h trend: slope={trend_slope}, fit R²={trend_r2} (1.0=perfect trend, 0.0=noise)
- Range: {range_breakout} (+1=breaking up, -1=breaking down, 0=ranging/inside)

DERIVATIVES
- Funding rate: {funding_rate}% (4h trend: {funding_trend}%)
- Open interest change (4h): {oi_delta}%
- Basis spread: {basis_spread}%
- CVD (buy/sell pressure): {cvd} (-1.0=heavy selling, +1.0=heavy buying)
- CVD velocity: {cvd_velocity} (rate of change)
- Large-print direction: {large_print} (-1.0=whale selling, +1.0=whale buying)
- 1h volatility: {volatility}

SENTIMENT & POSITIONING
- Fear & Greed Index: {fear_greed}
- Kalshi implied probability: {kalshi_prob} (prediction market's UP probability)
- Volume vs 30-day avg: {volume_ratio}

RECENT KALSHI OUTCOMES (last 5 resolved 15-min markets):
{recent_outcomes}

IMPORTANT suppress_trading rules:
- Set suppress_trading=false for ALL normal conditions including ranging,
  low-volatility, high-uncertainty, or thin data. This system trades in
  these conditions by design.
- Set suppress_trading=true ONLY for extraordinary imminent events: active
  exchange hacks, ongoing flash crashes, FOMC announcement within 30 minutes,
  confirmed major exploits. One-in-a-month events only.

Output exactly this JSON:
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
        from datetime import datetime, timezone

        utc_now = datetime.now(timezone.utc)
        hour = utc_now.hour
        if 0 <= hour < 8:
            session = "Asian session"
        elif 8 <= hour < 13:
            session = "London session"
        elif 13 <= hour < 17:
            session = "NY session"
        else:
            session = "NY close / off-hours"
        utc_time = utc_now.strftime("%H:%M")

        def fmt_pct(ctx, key):
            val = ctx.get(key)
            if val is None or val == "n/a":
                return "n/a"
            try:
                return f"{float(val):+.3%}"
            except (TypeError, ValueError):
                return "n/a"

        def fmt_f(key, decimals=4):
            val = market_context.get(key)
            if val is None or val == "n/a":
                return "n/a"
            try:
                return f"{float(val):.{decimals}f}"
            except (TypeError, ValueError):
                return "n/a"

        btc_raw = market_context.get("composite_price")
        btc_price = f"${float(btc_raw):,.0f}" if btc_raw else "n/a"

        range_raw = market_context.get("range_breakout_flag")
        if range_raw is None or range_raw == "n/a":
            range_breakout = "n/a"
        else:
            v = float(range_raw)
            range_breakout = "+1 (breaking up)" if v > 0.5 else (
                "-1 (breaking down)" if v < -0.5 else "0 (ranging)"
            )

        fg = market_context.get("fear_greed")
        if fg and isinstance(fg, dict):
            fear_greed = f"{fg.get('value', 'n/a')} ({fg.get('label', 'n/a')})"
        else:
            fear_greed = "n/a"

        vr = market_context.get("volume_ratio_1h")
        volume_ratio = f"{float(vr):.2f}x 30-day avg" if vr else "n/a"

        kp = market_context.get("kalshi_implied_prob")
        kalshi_prob = f"{float(kp):.0%}" if kp else "n/a"

        recent = market_context.get("recent_outcomes", [])
        if recent:
            labels = ["UP" if o == 1 else "DOWN" for o in recent]
            recent_outcomes = " → ".join(labels) + f" ({sum(recent)}/{len(recent)} UP)"
        else:
            recent_outcomes = "n/a (insufficient history)"

        return _PROMPT_TEMPLATE.format(
            utc_time=utc_time,
            session=session,
            btc_price=btc_price,
            momentum_5min=fmt_pct(market_context, "brti_momentum_5min"),
            momentum_15min=fmt_pct(market_context, "brti_momentum_15min"),
            trend_slope=fmt_f("trend_slope_1h"),
            trend_r2=fmt_f("trend_r2_1h", decimals=2),
            range_breakout=range_breakout,
            funding_rate=fmt_f("funding_rate"),
            funding_trend=fmt_f("funding_rate_trend"),
            oi_delta=fmt_f("oi_delta_pct", decimals=2),
            basis_spread=fmt_f("basis_spread_pct"),
            cvd=fmt_f("cvd_normalized", decimals=3),
            cvd_velocity=fmt_f("cvd_velocity", decimals=4),
            large_print=fmt_f("large_print_direction", decimals=2),
            volatility=fmt_f("brti_volatility_1h"),
            fear_greed=fear_greed,
            kalshi_prob=kalshi_prob,
            volume_ratio=volume_ratio,
            recent_outcomes=recent_outcomes,
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
            "response_format": {"type": "json_object"},
            "max_tokens": 400,
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
