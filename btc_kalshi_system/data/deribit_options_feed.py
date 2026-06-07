"""
DeribitOptionsFeed — polls Deribit's public options chain REST API every 5
minutes and writes 4 regime features to Redis.

Keys written:
  options:features      JSON dict {atm_iv, pcr_oi, term_structure_slope, skew_25d}
                        TTL: 600s (2× refresh interval)
  options:features:lkg  same dict + _lkg_written_at timestamp
                        TTL: 14400s (4 hours)

All failures are logged and retried — never raises out of run().
"""

import asyncio
import json
import math
import time
from datetime import datetime, timezone, timedelta

import aiohttp
import redis
from loguru import logger

from config import REDIS_URL

_REFRESH_INTERVAL = 300          # 5 minutes
_OPTIONS_TTL = 600               # 2× refresh interval
_OPTIONS_LKG_TTL = 14_400        # 4 hours — options data moves slowly
_MIN_DAYS_TO_EXPIRY = 1          # skip same-day expiry only; 3 was too aggressive near weekly rolls
_MIN_OI_FOR_ATM = 10             # minimum OI for ATM IV candidates
_DERIBIT_URL = (
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    "?currency=BTC&kind=option"
)


class DeribitOptionsFeed:
    """
    Stateless async feed. Uses a fresh aiohttp.ClientSession per fetch call
    (REST is stateless — no reconnect complexity needed).
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._redis = redis.from_url(redis_url)
        self._prev_pcr_oi: float = 1.0    # neutral default (ratio, 1.0 = balanced)
        self._prev_skew_25d: float = 0.0  # neutral default

    # ── Public entry point ─────────────────────────────────────────────────────

    async def run(self) -> None:
        while True:
            success = False
            try:
                features = await self._fetch_features()
                self._write_features(features)
                logger.info(f"DeribitOptionsFeed: wrote options:features — {features}")
                success = True
            except Exception as exc:
                logger.warning(f"DeribitOptionsFeed: fetch failed — {exc}")
            # Refresh 60s early on success so the key never expires between writes.
            # On failure, wait the full interval before retrying.
            await asyncio.sleep(_REFRESH_INTERVAL - 60 if success else _REFRESH_INTERVAL)

    # ── Fetch ──────────────────────────────────────────────────────────────────

    async def _fetch_features(self) -> dict:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(_DERIBIT_URL) as resp:
                resp.raise_for_status()
                data = await resp.json()
        instruments = data.get("result", [])
        return self._compute_features(instruments)

    # ── Feature computation ────────────────────────────────────────────────────

    def _compute_features(self, instruments: list) -> dict:
        """Compute all 4 options features from the raw chain response."""
        try:
            expiries = self._group_by_expiry(instruments)
            sorted_expiries = sorted(expiries.keys())

            if not sorted_expiries:
                return {"atm_iv": None, "pcr_oi": 1.0, "term_structure_slope": 0.0, "skew_25d": 0.0,
                        "pcr_delta": 0.0, "skew_delta": 0.0}

            near_expiry = sorted_expiries[0]
            far_expiry = sorted_expiries[1] if len(sorted_expiries) >= 2 else None
            near_instruments = expiries[near_expiry]
            near_days = max(0.001, (near_expiry - datetime.now(timezone.utc)).total_seconds() / 86400)

            underlying_price = self._get_underlying_price(near_instruments)
            if underlying_price is None:
                return {"atm_iv": None, "pcr_oi": 1.0, "term_structure_slope": 0.0, "skew_25d": 0.0,
                        "pcr_delta": 0.0, "skew_delta": 0.0}

            # atm_iv — near expiry
            near_iv = self._compute_atm_iv(near_instruments, underlying_price)

            # pcr_oi
            near_puts = [i for i in near_instruments if i.get("_type") == "P"]
            near_calls = [i for i in near_instruments if i.get("_type") == "C"]
            put_oi = sum(float(i.get("open_interest") or 0) for i in near_puts)
            call_oi = sum(float(i.get("open_interest") or 0) for i in near_calls)
            pcr_oi = put_oi / call_oi if call_oi > 0 else 1.0

            # term_structure_slope
            if far_expiry is not None:
                far_instruments = expiries[far_expiry]
                far_underlying = self._get_underlying_price(far_instruments) or underlying_price
                far_iv = self._compute_atm_iv(far_instruments, far_underlying)
                if near_iv is not None and far_iv is not None and near_iv > 0:
                    term_structure_slope = (far_iv - near_iv) / near_iv
                else:
                    term_structure_slope = 0.0
            else:
                term_structure_slope = 0.0

            # skew_25d
            if near_iv is not None:
                T = near_days / 365.0
                skew_25d = self._compute_skew_25d(near_instruments, underlying_price, near_iv, T)
            else:
                skew_25d = 0.0

            pcr_delta = pcr_oi - self._prev_pcr_oi
            skew_delta = skew_25d - self._prev_skew_25d

            result = {
                "atm_iv": near_iv,
                "pcr_oi": pcr_oi,
                "term_structure_slope": term_structure_slope,
                "skew_25d": skew_25d,
                "pcr_delta": pcr_delta,
                "skew_delta": skew_delta,
            }
            self._prev_pcr_oi = pcr_oi
            self._prev_skew_25d = skew_25d
            return result
        except Exception as exc:
            logger.warning(f"DeribitOptionsFeed: feature computation failed — {exc}")
            return {"atm_iv": None, "pcr_oi": 1.0, "term_structure_slope": 0.0, "skew_25d": 0.0,
                    "pcr_delta": 0.0, "skew_delta": 0.0}

    def _group_by_expiry(self, instruments: list) -> dict:
        """Group parsed instruments by expiry datetime. Skip expiries < 3 days out."""
        now = datetime.now(timezone.utc)
        groups: dict[datetime, list] = {}
        for inst in instruments:
            name = inst.get("instrument_name", "")
            parsed = self._parse_instrument(name)
            if parsed is None:
                continue
            expiry, itype, strike = parsed
            days_to_expiry = (expiry - now).total_seconds() / 86400
            if days_to_expiry < _MIN_DAYS_TO_EXPIRY:
                continue
            if expiry not in groups:
                groups[expiry] = []
            enriched = dict(inst)
            enriched["_expiry"] = expiry
            enriched["_type"] = itype
            enriched["_strike"] = strike
            groups[expiry].append(enriched)
        return groups

    def _parse_instrument(self, name: str):
        """Parse BTC-DDMMMYY-STRIKE-TYPE. Returns (expiry_dt, type, strike) or None."""
        try:
            parts = name.split("-")
            if len(parts) != 4 or parts[0] != "BTC":
                return None
            expiry_str = parts[1]
            strike = float(parts[2])
            itype = parts[3]
            if itype not in ("C", "P"):
                return None
            expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=timezone.utc)
            return expiry_dt, itype, strike
        except (ValueError, IndexError):
            return None

    def _get_underlying_price(self, instruments: list) -> float | None:
        for inst in instruments:
            v = inst.get("underlying_price")
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    def _compute_atm_iv(self, instruments: list, underlying_price: float) -> float | None:
        """Interpolate ATM IV from bracketing call strikes. Requires OI >= 10."""
        calls = [
            i for i in instruments
            if i.get("_type") == "C"
            and float(i.get("open_interest") or 0) >= _MIN_OI_FOR_ATM
            and i.get("mark_iv") is not None
        ]
        if not calls:
            return None
        calls_sorted = sorted(calls, key=lambda x: x["_strike"])

        below = [c for c in calls_sorted if c["_strike"] <= underlying_price]
        above = [c for c in calls_sorted if c["_strike"] > underlying_price]

        if below and above:
            lower = below[-1]
            upper = above[0]
            s_low = lower["_strike"]
            s_high = upper["_strike"]
            iv_low = float(lower["mark_iv"])
            iv_high = float(upper["mark_iv"])
            if s_high == s_low:
                return (iv_low + iv_high) / 2.0
            # Linear interpolation: weight of upper increases as spot approaches s_high
            w_upper = (underlying_price - s_low) / (s_high - s_low)
            return iv_low * (1.0 - w_upper) + iv_high * w_upper
        elif below:
            return float(below[-1]["mark_iv"])
        elif above:
            return float(above[0]["mark_iv"])
        return None

    def _compute_skew_25d(
        self, instruments: list, underlying_price: float, atm_iv: float, T: float
    ) -> float:
        """25-delta skew = put_iv - call_iv. Returns 0.0 on any failure."""
        try:
            put_strike = underlying_price * (1.0 - 0.25 * (atm_iv / 100.0) * math.sqrt(T))
            call_strike = underlying_price * (1.0 + 0.25 * (atm_iv / 100.0) * math.sqrt(T))
            tolerance = 0.05 * underlying_price

            puts = [i for i in instruments if i.get("_type") == "P" and i.get("mark_iv") is not None]
            calls_list = [i for i in instruments if i.get("_type") == "C" and i.get("mark_iv") is not None]

            if not puts or not calls_list:
                return 0.0

            nearest_put = min(puts, key=lambda x: abs(x["_strike"] - put_strike))
            nearest_call = min(calls_list, key=lambda x: abs(x["_strike"] - call_strike))

            if abs(nearest_put["_strike"] - put_strike) > tolerance:
                return 0.0
            if abs(nearest_call["_strike"] - call_strike) > tolerance:
                return 0.0

            return float(nearest_put["mark_iv"]) - float(nearest_call["mark_iv"])
        except Exception:
            return 0.0

    # ── Redis write ────────────────────────────────────────────────────────────

    def _write_features(self, features: dict) -> None:
        serialized = json.dumps(features)
        self._redis.set("options:features", serialized, ex=_OPTIONS_TTL)
        lkg = dict(features)
        lkg["_lkg_written_at"] = time.time()
        self._redis.set("options:features:lkg", json.dumps(lkg), ex=_OPTIONS_LKG_TTL)
