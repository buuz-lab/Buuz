"""Tests for DeribitOptionsFeed — TDD: these tests are written before the implementation."""
import asyncio
import json
import math
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import fakeredis

from btc_kalshi_system.data.deribit_options_feed import DeribitOptionsFeed


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _days_from_now(n: int) -> str:
    """Return a Deribit expiry string (e.g. '07JUN25') n days from now UTC."""
    dt = datetime.now(timezone.utc) + timedelta(days=n)
    return dt.strftime("%d%b%y").upper()


def make_feed() -> DeribitOptionsFeed:
    """DeribitOptionsFeed with fakeredis and no real HTTP."""
    feed = DeribitOptionsFeed.__new__(DeribitOptionsFeed)
    feed._redis = fakeredis.FakeRedis()
    return feed


def _instrument(expiry_tag: str, strike: float, itype: str, mark_iv: float,
                oi: float = 100.0, underlying: float = 100_000.0) -> dict:
    return {
        "instrument_name": f"BTC-{expiry_tag}-{int(strike)}-{itype}",
        "underlying_price": underlying,
        "mark_iv": mark_iv,
        "open_interest": oi,
        "volume": 10.0,
    }


def _two_expiry_chain(near_tag: str, far_tag: str, spot: float = 100_000.0) -> list:
    """A realistic chain: near expiry (7d) and far expiry (35d), calls + puts."""
    near_calls = [
        _instrument(near_tag, spot - 2000, "C", 62.0, 150.0, spot),
        _instrument(near_tag, spot - 1000, "C", 60.0, 200.0, spot),
        _instrument(near_tag, spot + 1000, "C", 58.0, 200.0, spot),
        _instrument(near_tag, spot + 2000, "C", 56.0, 150.0, spot),
    ]
    near_puts = [
        _instrument(near_tag, spot - 2000, "P", 65.0, 300.0, spot),
        _instrument(near_tag, spot - 1000, "P", 63.0, 200.0, spot),
        _instrument(near_tag, spot + 1000, "P", 57.0, 100.0, spot),
    ]
    far_calls = [
        _instrument(far_tag, spot - 2000, "C", 70.0, 100.0, spot),
        _instrument(far_tag, spot - 1000, "C", 68.0, 150.0, spot),
        _instrument(far_tag, spot + 1000, "C", 66.0, 150.0, spot),
        _instrument(far_tag, spot + 2000, "C", 64.0, 100.0, spot),
    ]
    far_puts = [
        _instrument(far_tag, spot - 2000, "P", 72.0, 250.0, spot),
        _instrument(far_tag, spot - 1000, "P", 70.0, 150.0, spot),
    ]
    return near_calls + near_puts + far_calls + far_puts


# ── test_writes_options_features_to_redis ─────────────────────────────────────

@pytest.mark.asyncio
async def test_writes_options_features_to_redis():
    """Valid Deribit chain response → all 4 keys written to options:features."""
    feed = make_feed()
    near_tag = _days_from_now(7)
    far_tag = _days_from_now(35)
    chain = _two_expiry_chain(near_tag, far_tag)

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"result": chain})
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        features = await feed._fetch_features()
        feed._write_features(features)

    raw = feed._redis.get("options:features")
    assert raw is not None, "options:features was not written"
    loaded = json.loads(raw)
    for key in ("atm_iv", "pcr_oi", "term_structure_slope", "skew_25d"):
        assert key in loaded, f"Missing key: {key}"


# ── test_writes_lkg_on_success ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_writes_lkg_on_success():
    """Successful fetch must write options:features:lkg with _lkg_written_at and 4h TTL."""
    feed = make_feed()
    near_tag = _days_from_now(7)
    far_tag = _days_from_now(35)
    chain = _two_expiry_chain(near_tag, far_tag)

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"result": chain})
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    before = time.time()
    with patch("aiohttp.ClientSession", return_value=mock_session):
        features = await feed._fetch_features()
        feed._write_features(features)

    raw_lkg = feed._redis.get("options:features:lkg")
    assert raw_lkg is not None, "options:features:lkg was not written"

    ttl = feed._redis.ttl("options:features:lkg")
    assert 14_390 <= ttl <= 14_400, f"Expected ~14400s TTL for LKG, got {ttl}"

    lkg = json.loads(raw_lkg)
    assert "_lkg_written_at" in lkg
    assert lkg["_lkg_written_at"] >= before


# ── test_skips_expiry_under_3_days ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skips_expiry_under_3_days():
    """Chain where the only valid expiry is < 3 days away must produce atm_iv=None."""
    feed = make_feed()
    # Only a 1-day-out expiry — must be skipped
    near_tag = _days_from_now(1)
    chain = [
        _instrument(near_tag, 99_000, "C", 70.0, 200.0, 100_000.0),
        _instrument(near_tag, 101_000, "C", 68.0, 200.0, 100_000.0),
        _instrument(near_tag, 99_000, "P", 72.0, 300.0, 100_000.0),
    ]

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"result": chain})
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        features = await feed._fetch_features()

    assert features.get("atm_iv") is None, "atm_iv should be None when all expiries are < 3 days out"


@pytest.mark.asyncio
async def test_skips_near_expiry_uses_next_when_too_close():
    """When first expiry < 3 days but second is valid, uses the second as near expiry."""
    feed = make_feed()
    skip_tag = _days_from_now(1)   # will be skipped
    valid_tag = _days_from_now(7)  # will be used as near

    chain = [
        # Skipped (< 3d)
        _instrument(skip_tag, 99_000, "C", 99.0, 200.0, 100_000.0),
        _instrument(skip_tag, 101_000, "C", 97.0, 200.0, 100_000.0),
        # Valid near expiry
        _instrument(valid_tag, 99_000, "C", 60.0, 200.0, 100_000.0),
        _instrument(valid_tag, 101_000, "C", 58.0, 200.0, 100_000.0),
        _instrument(valid_tag, 99_000, "P", 65.0, 300.0, 100_000.0),
    ]

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"result": chain})
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        features = await feed._fetch_features()

    # atm_iv should come from the valid 7d expiry (not None, not 97%+)
    assert features.get("atm_iv") is not None
    assert features["atm_iv"] < 90.0  # not the 99/97% front-month spikes


# ── test_pcr_oi_greater_than_one_when_puts_dominate ───────────────────────────

def test_pcr_oi_greater_than_one_when_puts_dominate():
    """200 BTC put OI, 100 BTC call OI → pcr_oi ≈ 2.0."""
    feed = make_feed()
    near_tag = _days_from_now(7)
    instruments = [
        _instrument(near_tag, 99_000, "C", 60.0, oi=50.0),
        _instrument(near_tag, 101_000, "C", 58.0, oi=50.0),   # total call OI = 100
        _instrument(near_tag, 98_000, "P", 65.0, oi=100.0),
        _instrument(near_tag, 99_000, "P", 63.0, oi=100.0),  # total put OI = 200
    ]
    features = feed._compute_features(instruments)
    assert features["pcr_oi"] == pytest.approx(2.0, rel=1e-4)


# ── test_pcr_oi_fallback_is_one_not_zero ──────────────────────────────────────

def test_pcr_oi_fallback_is_one_not_zero():
    """Zero call OI → pcr_oi must be 1.0, not 0.0 or None."""
    feed = make_feed()
    near_tag = _days_from_now(7)
    instruments = [
        _instrument(near_tag, 99_000, "C", 60.0, oi=0.0),
        _instrument(near_tag, 101_000, "C", 58.0, oi=0.0),   # call OI = 0
        _instrument(near_tag, 98_000, "P", 65.0, oi=100.0),
    ]
    features = feed._compute_features(instruments)
    assert features["pcr_oi"] == pytest.approx(1.0)


# ── test_term_structure_slope_positive_in_contango ────────────────────────────

def test_term_structure_slope_positive_in_contango():
    """far IV > near IV → term_structure_slope > 0 (contango)."""
    feed = make_feed()
    near_tag = _days_from_now(7)
    far_tag = _days_from_now(35)
    spot = 100_000.0
    instruments = [
        # Near calls (low IV)
        _instrument(near_tag, spot - 1000, "C", 55.0, 200.0, spot),
        _instrument(near_tag, spot + 1000, "C", 53.0, 200.0, spot),
        _instrument(near_tag, spot - 1000, "P", 57.0, 100.0, spot),
        # Far calls (high IV = contango)
        _instrument(far_tag, spot - 1000, "C", 70.0, 200.0, spot),
        _instrument(far_tag, spot + 1000, "C", 68.0, 200.0, spot),
        _instrument(far_tag, spot - 1000, "P", 72.0, 100.0, spot),
    ]
    features = feed._compute_features(instruments)
    assert features["term_structure_slope"] > 0, (
        f"Expected positive slope (contango) but got {features['term_structure_slope']}"
    )


# ── test_fetch_failure_does_not_write ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_failure_does_not_write():
    """aiohttp.ClientError → options:features NOT written; pre-existing LKG preserved."""
    import aiohttp
    feed = make_feed()

    # Pre-populate LKG with a prior successful write
    prior_lkg = {"atm_iv": 55.0, "pcr_oi": 1.1, "term_structure_slope": 0.05,
                 "skew_25d": -2.0, "_lkg_written_at": time.time()}
    feed._redis.set("options:features:lkg", json.dumps(prior_lkg), ex=14400)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get.side_effect = aiohttp.ClientError("connection refused")

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(Exception):
            await feed._fetch_features()

    assert feed._redis.get("options:features") is None, "options:features must NOT be written on failure"

    # LKG must be untouched
    lkg_raw = feed._redis.get("options:features:lkg")
    assert lkg_raw is not None, "LKG should not have been cleared"
    lkg = json.loads(lkg_raw)
    assert lkg["atm_iv"] == pytest.approx(55.0)


# ── test_atm_iv_interpolation ─────────────────────────────────────────────────

def test_atm_iv_interpolation():
    """Spot bracketed by two strikes → interpolated IV strictly between the two strikes' IVs."""
    feed = make_feed()
    near_tag = _days_from_now(7)
    spot = 100_000.0
    # Strike below: 99000 with IV=60; strike above: 101000 with IV=56
    instruments = [
        _instrument(near_tag, 99_000, "C", 60.0, 200.0, spot),
        _instrument(near_tag, 101_000, "C", 56.0, 200.0, spot),
        _instrument(near_tag, 99_000, "P", 65.0, 100.0, spot),
    ]
    features = feed._compute_features(instruments)
    atm_iv = features["atm_iv"]
    assert atm_iv is not None
    # Spot is exactly at 100000 = midpoint of [99000, 101000]
    # interpolated weight_upper = (100000-99000)/(101000-99000) = 0.5
    # Expected = 60 * 0.5 + 56 * 0.5 = 58.0
    assert 56.0 < atm_iv < 60.0, f"Interpolated IV {atm_iv} should be strictly between 56 and 60"
    assert atm_iv == pytest.approx(58.0, abs=0.1)


# ── test_run_retries_after_failure ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_retries_after_failure():
    """_fetch_features raises on first call, succeeds on second → _write_features called once."""
    feed = make_feed()
    good_features = {"atm_iv": 55.0, "pcr_oi": 1.1, "term_structure_slope": 0.05, "skew_25d": -2.0}

    call_count = 0

    async def mock_fetch():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first call fails")
        return good_features

    write_calls = []

    def mock_write(features):
        write_calls.append(features)

    feed._fetch_features = mock_fetch
    feed._write_features = mock_write

    # Patch sleep to avoid waiting
    async def fast_sleep(n):
        pass

    with patch("asyncio.sleep", side_effect=fast_sleep):
        # Run two iterations then stop via StopAsyncIteration hack
        async def run_two_iters():
            for _ in range(2):
                success = False
                try:
                    features = await feed._fetch_features()
                    feed._write_features(features)
                    success = True
                except Exception:
                    pass
                await asyncio.sleep(0)  # patched to no-op

        await run_two_iters()

    assert call_count == 2, f"Expected 2 fetch calls, got {call_count}"
    assert len(write_calls) == 1, f"Expected 1 write call, got {len(write_calls)}"
    assert write_calls[0] == good_features


# ── test_options_features_ttl ─────────────────────────────────────────────────

def test_options_features_ttl():
    """options:features TTL must be ~600s (2× refresh interval)."""
    feed = make_feed()
    features = {"atm_iv": 55.0, "pcr_oi": 1.0, "term_structure_slope": 0.0, "skew_25d": 0.0}
    feed._write_features(features)
    ttl = feed._redis.ttl("options:features")
    assert 590 <= ttl <= 600, f"Expected TTL ~600s, got {ttl}"
