import time
from unittest.mock import MagicMock, patch

import pytest

from btc_kalshi_system.execution.router import ClientState, KalshiClientRouter


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_router(primary_available: bool = True, raw_mock: MagicMock | None = None):
    """Build a KalshiClientRouter with mocked internals."""
    raw = raw_mock or MagicMock()
    primary = MagicMock() if primary_available else None

    with patch("btc_kalshi_system.execution.router.KalshiRawClient", return_value=raw), \
         patch.object(KalshiClientRouter, "_init_pykalshi", return_value=primary):
        router = KalshiClientRouter(
            api_key_id="test-key",
            private_key_path="./keys/test.key",
        )

    router._raw = raw
    router._primary = primary
    router._state = ClientState.PRIMARY if primary_available else ClientState.FALLBACK
    return router, raw, primary


# ------------------------------------------------------------------
# Initial state
# ------------------------------------------------------------------

def test_starts_in_fallback_when_pykalshi_unavailable():
    router, _, _ = make_router(primary_available=False)
    assert router.state is ClientState.FALLBACK


def test_starts_in_primary_when_pykalshi_available():
    router, _, _ = make_router(primary_available=True)
    assert router.state is ClientState.PRIMARY


# ------------------------------------------------------------------
# get_orderbook always uses raw
# ------------------------------------------------------------------

def test_get_orderbook_always_uses_raw_client():
    router, raw, primary = make_router(primary_available=True)
    raw.get_orderbook.return_value = {"orderbook": {}}
    result = router.get_orderbook("KXBTC-25JUN-T95000")
    raw.get_orderbook.assert_called_once_with("KXBTC-25JUN-T95000")
    primary.portfolio.get_orderbook.assert_not_called() if hasattr(primary, "portfolio") else None
    assert result == {"orderbook": {}}


# ------------------------------------------------------------------
# Failover after 3 consecutive primary failures
# ------------------------------------------------------------------

def test_three_primary_failures_switches_to_fallback():
    router, raw, primary = make_router(primary_available=True)

    # Make _route_through_primary always raise
    router._route_through_primary = MagicMock(side_effect=Exception("pykalshi down"))
    raw.get_balance.return_value = {"balance": 100}

    # Failures 1 and 2 — still PRIMARY, falls through to raw
    router.get_balance()
    assert router.state is ClientState.PRIMARY
    router.get_balance()
    assert router.state is ClientState.PRIMARY

    # Failure 3 — switches to FALLBACK
    router.get_balance()
    assert router.state is ClientState.FALLBACK


def test_consecutive_failures_reset_on_success():
    router, raw, primary = make_router(primary_available=True)
    router._consecutive_failures = 2

    router._route_through_primary = MagicMock(return_value={"balance": 50})
    router.get_balance()

    assert router._consecutive_failures == 0


# ------------------------------------------------------------------
# BOTH_FAILED raises immediately
# ------------------------------------------------------------------

def test_both_failed_raises_runtime_error():
    router, raw, primary = make_router(primary_available=False)
    router._state = ClientState.BOTH_FAILED
    router._last_recovery_attempt = time.time()  # suppress recovery so state stays BOTH_FAILED

    with pytest.raises(RuntimeError, match="Both Kalshi clients failed"):
        router.get_balance()


def test_both_failed_does_not_call_raw():
    router, raw, _ = make_router(primary_available=False)
    router._state = ClientState.BOTH_FAILED
    router._last_recovery_attempt = time.time()  # suppress recovery so state stays BOTH_FAILED

    with pytest.raises(RuntimeError):
        router.get_balance()
    raw.get_balance.assert_not_called()


# ------------------------------------------------------------------
# Recovery
# ------------------------------------------------------------------

def test_recovery_resets_to_primary():
    router, raw, _ = make_router(primary_available=False)
    router._state = ClientState.FALLBACK
    router._last_recovery_attempt = 0.0  # force recovery eligible

    new_primary = MagicMock()
    with patch.object(router, "_init_pykalshi", return_value=new_primary):
        router._maybe_attempt_recovery()

    assert router.state is ClientState.PRIMARY
    assert router._consecutive_failures == 0
    assert router._primary is new_primary


def test_recovery_not_attempted_before_interval():
    router, _, _ = make_router(primary_available=False)
    router._state = ClientState.FALLBACK
    router._last_recovery_attempt = time.time()  # just attempted

    with patch.object(router, "_init_pykalshi") as mock_init:
        router._maybe_attempt_recovery()
        mock_init.assert_not_called()

    assert router.state is ClientState.FALLBACK


def test_failed_recovery_stays_in_fallback():
    router, _, _ = make_router(primary_available=False)
    router._state = ClientState.FALLBACK
    router._last_recovery_attempt = 0.0

    with patch.object(router, "_init_pykalshi", return_value=None):
        router._maybe_attempt_recovery()

    assert router.state is ClientState.FALLBACK


# ------------------------------------------------------------------
# place_order falls through to raw when in FALLBACK
# ------------------------------------------------------------------

def test_place_order_uses_raw_in_fallback():
    router, raw, _ = make_router(primary_available=False)
    router._state = ClientState.FALLBACK
    router._last_recovery_attempt = time.time()  # suppress recovery

    raw.place_order.return_value = {"order_id": "abc123"}

    result = router.place_order(
        ticker="KXBTC-25JUN-T95000",
        side="yes",
        count=5,
        price_cents=55,
    )

    raw.place_order.assert_called_once_with(
        ticker="KXBTC-25JUN-T95000",
        side="yes",
        count=5,
        price_cents=55,
        client_order_id=None,
    )
    assert result == {"order_id": "abc123"}


def test_place_order_raw_failure_sets_both_failed():
    router, raw, _ = make_router(primary_available=False)
    router._state = ClientState.FALLBACK
    router._last_recovery_attempt = time.time()

    raw.place_order.side_effect = Exception("network error")

    with pytest.raises(RuntimeError, match="Both Kalshi clients failed"):
        router.place_order("KXBTC", "yes", 1, 50)

    assert router.state is ClientState.BOTH_FAILED
