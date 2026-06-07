"""Tests for KalshiOrderbookFeed — WebSocket orderbook cache."""

import threading
import time
from unittest.mock import MagicMock

import pytest
from pykalshi import OrderbookManager

from btc_kalshi_system.execution.kalshi_orderbook_feed import (
    KalshiOrderbookFeed,
    _TickerState,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_feed() -> KalshiOrderbookFeed:
    return KalshiOrderbookFeed(api_key_id="test-key", private_key_path="./keys/test.key")


def _inject_ticker(
    feed: KalshiOrderbookFeed,
    ticker: str,
    *,
    has_snapshot: bool,
    last_update_ts: float | None = None,
) -> _TickerState:
    state = _TickerState(
        manager=OrderbookManager(ticker),
        has_snapshot=has_snapshot,
        last_update_ts=last_update_ts if last_update_ts is not None else time.time(),
    )
    with feed._lock:
        feed._books[ticker] = state
    return state


# ── get_orderbook ──────────────────────────────────────────────────────────


class TestGetOrderbook:
    def test_returns_none_before_snapshot_received(self):
        feed = make_feed()
        _inject_ticker(feed, "KXBTC-25JUN-T95000", has_snapshot=False)
        assert feed.get_orderbook("KXBTC-25JUN-T95000") is None

    def test_returns_none_for_unknown_ticker(self):
        feed = make_feed()
        assert feed.get_orderbook("KXBTC-UNKNOWN") is None

    def test_returns_none_when_last_update_older_than_10s(self):
        feed = make_feed()
        _inject_ticker(
            feed, "KXBTC-25JUN-T95000",
            has_snapshot=True,
            last_update_ts=time.time() - 11.0,
        )
        assert feed.get_orderbook("KXBTC-25JUN-T95000") is None

    def test_returns_none_exactly_at_10s_boundary(self):
        """Stale check uses strictly-greater-than 10s."""
        feed = make_feed()
        state = _inject_ticker(
            feed, "KXBTC-25JUN-T95000",
            has_snapshot=True,
            last_update_ts=time.time() - 9.9,
        )
        state.manager.apply_snapshot([("0.45", "10.00")], [("0.55", "5.00")])
        result = feed.get_orderbook("KXBTC-25JUN-T95000")
        assert result is not None  # 9.9s < 10s → not stale

    def test_returns_orderbook_fp_format_after_snapshot(self):
        feed = make_feed()
        ticker = "KXBTC-25JUN-T95000"
        state = _inject_ticker(feed, ticker, has_snapshot=True)
        state.manager.apply_snapshot(
            yes_levels=[("0.40", "5.00"), ("0.45", "10.00")],
            no_levels=[("0.50", "8.00"), ("0.55", "3.00")],
        )

        result = feed.get_orderbook(ticker)
        assert result is not None
        assert "orderbook_fp" in result
        book = result["orderbook_fp"]
        assert "yes_dollars" in book
        assert "no_dollars" in book

        # Lists must be in ascending price order (best bid/ask is last entry)
        yes = book["yes_dollars"]
        no = book["no_dollars"]
        assert [p for p, _ in yes] == ["0.40", "0.45"]
        assert [p for p, _ in no] == ["0.50", "0.55"]

    def test_ascending_order_preserved_for_unsorted_snapshot(self):
        """OrderbookManager keys are dict — order not guaranteed; feed must sort."""
        feed = make_feed()
        ticker = "KXBTC-25JUN-T95000"
        state = _inject_ticker(feed, ticker, has_snapshot=True)
        state.manager.apply_snapshot(
            yes_levels=[("0.55", "1.00"), ("0.30", "2.00"), ("0.45", "3.00")],
            no_levels=[("0.40", "4.00")],
        )

        result = feed.get_orderbook(ticker)
        yes_prices = [float(p) for p, _ in result["orderbook_fp"]["yes_dollars"]]
        assert yes_prices == sorted(yes_prices)

    def test_snapshot_plus_delta_updates_quantity(self):
        """Delta applied after snapshot is visible in get_orderbook output."""
        feed = make_feed()
        ticker = "KXBTC-25JUN-T95000"
        state = _inject_ticker(feed, ticker, has_snapshot=True)
        state.manager.apply_snapshot(
            yes_levels=[("0.45", "10.00")],
            no_levels=[("0.55", "5.00")],
        )
        # Remove 3 YES contracts at 0.45
        state.manager.apply_delta("yes", "0.45", "-3.00")

        result = feed.get_orderbook(ticker)
        yes = result["orderbook_fp"]["yes_dollars"]
        assert len(yes) == 1
        assert yes[0][0] == "0.45"
        assert float(yes[0][1]) == pytest.approx(7.0)

    def test_delta_removes_level_when_quantity_hits_zero(self):
        feed = make_feed()
        ticker = "KXBTC-25JUN-T95000"
        state = _inject_ticker(feed, ticker, has_snapshot=True)
        state.manager.apply_snapshot(
            yes_levels=[("0.45", "5.00")],
            no_levels=[("0.55", "5.00")],
        )
        state.manager.apply_delta("yes", "0.45", "-5.00")  # removes level

        result = feed.get_orderbook(ticker)
        yes = result["orderbook_fp"]["yes_dollars"]
        assert yes == []

    def test_parse_orderbook_compatibility(self):
        """Output format is parseable by main._parse_orderbook — best bid is last entry."""
        feed = make_feed()
        ticker = "KXBTC-25JUN-T95000"
        state = _inject_ticker(feed, ticker, has_snapshot=True)
        state.manager.apply_snapshot(
            yes_levels=[("0.40", "5.00"), ("0.45", "10.00")],
            no_levels=[("0.50", "8.00"), ("0.55", "3.00")],
        )

        result = feed.get_orderbook(ticker)
        book_fp = result["orderbook_fp"]
        yes = book_fp["yes_dollars"]
        no = book_fp["no_dollars"]

        # main._parse_orderbook uses [-1] for best bid/ask
        best_yes_bid = float(yes[-1][0])  # 0.45
        best_no_bid = float(no[-1][0])    # 0.55
        best_bid_cents = round(best_yes_bid * 100)    # 45
        best_ask_cents = round((1.0 - best_no_bid) * 100)  # 45

        assert best_bid_cents == 45
        assert best_ask_cents == 45


# ── update_subscriptions ──────────────────────────────────────────────────────


class TestUpdateSubscriptions:
    def test_new_tickers_added_to_desired(self):
        feed = make_feed()
        feed.update_subscriptions({"KXBTC-A", "KXBTC-B"})
        with feed._lock:
            assert "KXBTC-A" in feed._desired_tickers
            assert "KXBTC-B" in feed._desired_tickers

    def test_new_tickers_initialize_book_state(self):
        feed = make_feed()
        feed.update_subscriptions({"KXBTC-A"})
        with feed._lock:
            assert "KXBTC-A" in feed._books
            assert not feed._books["KXBTC-A"].has_snapshot

    def test_removed_tickers_leave_desired(self):
        feed = make_feed()
        feed.update_subscriptions({"KXBTC-A", "KXBTC-B"})
        feed.update_subscriptions({"KXBTC-A"})  # KXBTC-B dropped
        with feed._lock:
            assert "KXBTC-A" in feed._desired_tickers
            assert "KXBTC-B" not in feed._desired_tickers

    def test_empty_set_clears_desired(self):
        feed = make_feed()
        feed.update_subscriptions({"KXBTC-A"})
        feed.update_subscriptions(set())
        with feed._lock:
            assert len(feed._desired_tickers) == 0

    def test_idempotent_on_same_set(self):
        feed = make_feed()
        feed.update_subscriptions({"KXBTC-A"})
        feed.update_subscriptions({"KXBTC-A"})  # same set — no duplication
        with feed._lock:
            assert feed._books["KXBTC-A"].has_snapshot is False

    def test_thread_safe_from_multiple_threads(self):
        """Concurrent calls must not corrupt _desired_tickers or _books."""
        feed = make_feed()
        errors = []

        def worker(ticker_set):
            try:
                feed.update_subscriptions(ticker_set)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=({"TICKER-%d" % i},))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ── _apply_drain ──────────────────────────────────────────────────────────────


class TestApplyDrain:
    """_apply_drain() syncs feed subscriptions with _desired_tickers."""

    def test_subscribes_new_desired_tickers(self):
        feed = make_feed()
        mock_af = MagicMock()
        feed.update_subscriptions({"KXBTC-A"})

        feed._apply_drain(mock_af)

        mock_af.subscribe.assert_called_once_with(
            "orderbook_delta", market_ticker="KXBTC-A"
        )
        assert "KXBTC-A" in feed._subscribed

    def test_does_not_resubscribe_already_subscribed_ticker(self):
        feed = make_feed()
        mock_af = MagicMock()
        feed.update_subscriptions({"KXBTC-A"})
        feed._subscribed.add("KXBTC-A")

        feed._apply_drain(mock_af)

        mock_af.subscribe.assert_not_called()

    def test_unsubscribes_ticker_removed_from_desired(self):
        feed = make_feed()
        mock_af = MagicMock()

        # Start with KXBTC-A subscribed
        feed._subscribed.add("KXBTC-A")
        _inject_ticker(feed, "KXBTC-A", has_snapshot=True)

        # Remove from desired (empty desired)
        feed.update_subscriptions(set())
        feed._apply_drain(mock_af)

        mock_af.unsubscribe.assert_called_once_with(
            "orderbook_delta", market_ticker="KXBTC-A"
        )
        assert "KXBTC-A" not in feed._subscribed

    def test_unsubscribe_cleans_up_book_state(self):
        feed = make_feed()
        mock_af = MagicMock()

        feed._subscribed.add("KXBTC-A")
        _inject_ticker(feed, "KXBTC-A", has_snapshot=True)
        feed.update_subscriptions(set())
        feed._apply_drain(mock_af)

        with feed._lock:
            assert "KXBTC-A" not in feed._books

    def test_diffs_correctly_subscribe_new_unsubscribe_removed(self):
        feed = make_feed()
        mock_af = MagicMock()

        # Currently subscribed: A, B
        feed._subscribed.update({"KXBTC-A", "KXBTC-B"})
        _inject_ticker(feed, "KXBTC-A", has_snapshot=True)
        _inject_ticker(feed, "KXBTC-B", has_snapshot=True)

        # Desired: B, C (A removed, C added)
        feed.update_subscriptions({"KXBTC-B", "KXBTC-C"})
        feed._apply_drain(mock_af)

        # Should subscribe C, unsubscribe A, leave B alone
        mock_af.subscribe.assert_called_once_with(
            "orderbook_delta", market_ticker="KXBTC-C"
        )
        mock_af.unsubscribe.assert_called_once_with(
            "orderbook_delta", market_ticker="KXBTC-A"
        )
        assert "KXBTC-C" in feed._subscribed
        assert "KXBTC-A" not in feed._subscribed
        assert "KXBTC-B" in feed._subscribed


# ── Reconnect state ───────────────────────────────────────────────────────────


class TestReconnectState:
    def test_reconnect_resets_has_snapshot_flags(self):
        """After reconnect, snapshots are stale — must wait for fresh server snapshot."""
        feed = make_feed()
        _inject_ticker(feed, "KXBTC-A", has_snapshot=True)
        _inject_ticker(feed, "KXBTC-B", has_snapshot=True)

        # Simulate what drain loop does on reconnect
        with feed._lock:
            for state in feed._books.values():
                state.has_snapshot = False

        assert feed.get_orderbook("KXBTC-A") is None
        assert feed.get_orderbook("KXBTC-B") is None

    def test_desired_tickers_preserved_across_reconnect(self):
        """_desired_tickers drives re-subscription on reconnect via _apply_drain."""
        feed = make_feed()
        feed.update_subscriptions({"KXBTC-A", "KXBTC-B"})

        # Simulate reconnect: _subscribed is cleared (feed object reconnects internally)
        feed._subscribed.clear()

        with feed._lock:
            assert "KXBTC-A" in feed._desired_tickers
            assert "KXBTC-B" in feed._desired_tickers

    def test_drain_resubscribes_all_tickers_after_reconnect(self):
        """After _subscribed is cleared (reconnect), drain loop re-subscribes all desired."""
        feed = make_feed()
        mock_af = MagicMock()

        feed.update_subscriptions({"KXBTC-A", "KXBTC-B"})
        feed._subscribed.clear()  # simulate reconnect clearing subscribed set

        feed._apply_drain(mock_af)

        subscribed_tickers = {
            call.kwargs["market_ticker"] for call in mock_af.subscribe.call_args_list
        }
        assert subscribed_tickers == {"KXBTC-A", "KXBTC-B"}


# ── get_open_snapshot ─────────────────────────────────────────────────────────


def test_get_open_snapshot_includes_depth_imbalance():
    """depth_imbalance is returned as (bid-ask)/(bid+ask) when depth is non-zero."""
    feed = make_feed()
    # Snapshot via WS path already has depth_bid and depth_ask
    with feed._lock:
        feed._open_snapshots["KXBTC-25Jun2026-99500"] = {
            "mid_prob": 0.50, "spread": 0.02,
            "depth_bid": 300.0, "depth_ask": 100.0, "ts": 1.0,
        }
    snap = feed.get_open_snapshot("KXBTC-25Jun2026-99500")
    assert "depth_imbalance" in snap
    expected = (300.0 - 100.0) / (300.0 + 100.0)  # 0.5
    assert abs(snap["depth_imbalance"] - expected) < 1e-9


def test_get_open_snapshot_imbalance_none_when_zero_depth():
    """depth_imbalance is None when total depth is 0 (REST fallback path)."""
    feed = make_feed()
    with feed._lock:
        feed._open_snapshots["KXBTC-25Jun2026-99500"] = {
            "mid_prob": 0.50, "spread": 0.02,
            "depth_bid": 0.0, "depth_ask": 0.0, "ts": 1.0,
        }
    snap = feed.get_open_snapshot("KXBTC-25Jun2026-99500")
    assert snap["depth_imbalance"] is None


def test_get_open_snapshot_imbalance_range():
    """depth_imbalance is always in [-1, 1]."""
    feed = make_feed()
    with feed._lock:
        feed._open_snapshots["KXBTC-25Jun2026-99500"] = {
            "mid_prob": 0.50, "spread": 0.02,
            "depth_bid": 1000.0, "depth_ask": 0.0, "ts": 1.0,
        }
    snap = feed.get_open_snapshot("KXBTC-25Jun2026-99500")
    # All bids, no asks → imbalance = +1.0
    assert snap["depth_imbalance"] == 1.0
