import json
import pytest
from btc_kalshi_system.data.exchange_feed import CoinbaseFeed, KrakenFeed, BitstampFeed


# ── Coinbase ───────────────────────────────────────────────────────────────

def test_coinbase_parse_ticker_message():
    feed = CoinbaseFeed()
    msg = json.dumps({
        "channel": "ticker",
        "events": [{"type": "update", "tickers": [
            {"product_id": "BTC-USD", "price": "103500.00", "last_size": "0.01234"}
        ]}]
    })
    tick = feed.parse_message(msg)
    assert tick is not None
    assert tick.exchange == "coinbase"
    assert tick.price == pytest.approx(103500.0)
    assert tick.volume == pytest.approx(0.01234)


def test_coinbase_returns_none_for_subscription_confirmation():
    feed = CoinbaseFeed()
    assert feed.parse_message(json.dumps({"channel": "subscriptions", "events": []})) is None


def test_coinbase_returns_none_for_non_update_event():
    feed = CoinbaseFeed()
    msg = json.dumps({"channel": "ticker", "events": [{"type": "snapshot", "tickers": []}]})
    assert feed.parse_message(msg) is None


# ── Kraken ─────────────────────────────────────────────────────────────────

def test_kraken_parse_trade_message():
    feed = KrakenFeed()
    msg = json.dumps({
        "channel": "trade",
        "type": "update",
        "data": [{"symbol": "BTC/USD", "price": 103500.0, "qty": 0.5, "side": "buy"}]
    })
    tick = feed.parse_message(msg)
    assert tick is not None
    assert tick.exchange == "kraken"
    assert tick.price == pytest.approx(103500.0)
    assert tick.volume == pytest.approx(0.5)


def test_kraken_returns_none_for_subscribe_response():
    feed = KrakenFeed()
    assert feed.parse_message(json.dumps({"method": "subscribe", "success": True})) is None


def test_kraken_returns_none_for_snapshot():
    feed = KrakenFeed()
    msg = json.dumps({"channel": "trade", "type": "snapshot", "data": []})
    assert feed.parse_message(msg) is None


# ── Bitstamp ───────────────────────────────────────────────────────────────

def test_bitstamp_parse_trade_message():
    feed = BitstampFeed()
    msg = json.dumps({
        "event": "trade",
        "channel": "live_trades_btcusd",
        "data": {"price": 103500.0, "amount": 0.5}
    })
    tick = feed.parse_message(msg)
    assert tick is not None
    assert tick.exchange == "bitstamp"
    assert tick.price == pytest.approx(103500.0)
    assert tick.volume == pytest.approx(0.5)


def test_bitstamp_returns_none_for_subscription_succeeded():
    feed = BitstampFeed()
    msg = json.dumps({
        "event": "bts:subscription_succeeded",
        "data": {},
        "channel": "live_trades_btcusd"
    })
    assert feed.parse_message(msg) is None


def test_bitstamp_returns_none_for_heartbeat():
    feed = BitstampFeed()
    assert feed.parse_message(json.dumps({"event": "bts:heartbeat", "data": {}})) is None


# ── Malformed JSON ──────────────────────────────────────────────────────────

def test_coinbase_returns_none_for_malformed_json():
    assert CoinbaseFeed().parse_message("not json") is None


def test_kraken_returns_none_for_malformed_json():
    assert KrakenFeed().parse_message("not json") is None


def test_bitstamp_returns_none_for_malformed_json():
    assert BitstampFeed().parse_message("not json") is None


# ── Gemini ─────────────────────────────────────────────────────────────────

def test_gemini_parse_trade_event():
    from btc_kalshi_system.data.exchange_feed import GeminiFeed
    feed = GeminiFeed()
    msg = json.dumps({
        "type": "update",
        "eventId": 12345,
        "events": [{"type": "trade", "tid": 99, "price": "103500.00", "amount": "0.025", "makerSide": "ask"}]
    })
    tick = feed.parse_message(msg)
    assert tick is not None
    assert tick.exchange == "gemini"
    assert tick.price == pytest.approx(103500.0)
    assert tick.volume == pytest.approx(0.025)


def test_gemini_returns_none_for_non_update_message():
    from btc_kalshi_system.data.exchange_feed import GeminiFeed
    feed = GeminiFeed()
    assert feed.parse_message(json.dumps({"type": "heartbeat", "heartbeat_sequence": 0})) is None


def test_gemini_returns_none_for_non_trade_event():
    from btc_kalshi_system.data.exchange_feed import GeminiFeed
    feed = GeminiFeed()
    msg = json.dumps({
        "type": "update",
        "events": [{"type": "change", "side": "bid", "price": "103500.00", "remaining": "1.0", "delta": "0.5"}]
    })
    assert feed.parse_message(msg) is None


def test_gemini_returns_none_for_malformed_json():
    from btc_kalshi_system.data.exchange_feed import GeminiFeed
    assert GeminiFeed().parse_message("not json") is None
