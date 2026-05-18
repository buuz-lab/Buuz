import json
import pytest
from btc_kalshi_system.data.exchange_feed import CoinbaseFeed, KrakenFeed, BitstampFeed


# ── Coinbase ───────────────────────────────────────────────────────────────

def test_coinbase_parse_ticker_message():
    feed = CoinbaseFeed()
    msg = json.dumps({
        "channel": "ticker",
        "events": [{"type": "update", "tickers": [
            {"product_id": "BTC-USD", "price": "103500.00", "volume_24_h": "15234.5"}
        ]}]
    })
    tick = feed.parse_message(msg)
    assert tick is not None
    assert tick.exchange == "coinbase"
    assert tick.price == pytest.approx(103500.0)
    assert tick.volume == pytest.approx(15234.5)


def test_coinbase_returns_none_for_subscription_confirmation():
    feed = CoinbaseFeed()
    assert feed.parse_message(json.dumps({"channel": "subscriptions", "events": []})) is None


def test_coinbase_returns_none_for_non_update_event():
    feed = CoinbaseFeed()
    msg = json.dumps({"channel": "ticker", "events": [{"type": "snapshot", "tickers": []}]})
    assert feed.parse_message(msg) is None


# ── Kraken ─────────────────────────────────────────────────────────────────

def test_kraken_parse_ticker_message():
    feed = KrakenFeed()
    msg = json.dumps({
        "channel": "ticker",
        "type": "update",
        "data": [{"symbol": "BTC/USD", "last": 103500.0, "volume": 3252.6}]
    })
    tick = feed.parse_message(msg)
    assert tick is not None
    assert tick.exchange == "kraken"
    assert tick.price == pytest.approx(103500.0)
    assert tick.volume == pytest.approx(3252.6)


def test_kraken_returns_none_for_subscribe_response():
    feed = KrakenFeed()
    assert feed.parse_message(json.dumps({"method": "subscribe", "success": True})) is None


def test_kraken_returns_none_for_snapshot():
    feed = KrakenFeed()
    msg = json.dumps({"channel": "ticker", "type": "snapshot", "data": []})
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
