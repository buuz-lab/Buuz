import asyncio
import json
import time
from abc import ABC, abstractmethod

import websockets
from loguru import logger

from btc_kalshi_system.data.models import Tick
from config import RECONNECT_DELAYS


class ExchangeFeed(ABC):

    def __init__(self) -> None:
        self._connected = False

    @property
    @abstractmethod
    def ws_url(self) -> str: ...

    @abstractmethod
    def subscribe_message(self) -> dict: ...

    @abstractmethod
    def parse_message(self, raw: str) -> Tick | None:
        """Parse a raw WebSocket message string. Returns None if not a price tick."""

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def run(self, queue: asyncio.Queue) -> None:
        """Connect and stream forever, reconnecting with exponential backoff."""
        attempt = 0
        while True:
            try:
                await self._connect_and_stream(queue)
                attempt = 0
            except Exception as exc:
                self._connected = False
                delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
                logger.warning(f"{self.__class__.__name__} disconnected ({exc}), retry in {delay}s")
                await asyncio.sleep(delay)
                attempt += 1

    async def _connect_and_stream(self, queue: asyncio.Queue) -> None:
        async with websockets.connect(self.ws_url) as ws:
            self._connected = True
            logger.info(f"{self.__class__.__name__} connected")
            await ws.send(json.dumps(self.subscribe_message()))
            async for raw in ws:
                tick = self.parse_message(raw)
                if tick is not None:
                    await queue.put(tick)


class CoinbaseFeed(ExchangeFeed):

    @property
    def ws_url(self) -> str:
        return "wss://advanced-trade-api.coinbase.com/ws/public"

    def subscribe_message(self) -> dict:
        return {"type": "subscribe", "channel": "ticker", "product_ids": ["BTC-USD"]}

    def parse_message(self, raw: str) -> Tick | None:
        try:
            msg = json.loads(raw)
            if msg.get("channel") != "ticker":
                return None
            for event in msg.get("events", []):
                if event.get("type") != "update":
                    continue
                for ticker in event.get("tickers", []):
                    if ticker.get("product_id") == "BTC-USD":
                        return Tick(
                            exchange="coinbase",
                            price=float(ticker["price"]),
                            volume=float(ticker["volume_24_h"]),
                            timestamp=time.time(),
                        )
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
        return None


class KrakenFeed(ExchangeFeed):

    @property
    def ws_url(self) -> str:
        return "wss://ws.kraken.com/v2"

    def subscribe_message(self) -> dict:
        return {
            "method": "subscribe",
            "params": {"channel": "ticker", "symbol": ["BTC/USD"]},
            "req_id": 1,
        }

    def parse_message(self, raw: str) -> Tick | None:
        try:
            msg = json.loads(raw)
            if msg.get("channel") != "ticker" or msg.get("type") != "update":
                return None
            for item in msg.get("data", []):
                if item.get("symbol") == "BTC/USD":
                    return Tick(
                        exchange="kraken",
                        price=float(item["last"]),
                        volume=float(item["volume"]),
                        timestamp=time.time(),
                    )
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
        return None


class BitstampFeed(ExchangeFeed):

    @property
    def ws_url(self) -> str:
        return "wss://ws.bitstamp.net"

    def subscribe_message(self) -> dict:
        return {"event": "bts:subscribe", "data": {"channel": "live_trades_btcusd"}}

    def parse_message(self, raw: str) -> Tick | None:
        try:
            msg = json.loads(raw)
            if msg.get("event") != "trade":
                return None
            if msg.get("channel") != "live_trades_btcusd":
                return None
            data = msg["data"]
            return Tick(
                exchange="bitstamp",
                price=float(data["price"]),
                volume=float(data["amount"]),  # per-trade size, used as weight proxy
                timestamp=time.time(),
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
        return None
