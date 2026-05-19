import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

BRTI_TICK_BUFFER_SIZE: int = 7200          # 2 hours at ~1 tick/second
BRTI_STALE_THRESHOLD_SECONDS: float = 5.0  # exclude exchange ticks older than this
BRTI_RESOLUTION_WINDOW_SECONDS: int = 60   # rolling window for resolution estimate
RECONNECT_DELAYS: list[int] = [1, 2, 4, 8, 16, 32, 60]

COINBASE_WS_URL: str = "wss://advanced-trade-ws.coinbase.com"
KRAKEN_WS_URL: str = "wss://ws.kraken.com/v2"
BITSTAMP_WS_URL: str = "wss://ws.bitstamp.net"
GEMINI_WS_URL: str = "wss://api.gemini.com/v1/marketdata/BTCUSD"

REDIS_TTL_RESOLUTION_ESTIMATE: int = 10
REDIS_TTL_OHLCV: dict[str, int] = {"5min": 600, "15min": 1800, "1h": 7200}
OHLCV_TIMEFRAMES: list[str] = ["5min", "15min", "1h"]

CF_BENCHMARKS_API_KEY: str = os.getenv("CF_BENCHMARKS_API_KEY", "")
