import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

BRTI_TICK_BUFFER_SIZE: int = 150_000  # ~42 hours at 1 tick/second
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
COINGLASS_API_KEY: str = os.getenv("COINGLASS_API_KEY", "")
HYPERLIQUID_BASE_URL: str = "https://api.hyperliquid.xyz"
KRAKEN_FUTURES_BASE_URL: str = "https://futures.kraken.com/derivatives/api/v3"

KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./keys/kalshi_private.key")

# Set to true to simulate trades without placing real orders.
# Use this to bootstrap the calibrator and edge tracker before going live.
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "false").lower() == "true"

# Path to the serialized XGBoost RegimeModel. Created by scripts/train_regime.py
# once ≥500 non-stale resolved trades are accumulated in trades.db. KronosV2 will
# attempt to load this file at startup; if missing, the system runs in bootstrap
# mode (Kronos-only with _BOOTSTRAP_SHRINK).
REGIME_MODEL_PATH: str = os.getenv("REGIME_MODEL_PATH", "models/regime.pkl")

# Gate 2 enforcement mode.
#   False (default): Kronos/regime disagreements are logged but the trade still
#       proceeds. Use this for the first ~50 trades after loading a freshly
#       trained model to observe the disagreement rate and confidence
#       distribution before letting the gate block trades.
#   True: Disagreements return None (current "trained-path" behavior).
# Has no effect while RegimeModel is untrained — Gate 2 is bypassed entirely
# in the NotTrainedError code path regardless of this flag.
REGIME_GATE2_ENFORCING: bool = os.getenv("REGIME_GATE2_ENFORCING", "false").lower() == "true"

# Gate 7 (CVD soft gate): block YES→UP when CVD < -threshold, NO→DOWN when CVD > +threshold.
# Statistical basis: YES→UP trades with negative CVD have a 32.3% win rate, well below
# the ~52% breakeven for typical fill prices. This addresses CVD oscillation loss streaks.
CVD_GATE_THRESHOLD: float = 0.3

# Gate 8 (Kalshi consensus gate): block trades when Kalshi prices >= this margin against
# our direction. 8% = Kalshi must price >=58% against us. Zero false positives on 248
# good-day trades at this threshold; 22% win rate on blocked trades.
KALSHI_CONSENSUS_THRESHOLD: float = 0.08

# Path to the serialized isotonic-regression Calibrator. Created by scripts/train_calibrator.py
# or by in-process refit every 25 resolutions. KronosV2 attempts to load on startup.
CALIBRATOR_MODEL_PATH: str = os.getenv("CALIBRATOR_MODEL_PATH", "models/calibrator.pkl")
