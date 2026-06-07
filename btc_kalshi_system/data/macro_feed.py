import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 900  # 15 minutes — daily bars don't change intraday
_TICKERS = ["BTC-USD", "^GSPC", "QQQ"]
_ROLLING_WINDOW = 8  # 8 × 1d = 8 trading days rolling correlation


class MacroFeed:
    """Computes 8-trading-day rolling BTC/SPX and BTC/QQQ correlations via yfinance.

    Results are cached for 15 minutes. Returns 0.0 on any fetch failure so
    trading is never blocked by a macro feed outage.
    """

    def __init__(self) -> None:
        self._last_fetch_ts: float = 0.0
        self._last_values: dict = {"btc_spx_corr_8d": 0.0, "btc_qqq_corr_8d": 0.0}

    def get_correlations(self) -> dict[str, float]:
        """Return {"btc_spx_corr_8d": float, "btc_qqq_corr_8d": float}.

        Uses 8-trading-day rolling correlation on daily bars for the past 60 days.
        Returns 0.0 for each metric on any yfinance failure.
        """
        if time.time() - self._last_fetch_ts < _CACHE_TTL_SECONDS:
            return dict(self._last_values)
        try:
            data = yf.download(_TICKERS, period="60d", interval="1d", progress=False, auto_adjust=True)
            close = data["Close"].dropna()   # align to shared trading days (removes weekends + holidays)
            btc = close["BTC-USD"].pct_change()
            spx_raw = float(btc.rolling(_ROLLING_WINDOW).corr(close["^GSPC"].pct_change()).iloc[-1])
            qqq_raw = float(btc.rolling(_ROLLING_WINDOW).corr(close["QQQ"].pct_change()).iloc[-1])
            result = {
                "btc_spx_corr_8d": spx_raw if pd.notna(spx_raw) else 0.0,
                "btc_qqq_corr_8d": qqq_raw if pd.notna(qqq_raw) else 0.0,
            }
            self._last_values = result
            self._last_fetch_ts = time.time()
            return result
        except Exception as exc:
            logger.warning(f"MacroFeed: fetch failed — {exc}")
            self._last_fetch_ts = time.time()
            return dict(self._last_values)
