import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 900  # 15 minutes — yfinance 1h bars don't change faster
_TICKERS = ["BTC-USD", "^GSPC", "QQQ"]
_ROLLING_WINDOW = 8  # 8 × 1h = 8h rolling correlation


class MacroFeed:
    """Computes 8h rolling BTC/SPX and BTC/QQQ correlations via yfinance.

    Results are cached for 15 minutes. Returns 0.0 on any fetch failure so
    trading is never blocked by a macro feed outage.
    """

    def __init__(self) -> None:
        self._last_fetch_ts: float = 0.0
        self._last_values: dict = {"btc_spx_corr_8h": 0.0, "btc_qqq_corr_8h": 0.0}

    def get_correlations(self) -> dict:
        """Return {"btc_spx_corr_8h": float, "btc_qqq_corr_8h": float}.

        Uses 8h rolling correlation on 1h bars for the past 5 days.
        Returns 0.0 for each metric on any yfinance failure.
        """
        if time.time() - self._last_fetch_ts < _CACHE_TTL_SECONDS:
            return self._last_values
        try:
            data = yf.download(_TICKERS, period="5d", interval="1h", progress=False, auto_adjust=True)
            close = data["Close"]
            btc = close["BTC-USD"].pct_change()
            spx_raw = float(btc.rolling(_ROLLING_WINDOW).corr(close["^GSPC"].pct_change()).iloc[-1])
            qqq_raw = float(btc.rolling(_ROLLING_WINDOW).corr(close["QQQ"].pct_change()).iloc[-1])
            result = {
                "btc_spx_corr_8h": spx_raw if pd.notna(spx_raw) else 0.0,
                "btc_qqq_corr_8h": qqq_raw if pd.notna(qqq_raw) else 0.0,
            }
            self._last_values = result
            self._last_fetch_ts = time.time()
            return result
        except Exception as exc:
            logger.debug(f"MacroFeed: fetch failed — {exc}")
            return self._last_values
