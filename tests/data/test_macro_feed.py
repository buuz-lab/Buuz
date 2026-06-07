from unittest.mock import patch, MagicMock
import pandas as pd
import time
import pytest

from btc_kalshi_system.data.macro_feed import MacroFeed


def _make_mock_download(btc_vals, spx_vals, qqq_vals):
    """Return a mock yfinance download result (MultiIndex DataFrame)."""
    idx = pd.date_range("2026-06-01", periods=len(btc_vals), freq="h")
    arrays = [["Close"] * 3, ["BTC-USD", "^GSPC", "QQQ"]]
    cols = pd.MultiIndex.from_arrays(arrays)
    df = pd.DataFrame(
        list(zip(btc_vals, spx_vals, qqq_vals)),
        index=idx,
        columns=cols,
    )
    return df


def test_get_correlations_returns_both_keys():
    feed = MacroFeed()
    with patch("btc_kalshi_system.data.macro_feed.yf") as mock_yf:
        mock_yf.download.return_value = _make_mock_download(
            [100 + i for i in range(20)],
            [4000 + i * 2 for i in range(20)],
            [300 + i for i in range(20)],
        )
        result = feed.get_correlations()
    assert "btc_spx_corr_8h" in result
    assert "btc_qqq_corr_8h" in result


def test_get_correlations_returns_float_values():
    feed = MacroFeed()
    with patch("btc_kalshi_system.data.macro_feed.yf") as mock_yf:
        mock_yf.download.return_value = _make_mock_download(
            [100 + i * 0.5 for i in range(20)],
            [4000 + i for i in range(20)],
            [300 + i * 0.5 for i in range(20)],
        )
        result = feed.get_correlations()
    assert isinstance(result["btc_spx_corr_8h"], float)
    assert isinstance(result["btc_qqq_corr_8h"], float)
    assert -1.0 <= result["btc_spx_corr_8h"] <= 1.0
    assert -1.0 <= result["btc_qqq_corr_8h"] <= 1.0


def test_get_correlations_returns_zeros_on_yfinance_failure():
    feed = MacroFeed()
    with patch("btc_kalshi_system.data.macro_feed.yf") as mock_yf:
        mock_yf.download.side_effect = Exception("network error")
        result = feed.get_correlations()
    assert result == {"btc_spx_corr_8h": 0.0, "btc_qqq_corr_8h": 0.0}


def test_get_correlations_uses_cache_within_15_min():
    feed = MacroFeed()
    call_count = 0

    def mock_download(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_mock_download(
            [100 + i for i in range(20)],
            [4000 + i for i in range(20)],
            [300 + i for i in range(20)],
        )

    with patch("btc_kalshi_system.data.macro_feed.yf") as mock_yf:
        mock_yf.download.side_effect = mock_download
        feed.get_correlations()
        feed.get_correlations()
        feed.get_correlations()

    assert call_count == 1  # Only one real fetch; rest served from cache


def test_get_correlations_refetches_after_cache_expires():
    feed = MacroFeed()
    feed._last_fetch_ts = time.time() - 901  # force cache miss

    with patch("btc_kalshi_system.data.macro_feed.yf") as mock_yf:
        mock_yf.download.return_value = _make_mock_download(
            [100 + i for i in range(20)],
            [4000 + i for i in range(20)],
            [300 + i for i in range(20)],
        )
        feed.get_correlations()
        assert mock_yf.download.call_count == 1


def test_get_correlations_returns_last_cached_on_failure_after_success():
    feed = MacroFeed()
    with patch("btc_kalshi_system.data.macro_feed.yf") as mock_yf:
        mock_yf.download.return_value = _make_mock_download(
            [100 + i * 0.5 for i in range(20)],
            [4000 + i for i in range(20)],
            [300 + i * 0.5 for i in range(20)],
        )
        first = feed.get_correlations()

    # Force cache miss, then fail
    feed._last_fetch_ts = 0.0
    with patch("btc_kalshi_system.data.macro_feed.yf") as mock_yf:
        mock_yf.download.side_effect = Exception("network gone")
        second = feed.get_correlations()

    assert second == first  # Last good values returned
