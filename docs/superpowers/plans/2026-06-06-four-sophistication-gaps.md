# Four Sophistication Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four system upgrades — Kalshi order imbalance feature, mid-candle exit paper mode, dynamic entry gate (rule-based, swappable), and macro BTC/SPX/QQQ correlation features — without touching the live trading signal or any gate enforcement logic beyond Gate 12.

**Architecture:** Two parallel groups: Group A (feature pipeline — new columns in `candle_features`, new entries in `_FEATURE_ORDER`) and Group B (execution logic — Gate 12 dynamic cap, mid-candle exit paper logging). Group A tasks 1-5 must be done in order; Group B tasks 6-7 are independent of Group A and of each other.

**Tech Stack:** Python 3.11, XGBoost, asyncio, SQLite, Redis, yfinance (new dependency), existing `KalshiOrderbookFeed`, `SignalFusionEngine`, `DerivativesFeed`.

---

## Group A: Feature Pipeline

### Task 1: MacroFeed — new file + yfinance dependency

**Files:**
- Create: `btc_kalshi_system/data/macro_feed.py`
- Create: `tests/data/test_macro_feed.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add yfinance to requirements.txt**

Open `requirements.txt` and add after `requests>=2.31`:
```
yfinance>=0.2
```

- [ ] **Step 2: Write the failing tests**

Create `tests/data/test_macro_feed.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_macro_feed.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'btc_kalshi_system.data.macro_feed'`

- [ ] **Step 4: Implement MacroFeed**

Create `btc_kalshi_system/data/macro_feed.py`:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_macro_feed.py -v
```
Expected: 6 tests PASS

- [ ] **Step 6: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/data/macro_feed.py tests/data/test_macro_feed.py requirements.txt && git commit -m "feat: MacroFeed — 8h rolling BTC/SPX and BTC/QQQ correlation via yfinance"
```

---

### Task 2: Integrate MacroFeed into DerivativesFeed

**Files:**
- Modify: `btc_kalshi_system/data/derivatives_feed.py`
- Modify: `tests/data/test_derivatives_feed.py`

- [ ] **Step 1: Write the failing test**

Open `tests/data/test_derivatives_feed.py` and add at the end of the file:
```python
def test_fetch_features_includes_macro_correlations(fake_redis, monkeypatch):
    """MacroFeed correlations are merged into the features dict."""
    from btc_kalshi_system.data.macro_feed import MacroFeed

    feed = DerivativesFeed(redis_url="redis://localhost")
    feed._redis = fake_redis

    # Stub out all the async fetch helpers to return zeros
    async def _zero_funding(): return 0.0, 0.0, 0.0, False
    async def _zero_trades(): return 0.0, 0.0, 0.0, True
    async def _zero_volume(): return 1.0
    monkeypatch.setattr(feed, "_fetch_funding_and_oi", _zero_funding)
    monkeypatch.setattr(feed, "_fetch_trades_data", _zero_trades)
    monkeypatch.setattr(feed, "_fetch_volume_ratio", _zero_volume)
    monkeypatch.setattr(feed, "_brti_volatility_1h", lambda: 0.0)

    # Stub MacroFeed
    mock_macro = MagicMock(spec=MacroFeed)
    mock_macro.get_correlations.return_value = {"btc_spx_corr_8h": 0.42, "btc_qqq_corr_8h": 0.38}
    feed._macro_feed = mock_macro

    import asyncio
    features = asyncio.get_event_loop().run_until_complete(feed._fetch_features())

    assert features["btc_spx_corr_8h"] == 0.42
    assert features["btc_qqq_corr_8h"] == 0.38
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py::test_fetch_features_includes_macro_correlations -v
```
Expected: FAIL — `AttributeError: '_macro_feed'` or similar

- [ ] **Step 3: Implement the integration**

In `btc_kalshi_system/data/derivatives_feed.py`:

At the top of the file add the import (after existing imports):
```python
from btc_kalshi_system.data.macro_feed import MacroFeed
```

In `__init__` of `DerivativesFeed`, after `self._redis = redis.from_url(redis_url)`, add:
```python
self._macro_feed = MacroFeed()
```

In `_fetch_features()`, after `features: dict = { ... }` is fully built and before `if okx_partial:`, add:
```python
macro = self._macro_feed.get_correlations()
features.update(macro)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/data/test_derivatives_feed.py -v
```
Expected: all existing tests + new test PASS

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/data/derivatives_feed.py tests/data/test_derivatives_feed.py && git commit -m "feat: integrate MacroFeed into DerivativesFeed — btc_spx/qqq_corr_8h in regime:features"
```

---

### Task 3: Kalshi Order Imbalance in orderbook feed

**Files:**
- Modify: `btc_kalshi_system/execution/kalshi_orderbook_feed.py`
- Modify: `tests/execution/test_kalshi_orderbook_feed.py`

- [ ] **Step 1: Write the failing tests**

Open `tests/execution/test_kalshi_orderbook_feed.py` and add:
```python
def test_get_open_snapshot_includes_depth_imbalance(feed):
    """depth_imbalance is returned as (bid-ask)/(bid+ask) when depth is non-zero."""
    feed._inject_ticker("KXBTC-25Jun2026-99500")
    # Snapshot via WS path already has depth_bid and depth_ask
    feed._open_snapshots["KXBTC-25Jun2026-99500"] = {
        "mid_prob": 0.50, "spread": 0.02,
        "depth_bid": 300.0, "depth_ask": 100.0, "ts": 1.0,
    }
    snap = feed.get_open_snapshot("KXBTC-25Jun2026-99500")
    assert "depth_imbalance" in snap
    expected = (300.0 - 100.0) / (300.0 + 100.0)  # 0.5
    assert abs(snap["depth_imbalance"] - expected) < 1e-9


def test_get_open_snapshot_imbalance_none_when_zero_depth(feed):
    """depth_imbalance is None when total depth is 0 (REST fallback path)."""
    feed._open_snapshots["KXBTC-25Jun2026-99500"] = {
        "mid_prob": 0.50, "spread": 0.02,
        "depth_bid": 0.0, "depth_ask": 0.0, "ts": 1.0,
    }
    snap = feed.get_open_snapshot("KXBTC-25Jun2026-99500")
    assert snap["depth_imbalance"] is None


def test_get_open_snapshot_imbalance_range(feed):
    """depth_imbalance is always in [-1, 1]."""
    feed._open_snapshots["KXBTC-25Jun2026-99500"] = {
        "mid_prob": 0.50, "spread": 0.02,
        "depth_bid": 1000.0, "depth_ask": 0.0, "ts": 1.0,
    }
    snap = feed.get_open_snapshot("KXBTC-25Jun2026-99500")
    # All bids, no asks → imbalance = +1.0
    assert snap["depth_imbalance"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_kalshi_orderbook_feed.py::test_get_open_snapshot_includes_depth_imbalance tests/execution/test_kalshi_orderbook_feed.py::test_get_open_snapshot_imbalance_none_when_zero_depth tests/execution/test_kalshi_orderbook_feed.py::test_get_open_snapshot_imbalance_range -v
```
Expected: 3 tests FAIL — `AssertionError: 'depth_imbalance' not in snap`

- [ ] **Step 3: Implement imbalance computation**

In `btc_kalshi_system/execution/kalshi_orderbook_feed.py`, replace `get_open_snapshot()`:
```python
def get_open_snapshot(self, ticker: str) -> dict | None:
    """Return the first orderbook snapshot captured for this ticker, or None.

    Keys: mid_prob (float, 0-1), spread (float, 0-1), depth_bid, depth_ask,
    depth_imbalance (float -1 to +1, or None when total depth is 0), ts.
    Safe to call from any thread after the contract has opened.
    """
    with self._lock:
        snap = self._open_snapshots.get(ticker)
        if snap is None:
            return None
        total = snap["depth_bid"] + snap["depth_ask"]
        imbalance = (snap["depth_bid"] - snap["depth_ask"]) / total if total > 0 else None
        return {**snap, "depth_imbalance": imbalance}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_kalshi_orderbook_feed.py -v
```
Expected: all tests PASS (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/execution/kalshi_orderbook_feed.py tests/execution/test_kalshi_orderbook_feed.py && git commit -m "feat: add depth_imbalance to get_open_snapshot() — (bid-ask)/(bid+ask), None on zero depth"
```

---

### Task 4: Fusion engine imbalance cache

**Files:**
- Modify: `btc_kalshi_system/signal/fusion.py`
- Modify: `tests/signal/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

Open `tests/signal/test_fusion.py` and add:
```python
def test_set_kalshi_imbalance_updates_regime_features(make_fusion):
    """set_kalshi_imbalance() causes kalshi_open_imbalance to appear in _regime_features."""
    fusion = make_fusion()
    fusion.set_kalshi_imbalance(0.42)
    features, _, _, _ = fusion._regime_features()
    assert features.get("kalshi_open_imbalance") == 0.42


def test_set_kalshi_imbalance_none_passes_through(make_fusion):
    """None imbalance (REST fallback) is passed through as None."""
    fusion = make_fusion()
    fusion.set_kalshi_imbalance(None)
    features, _, _, _ = fusion._regime_features()
    assert features.get("kalshi_open_imbalance") is None


def test_kalshi_imbalance_defaults_to_none_before_set(make_fusion):
    """Before set_kalshi_imbalance() is called, value is None."""
    fusion = make_fusion()
    features, _, _, _ = fusion._regime_features()
    assert features.get("kalshi_open_imbalance") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_fusion.py::test_set_kalshi_imbalance_updates_regime_features tests/signal/test_fusion.py::test_set_kalshi_imbalance_none_passes_through tests/signal/test_fusion.py::test_kalshi_imbalance_defaults_to_none_before_set -v
```
Expected: 3 tests FAIL — `AttributeError: 'SignalFusionEngine' has no attribute 'set_kalshi_imbalance'`

- [ ] **Step 3: Add cache + setter + _regime_features entry**

In `btc_kalshi_system/signal/fusion.py`:

In `__init__`, after `self._last_kronos_raw_5min: float | None = None`, add:
```python
self._last_kalshi_open_imbalance: float | None = None
```

After `update_kalshi_spread()` method, add:
```python
def set_kalshi_imbalance(self, imbalance: float | None) -> None:
    self._last_kalshi_open_imbalance = imbalance
```

In `_regime_features()`, in the returned features dict, after the `"kronos_raw_5min"` entry, add:
```python
"kalshi_open_imbalance": self._last_kalshi_open_imbalance,
```

Also add the macro correlation features — they come from Redis via `ctx` (same path as funding_rate etc.):
```python
"btc_spx_corr_8h": float(ctx.get("btc_spx_corr_8h") or 0.0),
"btc_qqq_corr_8h": float(ctx.get("btc_qqq_corr_8h") or 0.0),
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_fusion.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/signal/fusion.py tests/signal/test_fusion.py && git commit -m "feat: fusion engine caches kalshi_open_imbalance + exposes macro corr from Redis ctx"
```

---

### Task 5: Feature order expansion + main.py Group A wiring

**Files:**
- Modify: `btc_kalshi_system/models/regime_model.py`
- Modify: `main.py`
- Modify: `tests/signal/test_feature_order.py`

- [ ] **Step 1: Update the feature count test first**

In `tests/signal/test_feature_order.py`, find and update:
```python
# Before:
assert len(_FEATURE_ORDER) == 29  # 27 market features + kronos_raw_15min + kronos_raw_5min

# After:
assert len(_FEATURE_ORDER) == 32  # 29 prev + kalshi_open_imbalance + btc_spx_corr_8h + btc_qqq_corr_8h
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_feature_order.py -v
```
Expected: `test_feature_order_all_three_match` FAIL — `assert 29 == 32`

- [ ] **Step 3: Add features to _FEATURE_ORDER**

In `btc_kalshi_system/models/regime_model.py`, after `"skew_25d"` and before `"btc_24h_return"`, add:
```python
    # Kalshi open orderbook imbalance (session 35) — order flow conviction at candle open.
    # Analogous to CVD: measures who is aggressively buying vs selling at contract open.
    # NULL → NaN on rows captured before this was deployed; XGBoost treats NaN as missing.
    "kalshi_open_imbalance",
```

After `"kronos_raw_5min"` (the last existing feature), add:
```python
    # Macro correlation features (session 35) — 8h rolling BTC/SPX and BTC/QQQ correlation.
    # Signals when BTC is trading as a macro risk asset vs independently.
    # 0.0 when MacroFeed fetch fails or outside data availability window.
    "btc_spx_corr_8h",
    "btc_qqq_corr_8h",
```

- [ ] **Step 4: Wire imbalance into main.py — process_market**

In `main.py`, in `_process_market()`, after `self._fusion.update_kalshi_mid(mid_cents)` (the existing step e), add:
```python
        # Inject Kalshi open imbalance so regime model can use it as a feature.
        _imbal_snap = self._orderbook_feed.get_open_snapshot(ticker)
        self._fusion.set_kalshi_imbalance(
            _imbal_snap.get("depth_imbalance") if _imbal_snap else None
        )
```

- [ ] **Step 5: Wire imbalance into main.py — candle logger**

In `main.py`, in `_candle_logger_loop()`, after:
```python
kalshi_open_depth  = (
    (_open_snap["depth_bid"] + _open_snap["depth_ask"]) if _open_snap else None
)
```
Add:
```python
kalshi_open_imbalance = _open_snap.get("depth_imbalance") if _open_snap else None
```

Then in the `col_names` string, add `"kalshi_open_imbalance, "` after `"kalshi_open_depth, "`.

In the VALUES list, add `kalshi_open_imbalance,` after `kalshi_open_depth,`.

- [ ] **Step 6: Add migrations to _CANDLE_FEATURES_COLUMN_MIGRATIONS**

In `main.py`, find `_CANDLE_FEATURES_COLUMN_MIGRATIONS` and add:
```python
    ("kalshi_open_imbalance", "REAL DEFAULT NULL"),
    ("btc_spx_corr_8h",      "REAL DEFAULT NULL"),
    ("btc_qqq_corr_8h",      "REAL DEFAULT NULL"),
```

- [ ] **Step 7: Run all tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_feature_order.py tests/signal/test_fusion.py tests/execution/test_kalshi_orderbook_feed.py -v
```
Expected: all tests PASS, feature count = 32

- [ ] **Step 8: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -10
```
Expected: all tests PASS (count should be ~504+)

- [ ] **Step 9: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/models/regime_model.py main.py tests/signal/test_feature_order.py && git commit -m "feat: expand _FEATURE_ORDER to 32 — kalshi_open_imbalance + btc_spx/qqq_corr_8h; wire main.py"
```

---

## Group B: Execution Logic

### Task 6: ProgressCapModel + Gate 12 dynamic threshold

**Files:**
- Modify: `btc_kalshi_system/execution/pretrade_checklist.py`
- Create: `tests/execution/test_progress_cap_model.py`
- Modify: `tests/execution/test_pretrade_checklist.py`

- [ ] **Step 1: Write failing tests for RuleBasedProgressCap**

Create `tests/execution/test_progress_cap_model.py`:
```python
from btc_kalshi_system.execution.pretrade_checklist import RuleBasedProgressCap


def test_low_vol_tight_spread_allows_20_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.001, spread=0.02, volume_ratio=1.0)
    assert result == 0.20


def test_high_vol_wide_spread_requires_5_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.005, spread=0.06, volume_ratio=1.0)
    assert result == 0.05


def test_high_vol_tight_spread_gives_10_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.005, spread=0.02, volume_ratio=1.0)
    assert result == 0.10


def test_low_vol_wide_spread_gives_10_pct():
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.001, spread=0.06, volume_ratio=1.0)
    assert result == 0.10


def test_boundary_volatility_not_high_vol():
    """Exactly at the threshold is not high vol (> not >=)."""
    cap = RuleBasedProgressCap()
    result = cap.get_cap(volatility=0.003, spread=0.06, volume_ratio=1.0)
    assert result == 0.10  # wide spread only → 0.10, not 0.05


def test_returns_float():
    cap = RuleBasedProgressCap()
    assert isinstance(cap.get_cap(0.001, 0.02, 1.0), float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_progress_cap_model.py -v
```
Expected: `ImportError: cannot import name 'RuleBasedProgressCap'`

- [ ] **Step 3: Add ProgressCapModel + RuleBasedProgressCap to pretrade_checklist.py**

At the top of `btc_kalshi_system/execution/pretrade_checklist.py`, before any existing code (after imports), add:
```python
class ProgressCapModel:
    """Interface for dynamic candle-progress entry cap. Swap RuleBasedProgressCap
    for a LogisticProgressCap once 200+ candle_features rows under regime v2 exist."""
    def get_cap(self, volatility: float, spread: float, volume_ratio: float) -> float:
        raise NotImplementedError


class RuleBasedProgressCap(ProgressCapModel):
    """Rule-based entry window cap based on BRTI volatility and Kalshi spread.

    Thresholds calibrated after 100+ candle_features rows under regime v2.
    Query: SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY brti_volatility_1h)
           FROM candle_features WHERE features_stale=0 AND brti_volatility_1h IS NOT NULL;
    """
    _HIGH_VOL = 0.003     # ~0.3% per 5min — active market
    _WIDE_SPREAD = 0.04   # >4¢ spread — thin or rapidly repricing

    def get_cap(self, volatility: float, spread: float, volume_ratio: float) -> float:
        high_vol    = volatility > self._HIGH_VOL
        wide_spread = spread    > self._WIDE_SPREAD
        if high_vol and wide_spread:
            return 0.05
        elif high_vol or wide_spread:
            return 0.10
        else:
            return 0.20


_PROGRESS_CAP_MODEL = RuleBasedProgressCap()
```

- [ ] **Step 4: Run RuleBasedProgressCap tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_progress_cap_model.py -v
```
Expected: 6 tests PASS

- [ ] **Step 5: Update Gate 12 to use the dynamic cap**

In `btc_kalshi_system/execution/pretrade_checklist.py`, replace Gate 12:
```python
# Before:
        candle_progress = (signal.regime_features or {}).get("candle_progress", 0.0) or 0.0
        if candle_progress > 0.15:
            return fail(12, f"Candle progress {candle_progress:.2f} exceeds 0.15 — entry window closed")

# After:
        candle_progress = (signal.regime_features or {}).get("candle_progress", 0.0) or 0.0
        _volatility  = (signal.regime_features or {}).get("brti_volatility_1h", 0.0) or 0.0
        _spread      = (signal.market_context  or {}).get("kalshi_spread_normalized", 0.0) or 0.0
        _vol_ratio   = (signal.regime_features or {}).get("volume_ratio_1h", 1.0) or 1.0
        _cap = _PROGRESS_CAP_MODEL.get_cap(_volatility, _spread, _vol_ratio)
        if candle_progress > _cap:
            return fail(12, (
                f"Candle progress {candle_progress:.2f} exceeds dynamic cap {_cap:.2f} "
                f"(vol={_volatility:.4f} spread={_spread:.3f})"
            ))
```

- [ ] **Step 6: Write Gate 12 pretrade_checklist tests**

Open `tests/execution/test_pretrade_checklist.py` and add:
```python
def test_gate12_quiet_market_allows_18_pct_progress(make_signal):
    """Low vol + tight spread → cap=0.20, progress 18% passes Gate 12."""
    signal = make_signal(
        regime_features={
            "candle_progress": 0.18,
            "brti_volatility_1h": 0.001,
            "volume_ratio_1h": 1.0,
        },
        market_context={"kalshi_spread_normalized": 0.02},
    )
    result = run_checklist(signal)
    # Should NOT be rejected by Gate 12 (cap=0.20, progress=0.18 <= 0.20)
    assert result.failed_gate != 12


def test_gate12_volatile_market_rejects_8_pct_progress(make_signal):
    """High vol + wide spread → cap=0.05, progress 8% fails Gate 12."""
    signal = make_signal(
        regime_features={
            "candle_progress": 0.08,
            "brti_volatility_1h": 0.005,
            "volume_ratio_1h": 1.0,
        },
        market_context={"kalshi_spread_normalized": 0.06},
    )
    result = run_checklist(signal)
    assert result.failed_gate == 12
    assert "dynamic cap 0.05" in result.failed_reason


def test_gate12_mixed_conditions_cap_is_10_pct(make_signal):
    """High vol only (tight spread) → cap=0.10."""
    signal = make_signal(
        regime_features={
            "candle_progress": 0.12,
            "brti_volatility_1h": 0.005,
            "volume_ratio_1h": 1.0,
        },
        market_context={"kalshi_spread_normalized": 0.02},
    )
    result = run_checklist(signal)
    assert result.failed_gate == 12
    assert "dynamic cap 0.10" in result.failed_reason


def test_gate12_cap_logged_in_failed_reason(make_signal):
    """failed_reason includes cap value and inputs for analysis."""
    signal = make_signal(
        regime_features={
            "candle_progress": 0.08,
            "brti_volatility_1h": 0.005,
            "volume_ratio_1h": 1.0,
        },
        market_context={"kalshi_spread_normalized": 0.06},
    )
    result = run_checklist(signal)
    assert result.failed_gate == 12
    assert "vol=" in result.failed_reason
    assert "spread=" in result.failed_reason
```

- [ ] **Step 7: Verify no existing Gate 12 tests conflict**

```bash
grep -n "gate.*12\|Gate 12\|candle_progress.*0.15\|exceeds 0.15" "/Users/ezrakornberg/Kronos V2/tests/execution/test_pretrade_checklist.py"
```

As of session 35, no Gate 12 tests exist in the file (verified: grep returns nothing). The new tests in Step 6 are the first. If this grep returns results in the future, update any test that hard-codes `candle_progress = 0.16` with the old cap: add `brti_volatility_1h=0.005, kalshi_spread_normalized=0.06` to the signal so the dynamic cap resolves to 0.05 and 0.16 still exceeds it.

- [ ] **Step 8: Run all pretrade_checklist tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_pretrade_checklist.py tests/execution/test_progress_cap_model.py -v
```
Expected: all tests PASS

- [ ] **Step 9: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -10
```
Expected: all tests PASS

- [ ] **Step 10: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/execution/pretrade_checklist.py tests/execution/test_progress_cap_model.py tests/execution/test_pretrade_checklist.py && git commit -m "feat: Gate 12 — dynamic candle progress cap via RuleBasedProgressCap (swappable interface)"
```

---

### Task 7: Mid-candle exit paper mode

**Files:**
- Modify: `main.py`
- Create: `tests/test_main_mid_candle_exit.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main_mid_candle_exit.py`:
```python
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


def _make_db_with_open_trade(direction: int, fill_price_cents: int) -> sqlite3.Connection:
    """Create an in-memory DB with one unresolved trade keyed by ticker."""
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            direction INTEGER,
            fill_price_cents INTEGER,
            outcome INTEGER
        )
    """)
    db.execute(
        "INSERT INTO trades (ticker, direction, fill_price_cents, outcome) VALUES (?,?,?,?)",
        ["KXBTC-25Jun2026-99500", direction, fill_price_cents, None],
    )
    db.commit()
    return db


def _check_would_exit(db, ticker: str, mid_candle_mid: float) -> tuple[bool, float | None]:
    """Import and call the helper under test."""
    from main import KronosV2System
    system = KronosV2System.__new__(KronosV2System)
    system._db = db
    return system._check_would_exit(ticker, mid_candle_mid)


def test_would_exit_yes_trade_when_market_drops_16_cents():
    """YES trade at 35¢: mid drops to 18¢ (35-16=19 > 18) → would_exit=True."""
    db = _make_db_with_open_trade(direction=1, fill_price_cents=35)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.18)
    assert would_exit is True
    assert abs(price - 18.0) < 0.001


def test_would_not_exit_yes_trade_when_market_drops_14_cents():
    """YES trade at 35¢: mid at 22¢ (35-14=21 < 22) → would_exit=False."""
    db = _make_db_with_open_trade(direction=1, fill_price_cents=35)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.22)
    assert would_exit is False
    assert price is None


def test_would_exit_no_trade_when_market_rises_16_cents():
    """NO trade at 30¢ fill (YES entry = 70¢): mid rises to 87¢ (70+15=85 < 87) → would_exit=True."""
    db = _make_db_with_open_trade(direction=0, fill_price_cents=30)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.87)
    assert would_exit is True


def test_would_not_exit_no_trade_when_market_rises_14_cents():
    """NO trade at 30¢ fill (YES entry = 70¢): mid at 83¢ (70+15=85 > 83) → would_exit=False."""
    db = _make_db_with_open_trade(direction=0, fill_price_cents=30)
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.83)
    assert would_exit is False


def test_no_open_trades_returns_false():
    """No unresolved trades → would_exit=False, price=None."""
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            direction INTEGER,
            fill_price_cents INTEGER,
            outcome INTEGER
        )
    """)
    db.commit()
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.18)
    assert would_exit is False
    assert price is None


def test_already_resolved_trade_not_checked():
    """Trade with outcome != NULL is not an open trade → would_exit=False."""
    db = _make_db_with_open_trade(direction=1, fill_price_cents=35)
    db.execute("UPDATE trades SET outcome=0")
    db.commit()
    would_exit, price = _check_would_exit(db, "KXBTC-25Jun2026-99500", 0.18)
    assert would_exit is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/test_main_mid_candle_exit.py -v 2>&1 | head -30
```
Expected: FAIL — `AttributeError: type object 'KronosV2System' has no attribute '_check_would_exit'`

- [ ] **Step 3: Implement _check_would_exit in main.py**

In `main.py`, add this method to `KronosV2System` (near the other private helper methods, before `_candle_logger_loop`):
```python
    _WOULD_EXIT_THRESHOLD = 0.15  # tune after 50+ resolved would-exit rows under regime v2

    def _check_would_exit(self, ticker: str, mid_candle_mid: float) -> tuple[bool, float | None]:
        """Paper exit check: would we have exited an open trade at this mid-candle price?

        Returns (would_exit, price_cents). price_cents is the mid-candle YES price in cents,
        or None when no exit would trigger.

        Finds open trades for this ticker (outcome IS NULL). Each Kalshi 15-min
        contract has a unique ticker, so this correctly scopes to the current candle.
        """
        rows = self._db.execute(
            "SELECT direction, fill_price_cents FROM trades "
            "WHERE ticker = ? AND outcome IS NULL",
            [ticker],
        ).fetchall()
        for direction, fill_price_cents in rows:
            if direction == 1:  # YES bet
                yes_entry = fill_price_cents / 100.0
                if mid_candle_mid < yes_entry - self._WOULD_EXIT_THRESHOLD:
                    return True, mid_candle_mid * 100.0
            else:  # NO bet; yes_entry = implied YES price at fill time
                yes_entry = (100 - fill_price_cents) / 100.0
                if mid_candle_mid > yes_entry + self._WOULD_EXIT_THRESHOLD:
                    return True, mid_candle_mid * 100.0
        return False, None
```

- [ ] **Step 4: Add would_exit columns to _CANDLE_FEATURES_COLUMN_MIGRATIONS**

In `main.py`, find `_CANDLE_FEATURES_COLUMN_MIGRATIONS` and add:
```python
    ("would_exit",             "INTEGER DEFAULT 0"),
    ("would_exit_price_cents", "REAL DEFAULT NULL"),
```

- [ ] **Step 5: Call _check_would_exit from _candle_logger_loop**

In `main.py`, in `_candle_logger_loop()`, in the mid-candle snapshot block (after the `_mid_candle_snaps[_in_progress_key] = {...}` dict is written), add:
```python
                                # Paper exit check: would we have exited at this price?
                                _would_exit, _would_exit_price = self._check_would_exit(
                                    _snap_ticker, (_bid + _ask) / 200.0
                                )
                                self._mid_candle_snaps[_in_progress_key]["would_exit"] = int(_would_exit)
                                self._mid_candle_snaps[_in_progress_key]["would_exit_price_cents"] = _would_exit_price
```

Then at candle close time, after `kalshi_mid_candle_progress = _mid_snap["progress"] if _mid_snap else None`, add:
```python
                would_exit             = _mid_snap.get("would_exit", 0)            if _mid_snap else 0
                would_exit_price_cents = _mid_snap.get("would_exit_price_cents")   if _mid_snap else None
```

And add both to `col_names` (after `kalshi_mid_candle_progress`) and the VALUES list.

- [ ] **Step 6: Run the new tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/test_main_mid_candle_exit.py -v
```
Expected: 6 tests PASS

- [ ] **Step 7: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -10
```
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add main.py tests/test_main_mid_candle_exit.py && git commit -m "feat: mid-candle exit paper mode — _check_would_exit logs would_exit/price to candle_features"
```

---

## Final verification

After all tasks are complete:

- [ ] **Full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -5
```
Expected: all tests PASS, count ≥ 515

- [ ] **Restart service**

```bash
launchctl kickstart -k gui/$(id -u)/com.kronos.v2
```

- [ ] **Verify candle_features schema**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "
import sqlite3
db = sqlite3.connect('trades.db')
cols = [r[1] for r in db.execute(\"PRAGMA table_info(candle_features)\").fetchall()]
for c in ['kalshi_open_imbalance', 'btc_spx_corr_8h', 'btc_qqq_corr_8h', 'would_exit', 'would_exit_price_cents']:
    print(c, '✓' if c in cols else '✗ MISSING')
"
```
Expected: all 5 columns with ✓

- [ ] **Verify _FEATURE_ORDER count**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "from btc_kalshi_system.models.regime_model import _FEATURE_ORDER; print(len(_FEATURE_ORDER))"
```
Expected: `32`

---

## Analysis queries (run after data accumulates)

**Would-exit analysis (after 50+ resolved candles):**
```sql
-- candle_features.candle_ts is the candle open ISO timestamp.
-- trades.timestamp is when the trade was placed (within the same 15-min candle).
-- Join via time window: trade was placed between candle_ts and candle_ts + 900s.
SELECT
    AVG(t.outcome) as win_rate,
    COUNT(*) as n,
    AVG(cf.would_exit_price_cents) as avg_exit_price
FROM candle_features cf
JOIN trades t ON
    t.timestamp >= cf.candle_ts AND
    t.timestamp < datetime(cf.candle_ts, '+900 seconds')
WHERE cf.would_exit = 1 AND t.outcome IS NOT NULL;
```
Target: if `win_rate < 0.40`, flip to live execution.

**Gate 12 threshold calibration (after 100+ candles):**
```sql
SELECT
    ROUND(brti_volatility_1h, 4) as vol_bucket,
    ROUND(kalshi_open_spread, 3) as spread_bucket,
    AVG(btc_direction) as dir_accuracy,
    COUNT(*) as n
FROM candle_features
WHERE features_stale = 0 AND brti_volatility_1h IS NOT NULL
GROUP BY vol_bucket, spread_bucket
ORDER BY n DESC LIMIT 20;
```

**Macro feature importance (after next regime retrain):**
```python
import joblib
m = joblib.load("models/regime.pkl")
import pandas as pd
from btc_kalshi_system.models.regime_model import _FEATURE_ORDER
fi = pd.Series(m.get_booster().get_fscore(), name="importance")
print(fi.reindex(_FEATURE_ORDER).tail(5))  # last 5 = the new features
```
