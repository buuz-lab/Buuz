# Phase 3: Calibrator + DerivativesFeed + RegimeModel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three pre-written Phase 3 test suites pass by implementing `Calibrator`, `DerivativesFeed`, and `RegimeModel` exactly to their test specifications.

**Architecture:** Tests already exist in `tests/models/test_calibrator.py`, `tests/data/test_derivatives_feed.py`, and `tests/models/test_regime_model.py` — treat them as the spec. Each implementation is a single focused file with no cross-dependencies. `DerivativesFeed` adds two constants to `config.py`. All three use joblib for persistence.

**Tech Stack:** `sklearn.isotonic.IsotonicRegression`, `xgboost.XGBClassifier`, `ccxt.async_support.binance`, `fakeredis` (tests only), `joblib`, `numpy`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `btc_kalshi_system/models/calibrator.py` | Create | `Calibrator` — isotonic regression wrapper with pass-through guard |
| `btc_kalshi_system/data/derivatives_feed.py` | Create | `DerivativesFeed` — Binance ccxt derivatives features, Redis writer |
| `btc_kalshi_system/models/regime_model.py` | Create | `RegimeModel` + `NotTrainedError` — XGBoost binary classifier |
| `config.py` | Modify | Add `REDIS_TTL_REGIME_FEATURES = 300` and `BINANCE_PERP_SYMBOL` |

---

## Task 1: Calibrator

**Interfaces required by `tests/models/test_calibrator.py`:**
- `Calibrator()` constructor
- `.fit(raw: np.ndarray, outcomes: np.ndarray)` — trains isotonic regression; no-op (pass-through mode) when `len(raw) < 500`
- `.transform(p: float) -> float` — returns calibrated probability; returns `p` unchanged in pass-through mode
- `.brier_score(raw: np.ndarray, outcomes: np.ndarray) -> float` — mean((transform(p_i) - y_i)²)
- `.save(path: str)` — joblib dump
- `Calibrator.load(path: str) -> Calibrator` — joblib load; raises `FileNotFoundError` if missing

**Files:**
- Create: `btc_kalshi_system/models/calibrator.py`
- Test: `tests/models/test_calibrator.py` (already written — do not modify)

- [ ] **Step 1: Run the existing tests to confirm they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2"
python3 -m pytest tests/models/test_calibrator.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'btc_kalshi_system.models.calibrator'`

- [ ] **Step 2: Implement calibrator.py**

Create `btc_kalshi_system/models/calibrator.py`:

```python
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

_MIN_SAMPLES = 500


class Calibrator:
    """
    Wraps IsotonicRegression for post-hoc probability calibration.
    Pass-through (identity) when fewer than _MIN_SAMPLES training examples.
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self._fitted: bool = False

    def fit(self, raw: np.ndarray, outcomes: np.ndarray) -> None:
        if len(raw) < _MIN_SAMPLES:
            self._fitted = False
            return
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw, outcomes)
        self._fitted = True

    def transform(self, p: float) -> float:
        if not self._fitted:
            return float(p)
        return float(self._iso.predict([p])[0])

    def brier_score(self, raw: np.ndarray, outcomes: np.ndarray) -> float:
        calibrated = np.array([self.transform(float(p)) for p in raw])
        return float(np.mean((calibrated - outcomes) ** 2))

    def save(self, path: str) -> None:
        joblib.dump({"iso": self._iso, "fitted": self._fitted}, path)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        data = joblib.load(path)  # raises FileNotFoundError if missing
        obj = cls()
        obj._iso = data["iso"]
        obj._fitted = data["fitted"]
        return obj
```

- [ ] **Step 3: Run all tests — 10 calibrator tests + all prior tests must pass**

```bash
python3 -m pytest tests/models/test_calibrator.py tests/data/ -v 2>&1 | tail -20
```

Expected: `10 passed` for calibrator + all data tests pass. Total suite count increases by 10.

- [ ] **Step 4: Commit**

```bash
git add btc_kalshi_system/models/calibrator.py
git commit -m "feat: Calibrator — isotonic regression wrapper with pass-through guard and brier score"
```

---

## Task 2: DerivativesFeed

**Interfaces required by `tests/data/test_derivatives_feed.py`:**

The tests use `DerivativesFeed.__new__(DerivativesFeed)` then manually inject `feed._redis = fakeredis.FakeRedis()` and `feed._exchange = MagicMock()`. This means `__init__` must set `self._redis` and `self._exchange` (the tests bypass `__init__` but the attributes must exist on the class).

Pure computation methods (tested without mocking):
- `._funding_rate_trend(history: list[dict]) -> float`
  - `history` entries: `{"timestamp": int_ms, "fundingRate": float}`
  - Returns `history[-1]["fundingRate"] - history[0]["fundingRate"]`
  - Returns `0.0` when `len(history) < 2`
- `._oi_delta_pct(prev_oi: float, curr_oi: float) -> float`
  - Returns `(curr_oi - prev_oi) / prev_oi`
  - Returns `0.0` when `prev_oi == 0.0`
- `._cvd_normalized(trades: list[dict]) -> float`
  - `trades` entries: `{"amount": float, "side": "buy"|"sell"}`
  - Returns `(buy_vol - sell_vol) / (buy_vol + sell_vol)`
  - Returns `0.0` for empty trades or zero total volume
- `._brti_volatility_1h() -> float`
  - Reads `self._redis.lrange("brti:ticks", 0, -1)`
  - Each entry is `b"timestamp:price"` (same format as FeatureStore)
  - Filters to entries where `now - float(timestamp) <= 3600`
  - Returns `std(prices, ddof=1) / mean(prices)` (coefficient of variation)
  - Returns `0.0` if fewer than 2 ticks within window
- `._write_features(features: dict) -> None`
  - `self._redis.set("regime:features", json.dumps(features), ex=300)`

Also add two constants to `config.py`:
```python
REDIS_TTL_REGIME_FEATURES: int = 300
BINANCE_PERP_SYMBOL: str = "BTC/USDT:USDT"
```

**Files:**
- Modify: `config.py`
- Create: `btc_kalshi_system/data/derivatives_feed.py`
- Test: `tests/data/test_derivatives_feed.py` (already written — do not modify)

- [ ] **Step 1: Run the existing tests to confirm they fail**

```bash
python3 -m pytest tests/data/test_derivatives_feed.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'btc_kalshi_system.data.derivatives_feed'`

- [ ] **Step 2: Add constants to config.py**

Append these two lines to `config.py`:

```python
REDIS_TTL_REGIME_FEATURES: int = 300
BINANCE_PERP_SYMBOL: str = "BTC/USDT:USDT"
```

- [ ] **Step 3: Implement derivatives_feed.py**

Create `btc_kalshi_system/data/derivatives_feed.py`:

```python
import asyncio
import json
import time

import redis
from loguru import logger

from config import BINANCE_PERP_SYMBOL, REDIS_TTL_REGIME_FEATURES, REDIS_URL


class DerivativesFeed:
    """
    Pulls Binance USDM perpetual derivatives data every 5 minutes and writes
    six regime features to Redis key "regime:features" with a 300-second TTL.

    Pure computation methods (_funding_rate_trend, _oi_delta_pct, _cvd_normalized,
    _brti_volatility_1h) are stateless and take explicit arguments for testability.
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._redis = redis.from_url(redis_url)
        self._exchange = None  # set in run()

    async def run(self) -> None:
        """Connect to Binance and refresh features every 5 minutes. Runs forever."""
        import ccxt.async_support as ccxt_async

        async with ccxt_async.binance({"enableRateLimit": True}) as exchange:
            self._exchange = exchange
            while True:
                try:
                    features = await self._fetch_features()
                    self._write_features(features)
                    logger.info("DerivativesFeed: features written to Redis")
                except Exception as exc:
                    logger.warning(f"DerivativesFeed: fetch failed — {exc}")
                await asyncio.sleep(REDIS_TTL_REGIME_FEATURES)

    async def _fetch_features(self) -> dict:
        symbol = BINANCE_PERP_SYMBOL
        funding_task = asyncio.create_task(self._exchange.fetch_funding_rate(symbol))
        oi_task = asyncio.create_task(self._exchange.fetch_open_interest(symbol))
        trades_task = asyncio.create_task(self._exchange.fetch_trades(symbol, limit=500))
        funding_history_task = asyncio.create_task(
            self._exchange.fetch_funding_rate_history(symbol, limit=5)
        )

        funding, oi, trades, funding_history = await asyncio.gather(
            funding_task, oi_task, trades_task, funding_history_task
        )

        funding_rate = float(funding["fundingRate"])
        mark_price = float(funding.get("markPrice") or 0.0)
        curr_oi = float(oi.get("openInterestAmount") or oi.get("openInterest") or 0.0)

        # OI delta: compare against previous fetch stored in Redis
        prev_oi_raw = self._redis.get("regime:prev_oi")
        prev_oi = float(prev_oi_raw) if prev_oi_raw else 0.0
        self._redis.set("regime:prev_oi", curr_oi, ex=600)

        # Basis: (perp mark price - BRTI) / BRTI
        brti_raw = self._redis.get("brti:resolution_estimate")
        brti = float(brti_raw) if brti_raw else 0.0
        basis_spread_pct = (mark_price - brti) / brti if brti else 0.0

        return {
            "funding_rate": funding_rate,
            "funding_rate_trend": self._funding_rate_trend(funding_history),
            "oi_delta_pct": self._oi_delta_pct(prev_oi, curr_oi),
            "cvd_normalized": self._cvd_normalized(trades),
            "basis_spread_pct": basis_spread_pct,
            "brti_volatility_1h": self._brti_volatility_1h(),
        }

    # ── Pure computation methods (tested independently) ──────────────────────

    def _funding_rate_trend(self, history: list[dict]) -> float:
        """4h delta of funding rate. history entries: {"timestamp": ms, "fundingRate": float}."""
        if len(history) < 2:
            return 0.0
        return float(history[-1]["fundingRate"]) - float(history[0]["fundingRate"])

    def _oi_delta_pct(self, prev_oi: float, curr_oi: float) -> float:
        """Fractional change in open interest: (curr - prev) / prev."""
        if prev_oi == 0.0:
            return 0.0
        return (curr_oi - prev_oi) / prev_oi

    def _cvd_normalized(self, trades: list[dict]) -> float:
        """Cumulative volume delta normalized by total volume: (buys - sells) / total."""
        buy_vol = sum(float(t["amount"]) for t in trades if t.get("side") == "buy")
        sell_vol = sum(float(t["amount"]) for t in trades if t.get("side") == "sell")
        total = buy_vol + sell_vol
        if total == 0.0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def _brti_volatility_1h(self) -> float:
        """Coefficient of variation of BRTI tick prices over the past hour."""
        now = time.time()
        raw_ticks = self._redis.lrange("brti:ticks", 0, -1)
        prices = []
        for entry in raw_ticks:
            try:
                ts_str, price_str = entry.decode().split(":", 1)
                if now - float(ts_str) <= 3600:
                    prices.append(float(price_str))
            except (ValueError, AttributeError):
                continue
        if len(prices) < 2:
            return 0.0
        arr = np.array(prices)
        return float(np.std(arr, ddof=1) / np.mean(arr))

    def _write_features(self, features: dict) -> None:
        self._redis.set("regime:features", json.dumps(features), ex=REDIS_TTL_REGIME_FEATURES)
```

The `import numpy as np` line is missing — add it at the top after `import redis`:

```python
import asyncio
import json
import time

import numpy as np
import redis
from loguru import logger

from config import BINANCE_PERP_SYMBOL, REDIS_TTL_REGIME_FEATURES, REDIS_URL
```

- [ ] **Step 4: Run all tests — 12 derivatives_feed tests must pass**

```bash
python3 -m pytest tests/data/test_derivatives_feed.py -v
```

Expected: `12 passed`

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
python3 -m pytest tests/ -v --ignore=tests/models/test_regime_model.py 2>&1 | tail -10
```

Expected: all calibrator + all data tests pass (regime_model still failing — that's expected)

- [ ] **Step 6: Commit**

```bash
git add config.py btc_kalshi_system/data/derivatives_feed.py
git commit -m "feat: DerivativesFeed — Binance perp funding/OI/CVD features written to Redis every 5min"
```

---

## Task 3: RegimeModel

**Interfaces required by `tests/models/test_regime_model.py`:**

- `class NotTrainedError(RuntimeError)` — must be `RuntimeError` subclass
- `class RegimeModel`:
  - `.__init__()` — sets `self._model = None`
  - `.train(X: np.ndarray, y: np.ndarray)` — fits `XGBClassifier`; `X` has 6 columns matching `FEATURE_KEYS`
  - `.get_regime(features: dict) -> dict` — raises `NotTrainedError` if `_model is None`; returns `{"prob_up": float, "direction": int, "confidence": float}`
  - `direction = 1` if `prob_up >= 0.5`, else `0`
  - `confidence = abs(prob_up - 0.5) * 2` (0.0 at 50/50, 1.0 at 0% or 100%)
  - `.save(path: str)` — joblib dump of `_model`
  - `RegimeModel.load(path: str) -> RegimeModel` — joblib load; raises `FileNotFoundError` if missing

Feature key order (matches `_synthetic_features` shape of 6 columns):
```python
FEATURE_KEYS = [
    "funding_rate", "funding_rate_trend", "oi_delta_pct",
    "cvd_normalized", "basis_spread_pct", "brti_volatility_1h",
]
```

**Files:**
- Create: `btc_kalshi_system/models/regime_model.py`
- Test: `tests/models/test_regime_model.py` (already written — do not modify)

- [ ] **Step 1: Run the existing tests to confirm they fail**

```bash
python3 -m pytest tests/models/test_regime_model.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'btc_kalshi_system.models.regime_model'`

- [ ] **Step 2: Implement regime_model.py**

Create `btc_kalshi_system/models/regime_model.py`:

```python
import numpy as np
import joblib
import xgboost as xgb

FEATURE_KEYS = [
    "funding_rate",
    "funding_rate_trend",
    "oi_delta_pct",
    "cvd_normalized",
    "basis_spread_pct",
    "brti_volatility_1h",
]


class NotTrainedError(RuntimeError):
    """Raised when get_regime() is called before a model has been trained or loaded."""


class RegimeModel:
    """
    XGBoost binary classifier over six derivative/volatility regime features.
    Call train() before get_regime(). Save/load via joblib.
    """

    def __init__(self) -> None:
        self._model: xgb.XGBClassifier | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
        )
        self._model.fit(X, y)

    def get_regime(self, features: dict) -> dict:
        if self._model is None:
            raise NotTrainedError("Call train() or load() before get_regime()")
        x = np.array([[features[k] for k in FEATURE_KEYS]], dtype=np.float32)
        prob_up = float(self._model.predict_proba(x)[0, 1])
        direction = 1 if prob_up >= 0.5 else 0
        confidence = float(abs(prob_up - 0.5) * 2)
        return {"prob_up": prob_up, "direction": direction, "confidence": confidence}

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "RegimeModel":
        obj = cls()
        obj._model = joblib.load(path)  # raises FileNotFoundError if path missing
        return obj
```

- [ ] **Step 3: Run regime_model tests — all 10 must pass**

```bash
python3 -m pytest tests/models/test_regime_model.py -v
```

Expected: `10 passed`

- [ ] **Step 4: Run full test suite — all tests must pass**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -15
```

Expected: all tests pass (40 existing + 10 calibrator + 12 derivatives + 10 regime = 72 total)

- [ ] **Step 5: Commit**

```bash
git add btc_kalshi_system/models/regime_model.py
git commit -m "feat: RegimeModel — XGBoost classifier over 6 regime features, NotTrainedError guard"
```

---

## Self-Review

**Spec coverage:**
- ✅ `Calibrator` — fit/transform/brier_score/save/load, pass-through < 500
- ✅ `DerivativesFeed` — _funding_rate_trend (4h delta), _oi_delta_pct, _cvd_normalized, _brti_volatility_1h, _write_features (300s TTL)
- ✅ `DerivativesFeed.run()` — async loop, ccxt async Binance, 5-min refresh
- ✅ `RegimeModel` — XGBoost, get_regime dict with prob_up/direction/confidence, save/load
- ✅ `NotTrainedError(RuntimeError)` — RuntimeError subclass
- ✅ `direction` — integer 0/1 (not string)
- ✅ `REDIS_TTL_REGIME_FEATURES` and `BINANCE_PERP_SYMBOL` in config.py
- ✅ Tests do not modify pre-written test files

**Placeholder scan:** All code blocks are complete. No TBD or TODO.

**Type consistency:**
- `_funding_rate_trend` takes `list[dict]` in Task 2 and `_fetch_features` passes `funding_history` (the raw ccxt funding history list) — consistent
- `FEATURE_KEYS` in `regime_model.py` matches the 6 keys in `_feature_dict()` test fixture exactly
- `direction` is `int` (0 or 1) — consistent with test assertion `result["direction"] in (0, 1)`
- `Calibrator.load` and `RegimeModel.load` both rely on joblib raising `FileNotFoundError` natively — consistent
