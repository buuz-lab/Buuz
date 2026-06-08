# Regime V2 New Features: k15_kalshi_alignment + k15_delta Design — 2026-06-08

## Goal

Add two new features to regime v2 that make the k15/Kalshi interaction visible to the model. Derived from observed trade data showing high-conviction k15 catastrophically underperforms when Kalshi agrees (-$0.09/trade), while low-conviction + Kalshi-agreement is the best bucket (+$0.10/trade).

---

## Feature 1: `k15_kalshi_alignment`

**Formula:** `(kronos_raw_15min - 0.5) * (kalshi_open_mid - 0.5)`

**What it captures:** The product of k15's deviation from neutral and Kalshi's deviation from neutral. Positive = both lean same direction. Negative = they disagree. Magnitude encodes how strongly each signal commits.

**Why it separates the buckets:**

| Scenario | Value | Observed P&L |
|---|---|---|
| High conv k15 (0.85) + Kalshi agrees (0.65) | +0.053 | -$0.056/trade |
| High conv k15 (0.85) + Kalshi disagrees (0.38) | -0.042 | -$0.001/trade |
| Low conv k15 (0.58) + Kalshi agrees (0.57) | +0.006 | +$0.103/trade |
| Low conv k15 (0.58) + Kalshi disagrees (0.43) | -0.006 | -$0.083/trade |

XGBoost can learn "small positive → best; large positive → worst" with a single split.

**Circularity:** Not circular. The feature is a derived interaction term, not raw Kalshi price. The model trains on `btc_direction` (ground truth), not Kalshi's price. It can't simply learn to output Kalshi's price from this feature.

---

## Feature 2: `k15_delta`

**Formula:** `kronos_raw_15min_current_candle - kronos_raw_15min_prior_candle`

**What it captures:** How much Kronos's 15-min prediction changed from the prior candle. Near zero = k15 stalling at the same extreme (trend-following, already priced in). Large change = fresh signal (Kronos updated its view).

**Why it matters:** The highest-conviction k15 losses (-$0.09/trade) come from k15 "stalling" — outputting 0.85+ for multiple consecutive candles while Kalshi has already priced that view in. A large positive delta for a bullish k15 means Kronos just became bullish, not that it's been stuck.

---

## Implementation

### `btc_kalshi_system/models/regime_model.py`

Add to `_FEATURE_ORDER` after `"recent_up_fraction"`:
```python
    "k15_kalshi_alignment",
    "k15_delta",
```
Feature count: 39 → 41.

### `btc_kalshi_system/signal/fusion.py`

**`__init__`:** Add `self._k15_delta: float | None = None`

**`get_signal()`:** Before updating `_last_kronos_raw_15min`, compute delta:
```python
if kronos_raw_15min is not None and self._last_kronos_raw_15min is not None:
    self._k15_delta = round(kronos_raw_15min - self._last_kronos_raw_15min, 4)
else:
    self._k15_delta = None
self._last_kronos_raw_15min = kronos_raw_15min
```

**`_regime_features()`:** Add both features at the end of the returned dict:
```python
# k15/Kalshi interaction features
_k15 = self._last_kronos_raw_15min
_kalshi_mid = self._market_context.get("kalshi_mid_cents")
_k15_kalshi_alignment = (
    round((_k15 - 0.5) * (_kalshi_mid / 100.0 - 0.5), 4)
    if _k15 is not None and _kalshi_mid is not None
    else None
)
...
"k15_kalshi_alignment": _k15_kalshi_alignment,
"k15_delta":            self._k15_delta,
```

### `main.py`

**Schema migrations:** Add to `_CANDLE_FEATURES_COLUMN_MIGRATIONS`:
```python
("k15_kalshi_alignment", "REAL DEFAULT NULL"),
("k15_delta",            "REAL DEFAULT NULL"),
```

**Instance var in `__init__`:** After `self._candle_open_brti = {}`:
```python
self._prev_k15_logged: float | None = None
```

**Candle close logging in `_candle_logger_loop`:** At candle close, compute:
```python
k15_now = features.get("kronos_raw_15min")
kalshi_open = kalshi_open_mid  # already read from _open_snap

k15_kalshi_alignment = (
    round((k15_now - 0.5) * (kalshi_open - 0.5), 4)
    if k15_now is not None and kalshi_open is not None
    else None
)
k15_delta = (
    round(k15_now - self._prev_k15_logged, 4)
    if k15_now is not None and self._prev_k15_logged is not None
    else None
)
if k15_now is not None:
    self._prev_k15_logged = k15_now
```

Add both to the INSERT column list and VALUES tuple.

**One-time backfill:** Run immediately after schema migration applies (on restart), before the first candle is logged:
```python
self._db.execute("""
    UPDATE candle_features
    SET k15_kalshi_alignment = ROUND((kronos_raw_15min - 0.5) * (kalshi_open_mid - 0.5), 4)
    WHERE kronos_raw_15min IS NOT NULL AND kalshi_open_mid IS NOT NULL
      AND k15_kalshi_alignment IS NULL
""")
self._db.execute("""
    UPDATE candle_features AS c1
    SET k15_delta = ROUND(c1.kronos_raw_15min - (
        SELECT c2.kronos_raw_15min FROM candle_features c2
        WHERE c2.candle_ts < c1.candle_ts AND c2.kronos_raw_15min IS NOT NULL
        ORDER BY c2.candle_ts DESC LIMIT 1
    ), 4)
    WHERE c1.kronos_raw_15min IS NOT NULL AND c1.k15_delta IS NULL
""")
self._db.commit()
```

### `tests/signal/test_feature_order.py`

No manual changes needed. The test checks that `_FEATURE_ORDER` (regime_model.py), `_FEATURE_COLS` (train_regime.py, auto-derived from `list(_FEATURE_ORDER)`), and `fusion._regime_features()` keys are all identical. Adding both features to `_FEATURE_ORDER` and `_regime_features()` makes the test pass automatically.

---

## Data Availability After Deploy

- `k15_kalshi_alignment`: Backfills ~380 rows immediately (all rows with both `kronos_raw_15min` and `kalshi_open_mid` non-null)
- `k15_delta`: Backfills ~370 rows (all rows with non-null `kronos_raw_15min` and a prior row)
- Both NULL-safe — XGBoost treats NULLs as missing, no model breakage

---

## Files Changed

| File | Change |
|---|---|
| `btc_kalshi_system/models/regime_model.py` | `_FEATURE_ORDER` 39 → 41 |
| `btc_kalshi_system/signal/fusion.py` | `_k15_delta` init, delta computation in `get_signal()`, both features in `_regime_features()` |
| `main.py` | 2 schema migrations, `_prev_k15_logged` instance var, compute + log both at candle close, one-time backfill |
| `tests/signal/test_feature_order.py` | No count change — test checks consistency between `_FEATURE_ORDER`, `_FEATURE_COLS`, and `fusion._regime_features()`. Will pass automatically once all three are updated. |
