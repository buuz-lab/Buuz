# Four Sophistication Gaps — Design Spec
_2026-06-06 | Session 35_

## Context

System state at design time: 518/672 qualifying rows, regime v2 dry-run showing CV Brier 0.211 and +17.9% Kalshi edge advantage on 252 rows. Full train fires ~June 8-9 via cron. Gates 12/13/14 live since June 4; early-entry (0-5% progress) running at +$0.536/trade.

Four gaps identified in session 34 handoff. Implementing all four in two parallel groups:
- **Group A (Feature Pipeline):** Gap 1 (Kalshi order imbalance) + Gap 4 (Macro correlation)
- **Group B (Execution Logic):** Gap 2 (Mid-candle exit, paper mode) + Gap 3 (Dynamic entry gate, rule-based)

---

## Group A: Feature Pipeline

### Gap 1 — Kalshi Order Imbalance (`kalshi_open_imbalance`)

**Purpose:** Capture whether the Kalshi orderbook at candle open is skewed toward buyers (informed demand) or sellers (informed supply). Range: -1 (all asks) to +1 (all bids).

**Formula:**
```
imbalance = (depth_bid - depth_ask) / (depth_bid + depth_ask)
```
Returns `None` when total depth is 0 (REST fallback or no WS snapshot yet).

**Key finding:** `depth_bid` and `depth_ask` are already stored in `_open_snapshots` by the WS snapshot path (`kalshi_orderbook_feed.py` lines 113-114). No new data capture required.

**Files and changes:**

| File | Change |
|------|--------|
| `btc_kalshi_system/execution/kalshi_orderbook_feed.py` | `get_open_snapshot()` computes and returns `depth_imbalance` from existing `depth_bid`/`depth_ask`. One line added to return dict. REST fallback (`set_open_snapshot`) keeps `depth_bid=0.0`, `depth_ask=0.0` → imbalance=None. |
| `btc_kalshi_system/signal/fusion.py` | Add `self._last_kalshi_open_imbalance = None` to `__init__`. Add `set_kalshi_imbalance(v)` setter. `_regime_features()` includes `"kalshi_open_imbalance": self._last_kalshi_open_imbalance`. |
| `main.py` | In `_process_market()`, after `get_open_snapshot()`, call `self._fusion.set_kalshi_imbalance(_open_snap["depth_imbalance"])`. Candle logger reads it from snapshot dict for `candle_features` INSERT. Add `("kalshi_open_imbalance", "REAL DEFAULT NULL")` to `_CANDLE_FEATURES_COLUMN_MIGRATIONS`. |
| `btc_kalshi_system/models/regime_model.py` | Add `"kalshi_open_imbalance"` to `_FEATURE_ORDER` after `"skew_25d"` (becomes feature 30). XGBoost handles `None→NaN` natively. |

**Backfill:** Not possible — historical rows get `NULL`, treated as missing by XGBoost. Only new rows from deploy forward will be populated.

**Tests:**
- `tests/execution/test_kalshi_orderbook_feed.py` — `get_open_snapshot()` returns `depth_imbalance`; correct formula; `None` when total=0.
- `tests/signal/test_feature_order.py` — feature count 29→30; `kalshi_open_imbalance` present.
- `tests/signal/test_fusion.py` — `set_kalshi_imbalance()` updates `_regime_features()` output.

---

### Gap 4 — Macro Correlation (`btc_spx_corr_8h`, `btc_qqq_corr_8h`)

**Purpose:** 8-hour rolling correlation of BTC hourly returns vs SPX and QQQ. Signals when BTC is trading as a macro risk asset (high correlation) vs independently (low correlation). Helps the regime model anticipate direction moves that crypto-native features (CVD, funding) see only with a lag.

**Formula:**
```python
# For each ticker in ["^GSPC", "QQQ"]:
data = yf.download([ticker, "BTC-USD"], period="5d", interval="1h")
corr = data["BTC-USD"].pct_change().rolling(8).corr(data[ticker].pct_change())
result = float(corr.iloc[-1])  # last value; NaN → 0.0
```

**Out-of-hours behavior:** 8h rolling window spans the prior US session's close even during overnight BTC candles. Not frozen (like option A) and not zeroed (like option C). The correlation value degrades gracefully as the window moves further from US market hours.

**Both tickers:** `^GSPC` (broad macro) and `QQQ` (tech/sentiment, historically tighter crypto correlation). They diverge during value/growth rotations. XGBoost will surface which is more predictive via feature importance after a few weeks.

**New file:** `btc_kalshi_system/data/macro_feed.py`

```python
class MacroFeed:
    def get_correlations(self) -> dict:
        # Returns {"btc_spx_corr_8h": float, "btc_qqq_corr_8h": float}
        # 0.0 on any failure (yfinance timeout, parse error, NaN result)
```

**Fetch strategy:** Called from `derivatives_feed.py`'s `_fetch_features()`. 15-minute in-memory cache (`_last_macro_ts`, `_last_macro_values`) — yfinance 1h data doesn't change faster than that. On failure: returns last cached values, or `{0.0, 0.0}` if never successfully fetched. Failure is silent (DEBUG log only); trading is never blocked by a macro feed outage.

**Files and changes:**

| File | Change |
|------|--------|
| `btc_kalshi_system/data/macro_feed.py` | **New.** `MacroFeed` class with `get_correlations()`. In-class cache. Requires `yfinance` added to dependencies. |
| `btc_kalshi_system/data/derivatives_feed.py` | Instantiate `MacroFeed()` in `__init__`. Call `macro_feed.get_correlations()` in `_fetch_features()` with 15-min cache guard. Merge into returned features dict. |
| `btc_kalshi_system/models/regime_model.py` | Add `"btc_spx_corr_8h"` and `"btc_qqq_corr_8h"` to `_FEATURE_ORDER` (features 31, 32). |
| `main.py` | Add both to `_CANDLE_FEATURES_COLUMN_MIGRATIONS`. |

**Tests:**
- `tests/data/test_macro_feed.py` — returns dict with both keys; returns 0.0 on yfinance failure (mock); uses cache within 15 min; re-fetches after 15 min.
- `tests/signal/test_feature_order.py` — feature count 30→32.

**Dependency:** Add `yfinance` to `requirements.txt`. No API key required.

---

## Group B: Execution Logic

### Gap 2 — Mid-Candle Exit (Paper Mode)

**Purpose:** Detect when the Kalshi market has moved strongly against an open position by mid-candle (40-60% progress). Paper mode: log `would_exit=1` without placing any order. After 50+ resolved rows, analyze win rate of would-exit candles. If negative, flip to live execution.

**Exit threshold (tunable constant):**
```python
_WOULD_EXIT_THRESHOLD = 0.15  # Kalshi must move 15¢ against us to trigger
```

**Check logic:**
```python
# For YES (direction=1): yes_entry = fill_price_cents / 100.0
#   would_exit if kalshi_mid_candle_mid < yes_entry - _WOULD_EXIT_THRESHOLD

# For NO (direction=0): yes_entry = (100 - fill_price_cents) / 100.0
#   would_exit if kalshi_mid_candle_mid > yes_entry + _WOULD_EXIT_THRESHOLD
```

**Schema additions** (`_CANDLE_FEATURES_COLUMN_MIGRATIONS`):
- `would_exit INTEGER DEFAULT 0`
- `would_exit_price_cents REAL DEFAULT NULL` — the YES mid-price (in cents) at the time of would-be exit

**Implementation:**
- New `_check_would_exit(candle_ts, mid_candle_mid)` helper in `main.py`. Queries `trades` table for open positions (`outcome IS NULL`) in the current candle. Computes exit condition. Returns `(would_exit: bool, price_cents: float | None)`.
- Called from `_candle_logger_loop` at the point where the mid-candle snapshot is already captured (no new timing logic needed).
- Writes result to `candle_features` alongside existing mid-candle columns.

**Analysis query (ready when data accumulates):**
```sql
SELECT AVG(t.outcome), COUNT(*), AVG(cf.would_exit_price_cents)
FROM candle_features cf
JOIN trades t ON cf.candle_ts = t.candle_open_ts
WHERE cf.would_exit = 1 AND t.outcome IS NOT NULL
```
When `AVG(outcome)` is sufficiently bad (target: <40% win rate on would-exit candles), flip to live: replace the `candle_features` write with `router.place_order()`.

**Flip to live (future):**
- Add `exited_early INTEGER DEFAULT 0` and `exit_price_cents REAL DEFAULT NULL` to `trades` schema.
- Replace paper log with actual offsetting order via `router`.
- No gate or signal changes required.

**Tests:**
- `tests/test_main_mid_candle_exit.py` — YES trade: mid drops 16¢ → would_exit=1; mid drops 14¢ → would_exit=0. NO trade: mid rises 16¢ → would_exit=1. No open trades → would_exit=0. Already-resolved trade (outcome not NULL) → not checked.

---

### Gap 3 — Dynamic Entry Gate (Rule-Based, Swappable)

**Purpose:** Replace Gate 12's hard 15% candle progress cap with a threshold that adapts to market conditions. Quiet markets (low vol, tight spread) preserve edge longer — allow up to 20%. Active markets reprice fast — require entry by 5%.

**Swappable interface** (new class at top of `pretrade_checklist.py`):
```python
class ProgressCapModel:
    def get_cap(self, volatility: float, spread: float, volume_ratio: float) -> float:
        raise NotImplementedError
```

**Rule-based implementation (ships now):**
```python
class RuleBasedProgressCap(ProgressCapModel):
    _HIGH_VOL    = 0.003   # ~0.3% per 5min — active market
    _WIDE_SPREAD = 0.04    # >4¢ spread — thin or rapidly repricing

    def get_cap(self, volatility, spread, volume_ratio):
        high_vol    = volatility > self._HIGH_VOL
        wide_spread = spread    > self._WIDE_SPREAD
        if high_vol and wide_spread:  return 0.05
        elif high_vol or wide_spread: return 0.10
        else:                         return 0.20

_PROGRESS_CAP_MODEL = RuleBasedProgressCap()
```

**Gate 12 change** (one line replaces the hard `0.15`):
```python
_volatility  = (signal.regime_features or {}).get("brti_volatility_1h", 0.0) or 0.0
_spread      = (signal.market_context  or {}).get("kalshi_spread_normalized", 0.0) or 0.0
_vol_ratio   = (signal.regime_features or {}).get("volume_ratio_1h", 1.0) or 1.0
_cap = _PROGRESS_CAP_MODEL.get_cap(_volatility, _spread, _vol_ratio)
if candle_progress > _cap:
    return fail(12, f"Candle progress {candle_progress:.2f} exceeds dynamic cap {_cap:.2f} "
                    f"(vol={_volatility:.4f}, spread={_spread:.3f})")
```

**Swap path to learned model (future):**
When 200+ candle_features rows under regime v2 exist:
```python
class LogisticProgressCap(ProgressCapModel):
    def __init__(self, model_path):
        self._model = joblib.load(model_path)
    def get_cap(self, volatility, spread, volume_ratio):
        return float(self._model.predict([[volatility, spread, volume_ratio]])[0])

_PROGRESS_CAP_MODEL = LogisticProgressCap("models/progress_cap.pkl")
```
Zero changes to Gate 12 logic. Zero schema changes.

**Data collection for learned model** (automatic from day one):
`candle_features` already logs `brti_volatility_1h`, `kalshi_open_spread`, `volume_ratio_1h`, `candle_progress`, and `btc_direction`. The training query when ready:
```sql
SELECT brti_volatility_1h, kalshi_open_spread, volume_ratio_1h,
       candle_progress, btc_direction
FROM candle_features
WHERE features_stale = 0 AND brti_volatility_1h IS NOT NULL
```

**Threshold calibration:** `_HIGH_VOL=0.003` and `_WIDE_SPREAD=0.04` are initial estimates. After 100+ rows under regime v2, compute percentile breakpoints on `brti_volatility_1h` and `kalshi_open_spread` from `candle_features` to tune. Logged in the `failed_reason` field so every Gate 12 rejection includes the dynamic cap value and inputs.

**Tests:**
- `tests/execution/test_pretrade_checklist.py` — low vol + tight spread → cap=0.20, progress 18% passes; high vol + wide spread → cap=0.05, progress 8% fails; mixed → cap=0.10. Cap logged in failed_reason.
- `tests/execution/test_progress_cap_model.py` — `RuleBasedProgressCap` all three branches. Interface contract: `get_cap()` returns float between 0 and 1.

---

## Implementation Order (within each group)

**Group A:**
1. `macro_feed.py` (new file, isolated, easy to test first)
2. `derivatives_feed.py` integration
3. `kalshi_orderbook_feed.py` imbalance computation
4. `fusion.py` cache + setter
5. `main.py` wiring + migrations
6. `regime_model.py` feature order (last — all upstream must be correct first)

**Group B:**
1. `RuleBasedProgressCap` class + interface in `pretrade_checklist.py`
2. Gate 12 wiring
3. `_check_would_exit()` helper in `main.py`
4. Schema migrations for `would_exit` columns
5. `_candle_logger_loop` call

---

## What NOT included

- Live exit execution (Gap 2 flip) — gated behind 50+ resolved would-exit rows
- `LogisticProgressCap` model (Gap 3 swap) — gated behind 200+ candle_features rows under regime v2
- Phase 3c calibrator — separate trigger (200+ live regime trades)
- Gate 2 enforcement — separate trigger (~50 shadow trades)
