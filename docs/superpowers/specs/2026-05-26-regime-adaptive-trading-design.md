# Regime-Adaptive Trading Design
**Date:** 2026-05-26  
**Status:** Draft  
**Problem:** BTC dropped ~20% (May 20–26). Kronos stayed bearish. Market bounced. Calibrator in passthrough (wrong labels, never persisted, mixes regimes). Gate 8 (Kalshi consensus) missing. Drift monitor fires but nothing listens to it. Result: 3-day loss streak, win rates 38–42%, $347 loss on May 23 alone.

---

## Root Cause Summary

Three compounding failures:

1. **Calibrator has never worked correctly.** Four bugs ensure it stays in passthrough and would produce wrong output even if it activated: wrong training labels (`outcome` vs `y_up`), no persistence across restarts, no rolling window (mixes regimes), and `_MIN_SAMPLES=500` too high.

2. **No market-consensus gate.** When Kalshi prices UP and Kronos bets DOWN, the trade wins only 23.9% of the time. This is the single most predictive signal available and there's no gate on it.

3. **Drift detection fires but nothing listens.** `CalibrationDriftMonitor` correctly detected the May 23 regime shift (Brier jumped 0.21 → 0.46). But `is_drifting()` is not wired to Kelly, bootstrap shrink, or trade suppression. Detection without action is noise.

---

## Layer 1 — Calibrator (6 fixes)

### 1a. Fix training labels — `main.py:1166`

**Bug:** `SELECT kronos_raw, outcome` uses `outcome` (trade win = 0/1) as the calibration Y. For NO trades, `outcome=1` means market went DOWN, but `kronos_raw` = P(market UP). The calibrator receives contradictory labels across YES/NO trades and would produce garbage output.

**Fix:** SELECT `direction` alongside `kronos_raw, outcome`. Compute correct labels in Python:
```python
y_up = (directions == outcomes).astype(float)
# direction=0, outcome=0 (NO loss = market UP)   → y_up=1 ✓
# direction=0, outcome=1 (NO win  = market DOWN)  → y_up=0 ✓  
# direction=1, outcome=1 (YES win = market UP)    → y_up=1 ✓
# direction=1, outcome=0 (YES loss = market DOWN) → y_up=0 ✓
```

### 1b. Add rolling window + stale filter

**Bug:** Query fetches ALL resolved trades, blending May 20–22 (BTC trending down, 60–68% win) with May 23–26 (mean-reversion, 37–42% win). Isotonic regression averages contradictory regimes.

**Fix:** 
```python
rows = self._db.execute(
    "SELECT kronos_raw, direction, outcome FROM trades "
    "WHERE outcome IS NOT NULL AND features_stale=0 "
    "ORDER BY timestamp DESC LIMIT 300"
).fetchall()
```
300 rows ≈ 5.5 days at current volume. Captures the current regime without blending multiple shifts.

### 1c. Lower `_MIN_SAMPLES`: 500 → 300 — `calibrator.py:7`

With 460 resolved trades and ~54/day, the calibrator activates within 1 day of deployment instead of never.

### 1d. Persistence: save/load — `config.py` + `calibrator.py` + `main.py`

- Add `CALIBRATOR_MODEL_PATH: str = os.getenv("CALIBRATOR_MODEL_PATH", "models/calibrator.pkl")` to `config.py`
- `KronosV2.__init__()`: after `self._calibrator = Calibrator()`, attempt `self._calibrator = Calibrator.load(config.CALIBRATOR_MODEL_PATH)` in try/except FileNotFoundError
- After each successful refit in `_check_resolutions`: call `self._calibrator.save(config.CALIBRATOR_MODEL_PATH)` (mkdir-safe: `os.makedirs("models", exist_ok=True)`)

### 1e. Refit cadence: every 25 resolutions, not every single one

Track `calibration_drift:pending_refits` in Redis (INCR on each resolution, reset to 0 after refit). Only refit when counter hits 25. Keeps calibration fresh (~6h cadence) without a refit per trade.

### 1f. Add `scripts/train_calibrator.py`

Standalone script mirroring `train_regime.py`: `--db`, `--out`, `--window` (default 300), `--dry-run`. Reports Brier score before and after. Allows manual calibration check before go-live. Use `os.makedirs("models", exist_ok=True)` before save.

**Monotonicity guard:** After each refit, compare new Brier on training data against previous Brier. If new Brier is higher (worse), revert to previous calibrator file and log WARNING. Prevents degraded calibration from deploying.

---

## Layer 1b — CalibrationDriftMonitor (3 fixes)

### Fix drift record labels — `main.py:1138`

Same label bug as calibrator. Change:
```python
# Before:
self._drift_monitor.record(position.calibrated_prob, outcome)

# After:
y_up = int(position.direction == outcome)
self._drift_monitor.record(position.calibrated_prob, y_up)
```
`calibrated_prob` is P(UP) → should be compared to `y_up` (did market go UP), not `outcome` (did trade win).

### Add `reset_baseline()` — `calibration_drift_monitor.py`

```python
def reset_baseline(self) -> None:
    """Call after calibrator refit so baseline reflects new calibration era."""
    try:
        self._redis.delete(_KEY_BASELINE, _KEY_ALERT_COUNT, _KEY_TOTAL_COUNT)
        self._history.clear()
        self._total_count = 0
    except redis.RedisError as exc:
        logger.warning(f"CalibrationDriftMonitor: reset failed — {exc}")
```
Call in `_check_resolutions` after each successful calibrator save.

### Bootstrap shrink when drifting — `fusion.py`

When `is_drifting()=True`, reduce `_BOOTSTRAP_SHRINK` from 0.8 → 0.4:
```python
# In NotTrainedError branch:
base_shrink = _BOOTSTRAP_SHRINK  # 0.8
if self._drift_monitor.is_drifting():
    base_shrink = min(base_shrink, 0.4)
if deepseek_regime == "high_uncertainty":
    base_shrink = _UNCERTAINTY_SHRINK
elif deepseek_regime == "ranging":
    base_shrink = _RANGING_SHRINK
combined = 0.5 + (kronos_cal - 0.5) * base_shrink
```
With `kronos_cal=0.2` and drift active: combined = 0.5 + (0.2–0.5) × 0.4 = 0.38. Near-neutral → Gate 5 naturally rejects the trade instead of firing a strong directional bet.

`SignalFusionEngine.__init__()` needs a `drift_monitor: CalibrationDriftMonitor` parameter. Wire from `main.py`.

---

## Layer 2 — Gate 8: Kalshi Consensus Gate

### Data validation

| Threshold | Blocked (bad days) | Blocked win rate | False positives (good days) |
|---|---|---|---|
| 5% | 40 (20.4%) | 20% | **0** |
| 8% | 36 (18.4%) | 22% | **0** |
| 15% | 23 (11.7%) | 13% | **0** |

Zero false positives across 248 good-day trades. The gate fires only when the regime has shifted.

**Recommended threshold: 8%** (Kalshi must be pricing ≥58% against our direction).

### Component A — Hard block — `pretrade_checklist.py`

New parameter `fresh_kalshi_mid: float` added to `checklist.run()`. Placed after Gate 7 shadow, before return:

```python
# Gate 8 — Kalshi consensus
opposing = (fresh_kalshi_mid - 0.5) if signal.direction == 0 else (0.5 - fresh_kalshi_mid)
oi_squeeze = signal.regime_features.get("oi_delta_pct", 0.0) > 0.001
effective_threshold = config.KALSHI_CONSENSUS_THRESHOLD / 4.0 if oi_squeeze and signal.direction == 0 else config.KALSHI_CONSENSUS_THRESHOLD
if opposing > effective_threshold:
    return fail(8, f"Kalshi consensus {fresh_kalshi_mid:.3f} opposes {'NO→DOWN' if signal.direction==0 else 'YES→UP'} (threshold {effective_threshold:.3f})")
```

OI squeeze compound: when `oi_delta_pct > 0.001` (shorts piling in) AND we're betting NO→DOWN, halve the threshold (4% instead of 8%). OI rising + Kalshi at 52%+ = short-squeeze setup. Win rate: 14.3% on 14 trades.

### Component B — Continuous Kelly multiplier — `pretrade_checklist.py`

Applied immediately after Kelly is computed, before contract rounding:
```python
# Gate 8b — Kalshi Kelly multiplier (gradient reduction)
opposing_margin = max(0.0, (fresh_kalshi_mid - 0.5) if signal.direction == 0 else (0.5 - fresh_kalshi_mid))
kalshi_kelly_mult = max(0.0, 1.0 - opposing_margin / 0.20)
kelly_dollars *= kalshi_kelly_mult
```
At 5¢ opposition: Kelly × 0.75. At 10¢: × 0.50. At 20¢: × 0 (equivalent to block via Kelly floor, but Gate 8a catches this first). This smooths the transition into the hard block.

### Supporting changes

- Add `KALSHI_CONSENSUS_THRESHOLD: float = 0.08` to `config.py`
- `main.py`: compute `fresh_kalshi_mid = (result2_best_bid_cents + result2_best_ask_cents) / 200.0` from second-fetch data; pass to both checklist calls
- Gate 8 blocks logged to `gate_rejections` with `failed_gate=8, shadow=0` — same path as Gates 1–6
- Add `("kalshi_mid_at_block", "REAL DEFAULT NULL")` to `_GATE_REJECTIONS_COLUMN_MIGRATIONS` for post-hoc threshold analysis
- Drift monitor → Kelly: add `is_drifting: bool = False` to `checklist.run()`; when True, apply additional 50% Kelly shrink stacked after Gate 8b multiplier; main.py passes `self._drift_monitor.is_drifting()`

---

## Layer 3 — Structural

### 3a. Per-direction rolling win rate tracker — new `btc_kalshi_system/signal/direction_win_rate_tracker.py`

Lightweight alternative to full `StratifiedEdgeTracker`. Two Redis sorted sets (scores = timestamps, members = outcomes):
- `trading:win_history_no` — NO trade outcomes (0/1)
- `trading:win_history_yes` — YES trade outcomes (0/1)

On each resolution: `ZADD` the outcome, `ZREMRANGEBYSCORE` to evict entries older than 30 trades (by rank, not time). Expose `get_win_rate(direction: int) -> float | None`.

Wire into `KellySizer.compute_size()` via new `direction_win_rate: float | None = None` parameter:
```python
if direction_win_rate is not None and direction_win_rate < 0.45:
    size *= 0.60  # 40% reduction when that direction's rolling 30-trade win rate < 45%
```
Fires within 15–20 bad trades vs drift monitor's 60-trade window. Direction-specific — a YES slump doesn't penalize NO trades.

Main.py fetches `self._dir_tracker.get_win_rate(signal.direction)` before checklist call, passes to Kelly.

### 3b. `btc_24h_return` as Feature 28

Feature store stores 1h candles permanently in `brti:candles:1h` (no TTL on hash). 6+ days of data available. Computation in `fusion._regime_features()`:

```python
if df1h is not None and len(df1h) >= 25:
    btc_24h_return = float(df1h["close"].iloc[-1] / df1h["close"].iloc[-25] - 1)
else:
    btc_24h_return = 0.0
```

**3-file contract update** (all three must match exactly):
- `regime_model.py` `_FEATURE_ORDER`: append `"btc_24h_return"` (feature 28)
- `train_regime.py` `_FEATURE_COLS`: append; add `_EXTRA_FILTERS_28 = _EXTRA_FILTERS_27 + " AND btc_24h_return IS NOT NULL"` and update `_build_query()` to accept `use_28` flag (note: `use_28=True` implies `use_27=True` — 28-feature model is a strict superset)
- `fusion.py` `_regime_features()`: add `btc_24h_return` to features dict

**Schema:** Add `("btc_24h_return", "REAL DEFAULT NULL")` to `_TRADES_COLUMN_MIGRATIONS` and include in `_record_trade_sqlite()`.

Existing `test_feature_order` test enforces 3-file consistency — it will catch any mismatch.

### 3c. `auto_retrain.py`: `_ROW_TRIGGER_DELTA` 500 → 200

After the initial 500-row regime model trains (~June 3), subsequent retrains trigger every 200 new qualifying rows ≈ ~4 days. Keeps the regime model within 4 days of current data vs the current 10-day lag.

---

## File Map

| File | Change |
|---|---|
| `config.py` | Add `CALIBRATOR_MODEL_PATH`, `KALSHI_CONSENSUS_THRESHOLD` |
| `btc_kalshi_system/models/calibrator.py` | Lower `_MIN_SAMPLES` 500→300; add monotonicity guard to `fit()` |
| `btc_kalshi_system/signal/fusion.py` | Accept `drift_monitor` param; use `_BOOTSTRAP_SHRINK=0.4` when drifting |
| `btc_kalshi_system/signal/calibration_drift_monitor.py` | Add `reset_baseline()`; drift record uses y_up not outcome |
| `btc_kalshi_system/signal/direction_win_rate_tracker.py` | **New** — per-direction rolling 30-trade win rate |
| `btc_kalshi_system/execution/pretrade_checklist.py` | Gate 8 hard block + Kelly multiplier; accept `fresh_kalshi_mid`, `is_drifting` |
| `btc_kalshi_system/execution/kelly.py` | Accept `direction_win_rate` param; apply 40% shrink when < 0.45 |
| `btc_kalshi_system/models/regime_model.py` | Add `btc_24h_return` to `_FEATURE_ORDER` |
| `main.py` | Wire all: calibrator load/save/refit cadence; drift label fix; Gate 8 wiring; direction tracker; feature 28 |
| `scripts/train_regime.py` | Add `btc_24h_return` to `_FEATURE_COLS`; `_EXTRA_FILTERS_28`; `use_28` flag |
| `scripts/train_calibrator.py` | **New** — standalone calibrator training script |
| `scripts/auto_retrain.py` | `_ROW_TRIGGER_DELTA` 500→200 |

**Tests to add/update:**
- `tests/models/test_calibrator.py`: label bug fix test (y_up labels), rolling window test, save/load round-trip with correct labels, monotonicity guard test
- `tests/signal/test_calibration_drift_monitor.py`: `reset_baseline()` test, y_up label recording test, bootstrap-shrink-when-drifting test
- `tests/execution/test_pretrade_checklist.py`: Gate 8 block tests (NO opposing, YES opposing, OI squeeze compound, Kelly multiplier gradient), `fresh_kalshi_mid` parameter tests
- `tests/execution/test_kelly.py`: `direction_win_rate < 0.45` shrink test
- `tests/signal/test_direction_win_rate_tracker.py`: **New** — add, evict, get_win_rate tests
- `tests/signal/test_feature_order.py`: update to 28 features, 3-tuple unpack
- `tests/signal/test_regime_features.py`: `btc_24h_return` computation test
- `tests/models/test_regime_model.py`: add `btc_24h_return` to synthetic features

---

## Vulnerabilities and Mitigations

| Vulnerability | Mitigation |
|---|---|
| Gate 8 fires when Kalshi market is thin / manipulable | Gate 1 (3¢ spread) already filters thin markets before Gate 8 is reached |
| Calibrator monotonicity guard reverts a correct calibration | Guard only reverts if Brier is strictly worse on the same training data |
| `btc_24h_return` not available on cold start (<25 1h candles) | Default 0.0, set `features_stale=True`; excluded from training |
| Rolling 30-trade win rate fires on short unlucky streak | 40% shrink (not block); 30-trade minimum reduces noise; direction-specific |
| Drift monitor reset fires too aggressively (every refit) | Only called on successful calibrator refit (every 25 resolutions = ~6h) |
| OI squeeze compound at 2% threshold fires too broadly | Secondary condition — only applies to NO→DOWN with OI actively rising; also hit Gate 8A multiplier first |
| Continuous Kelly multiplier + drift shrink + streak shrink = near-zero Kelly | All multiplicative shrinks stack; floor is Kelly rounds-to-0 → Gate 2 blocks cleanly |
| `btc_24h_return` feature creates overfitting at high negative values | Only used in regime model (not as a hard gate); XGBoost learns to weight it alongside other features |

---

## Deployment Order

1. **Layer 2 first** (Gate 8 + drift→Kelly wiring): immediate bleeding stop, no model changes, deploying with restart
2. **Layer 1 fixes** (calibrator bugs + persistence): second restart; calibrator begins training correctly within 6 hours
3. **Layer 1b** (drift monitor label + bootstrap shrink): same deploy as Layer 1
4. **Layer 3a** (direction win rate tracker): third restart; completes the early-warning system
5. **Layer 3b** (Feature 28): requires `train_regime.py` and `train_calibrator.py` updates; deploy before June 3 regime model train
6. **Layer 3c** (auto_retrain delta): deploy any time before June 3

All changes maintain backward compatibility with existing paper trades and SQLite schema (migrations are idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS` equivalents).
