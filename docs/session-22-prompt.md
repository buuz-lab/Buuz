# Session 22 — Claude Code Prompt

## Context

You are working on **Kronos V2** — a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min up/down markets). The project is at `~/Kronos V2`. **Read `handoff.md` in full before writing a single line of code.** It has the full architecture, every gotcha, and the 3-file feature order contract. The test suite currently passes at 395 tests — do not break it.

---

## Task 1: 15-min candle feature logger

### Why

The regime model's training data only exists for candles where Kronos actually fired (trades + gate rejections) — roughly 50 rows/day. The planned regime label change (train on BTC close > open instead of Kronos prediction accuracy) needs features logged at **every** 15-min candle boundary, not just trade moments. This unlocks ~96 rows/day of clean labeled data. This session is infrastructure only — no retrain, no label change. Just collect the data.

### What to build

**1. New table `candle_features` in `trades.db`**

Add `_CREATE_CANDLE_FEATURES_TABLE` as a constant in `main.py`, and call `conn.execute(_CREATE_CANDLE_FEATURES_TABLE)` in `__init__` alongside the existing table creation calls (after `_CREATE_GATE_REJECTIONS_TABLE`). Also add an empty `_CANDLE_FEATURES_COLUMN_MIGRATIONS` list (same pattern as the others) and run it through the migration loop.

Schema:

```sql
CREATE TABLE IF NOT EXISTS candle_features (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candle_ts     TEXT NOT NULL UNIQUE,
    btc_direction INTEGER,
    logged_at     TEXT NOT NULL,
    features_stale  INTEGER DEFAULT 0,
    deribit_stale   INTEGER DEFAULT 0,
    funding_rate                REAL,
    funding_rate_trend          REAL,
    oi_delta_pct                REAL,
    cvd_normalized              REAL,
    basis_spread_pct            REAL,
    brti_volatility_1h          REAL,
    cvd_velocity                REAL,
    cvd_acceleration            REAL,
    brti_momentum_5min          REAL,
    brti_momentum_15min         REAL,
    candle_progress             REAL,
    hour_sin                    REAL,
    hour_cos                    REAL,
    kalshi_implied_prob         REAL,
    funding_window_proximity    REAL,
    trend_slope_1h              REAL,
    trend_r2_1h                 REAL,
    hourly_sr_proximity         REAL,
    range_breakout_flag         REAL,
    tape_speed_tpm              REAL,
    large_print_direction       REAL,
    atm_iv                      REAL,
    iv_rv_spread                REAL,
    pcr_oi                      REAL,
    term_structure_slope        REAL,
    skew_25d                    REAL,
    kalshi_spread_normalized    REAL
)
```

Feature columns are in `_FEATURE_ORDER` order from `btc_kalshi_system/models/regime_model.py` — import and use that constant, do not hardcode the list.

**2. `get_features_snapshot()` on `SignalFusionEngine` in `fusion.py`**

Add a public method that returns the current regime features without running MC, calibration, or Kalshi mid update:

```python
def get_features_snapshot(self) -> tuple[dict, bool, bool]:
    """
    Returns (features_dict, features_stale, deribit_stale).
    Lightweight — reads Redis + OHLCV only; no MC, no calibration, no market context mutation.
    Safe to call from a background loop.
    """
    features, features_stale, deribit_stale, _okx_stale = self._regime_features()
    return features, features_stale, deribit_stale
```

This wraps the existing `_regime_features()` which already has the full 4-tuple return signature. Do not duplicate that logic.

**3. `_candle_logger_loop()` coroutine in `main.py`**

```python
async def _candle_logger_loop(self) -> None:
    """Logs regime features + BTC direction at every 15-min candle close."""
    last_logged_ts = None
    while self._running:
        try:
            await asyncio.sleep(30)
            df15 = self._store.get_ohlcv("15min")
            if df15 is None or len(df15) < 3:
                continue
            # The second-to-last candle is the most recently *closed* 15-min candle.
            # The last candle is the in-progress one — skip it.
            closed_ts = df15.index[-2]
            if closed_ts == last_logged_ts:
                continue  # already logged this candle
            last_logged_ts = closed_ts
            closed_candle = df15.iloc[-2]
            btc_direction = 1 if closed_candle["close"] > closed_candle["open"] else 0
            features, features_stale, deribit_stale = self._fusion.get_features_snapshot()
            cols = list(_FEATURE_ORDER)  # import from regime_model.py
            vals = [features.get(c) for c in cols]
            placeholders = ", ".join(["?"] * (4 + len(cols)))
            col_names = "candle_ts, btc_direction, logged_at, features_stale, deribit_stale, " + ", ".join(cols)
            self._conn.execute(
                f"INSERT OR IGNORE INTO candle_features ({col_names}) VALUES ({placeholders})",
                [
                    closed_ts.isoformat(),
                    btc_direction,
                    datetime.utcnow().isoformat(),
                    int(features_stale),
                    int(deribit_stale),
                    *vals,
                ],
            )
            self._conn.commit()
            logger.info(f"CandleLogger: logged candle {closed_ts} direction={btc_direction}")
        except Exception as exc:
            logger.warning(f"CandleLogger: {exc}")
            # Never exit the loop
```

- Import `_FEATURE_ORDER` from `btc_kalshi_system.models.regime_model` at the top of `main.py`
- Use `INSERT OR IGNORE` — idempotent on `candle_ts UNIQUE`
- Catch ALL exceptions per-iteration so the loop never exits
- The 30s poll interval means the logger fires within 30s of each 15-min close — acceptable
- `df15.index[-2]` (second-to-last) is the **closed** candle; `df15.index[-1]` is in-progress

**4. Add to `asyncio.gather()` in `run()`**

Add `self._candle_logger_loop()` to the existing `asyncio.gather(...)` call alongside `_regime_watchdog`, `_kronos_background_loop`, etc.

**Tests: `tests/test_main_candle_logger.py`**

Write at least 5 tests using the existing fakeredis + in-memory SQLite patterns from `tests/test_main_bg_kronos.py`:

1. `test_candle_logger_writes_row_on_new_candle` — two closed 15-min candles in OHLCV, loop fires, row written with correct `btc_direction`
2. `test_candle_logger_no_duplicate_on_same_candle` — same candle seen twice, `INSERT OR IGNORE` means only one row
3. `test_candle_logger_survives_exception` — `get_features_snapshot` raises, loop continues (doesn't exit)
4. `test_candle_logger_btc_direction_close_above_open` — close > open → direction=1
5. `test_candle_logger_btc_direction_close_below_open` — close ≤ open → direction=0
6. `test_candle_features_table_created_on_init` — `candle_features` table exists after `KronosTrader.__init__()`

---

## Task 2: Regime confidence tracker script

### Why

The regime model just went live (session 21). As Gate 2 shadow data accumulates over the coming days, there is no existing script to see whether high-confidence regime calls are actually more accurate than low-confidence ones. This script gives that visibility and also reports `candle_features` table health.

### What to build: `scripts/regime_confidence_tracker.py`

Model on the existing `scripts/regime_health_check.py` (same arg parsing, same section pattern, no external deps beyond sqlite3 + standard library).

**`--days N` flag** (default 30) limits all queries to `DATE(timestamp) >= date('now', '-N days')`.

**Section 1 — Overall regime model live stats**

```
=== REGIME MODEL LIVE STATS ===
Trades with regime_prob (last 30d) : N
  Agreement with Kronos direction  : X%  (regime direction == trade direction)
  Win rate on agreements           : X%  (n=Y)
  Win rate on disagreements        : X%  (n=Z)
  (Note: Gate 2 is shadow mode — disagreements do NOT block trades)
```

"Agreement" = `(regime_prob >= 0.5 AND direction = 1) OR (regime_prob < 0.5 AND direction = 0)`.

**Section 2 — Confidence-stratified accuracy**

Bin trades by `ABS(regime_prob - 0.5)` (distance from 0.5 = confidence):

```
=== CONFIDENCE-STRATIFIED ACCURACY ===
Confidence bucket   n    win_pct   agrees_with_kronos
low   (<0.10)       Y    X%        X%
med   (0.10–0.20)   Y    X%        X%
high  (0.20–0.30)   Y    X%        X%
very  (>0.30)       Y    X%        X%
(Only rows with regime_prob IS NOT NULL AND outcome IS NOT NULL)
```

**Section 3 — Gate 2 shadow disagreements detail**

```
=== GATE 2 SHADOW DISAGREEMENTS ===
Total disagreements (last 30d) : N
  High-confidence (>0.70 away) : N  — win_pct X%
  Med-confidence  (0.60–0.70)  : N  — win_pct X%
  Low-confidence  (<0.60)      : N  — win_pct X%
```

"High-confidence disagreement" = `regime_prob > 0.7` when direction=0 (or `regime_prob < 0.3` when direction=1).

**Section 4 — Candle features table health**

```
=== CANDLE FEATURES LOGGER HEALTH ===
Total rows logged               : N
Date range                      : YYYY-MM-DD to YYYY-MM-DD
BTC up candles                  : N (X%)
BTC down candles                : N (X%)
Features stale                  : N (X%)
Deribit stale                   : N (X%)
Rows last 24h                   : N  (expected ~96)
```

If the `candle_features` table doesn't exist yet, print `(candle_features table not yet created — logger not running)` and skip.

---

## After both tasks

1. Run `python3 -m pytest tests/ -v` — all 395 existing tests must pass, plus the new candle logger tests.
2. Update `handoff.md` Current Progress section with a session 22 entry (table of files changed, what changed, why) — follow the exact pattern of the session 21 entry.

## Key constraints (from handoff.md — do not violate)

- **Feature order is a 3-file contract** (`regime_model.py`, `train_regime.py`, `fusion._regime_features()`). Do NOT add new features or change `_FEATURE_ORDER`. This session is data collection only.
- **`_regime_features()` returns a 4-tuple** `(features, stale, deribit_stale, okx_stale)`. `get_features_snapshot()` wraps it — do not bypass it.
- **Do NOT call `run_monte_carlo()` in `_candle_logger_loop`** — features only.
- **`INSERT OR IGNORE` is required** — the `candle_ts UNIQUE` constraint makes this idempotent.
- **Gate 2 is shadow mode** (`REGIME_GATE2_ENFORCING=False`) — do not change.
- **Calibrator is passthrough** — do not touch `models/calibrator.pkl`.
- **CVD test mocks must use `time.time()`**, not hardcoded epochs.
- The `candle_features` table is data collection only this session — it does NOT feed into any existing training pipeline yet.
