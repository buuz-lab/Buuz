# K15 Ungated Paper Experiment Design — 2026-06-08

## Goal

Run a $100 paper trading experiment using raw k15 signal with minimal gating to validate that k15's proven edge (Brier 0.184 vs Kalshi 0.222) translates into P&L. Runs embedded in KronosV2 alongside the main system, completely independent of regime v2 and all gates except spread and candle progress.

---

## Architecture

No new async loop. A synchronous `_run_experiment(markets)` method is called at the end of `_run_cycle()`. It runs in the same thread pool as the rest of the cycle, shares `_cached_kronos`, `_orderbook_feed`, `_router`, and `_store`. It cannot interfere with the main trading loop.

---

## DB Table

```sql
CREATE TABLE IF NOT EXISTS experiment_trades (
    candle_ts          TEXT    PRIMARY KEY,
    entered_at         REAL    NOT NULL,
    direction          INTEGER NOT NULL,
    k15_raw            REAL    NOT NULL,
    entry_price_cents  INTEGER NOT NULL,
    spread_cents       INTEGER NOT NULL,
    candle_progress    REAL    NOT NULL,
    outcome            INTEGER DEFAULT NULL,
    pnl                REAL    DEFAULT NULL,
    budget_after       REAL    DEFAULT NULL,
    resolved_at        REAL    DEFAULT NULL
)
```

`candle_ts` is the primary key — at most one experiment trade per 15-min candle.

---

## Instance Variables (added to `__init__`)

```python
self._exp_open: dict | None = None  # open unresolved position
self._exp_budget: float             # restored from DB on startup, else 100.0
self._exp_active: bool              # False when budget exhausted
```

**Budget persistence:** After creating `experiment_trades`, read last known budget:
```python
_last = self._db.execute(
    "SELECT budget_after FROM experiment_trades "
    "WHERE budget_after IS NOT NULL ORDER BY entered_at DESC LIMIT 1"
).fetchone()
self._exp_budget = float(_last[0]) if _last else 100.0
self._exp_active = self._exp_budget > 0
```

---

## _run_experiment(markets) Logic

Called at the end of `_run_cycle`, after `_resolve_gate_rejections()`.

### Step 1 — Try to resolve open position

```python
if self._exp_open is not None:
    df15 = self._store.get_ohlcv("15min")
    if df15 is not None and len(df15) >= 2:
        closed_ts = (
            df15.index[-2].to_pydatetime()
            .astimezone(timezone.utc)
            .replace(second=0, microsecond=0)
            .isoformat()
        )
        if self._exp_open["candle_ts"] == closed_ts:
            closed = df15.iloc[-2]
            btc_direction = 1 if closed["close"] > closed["open"] else 0
            won = btc_direction == self._exp_open["direction"]
            entry_cents = self._exp_open["entry_price_cents"]
            pnl = (100 - entry_cents) / 100.0 if won else -(entry_cents / 100.0)
            self._exp_budget += pnl
            self._db.execute(
                "UPDATE experiment_trades "
                "SET outcome=?, pnl=?, budget_after=?, resolved_at=? "
                "WHERE candle_ts=?",
                (btc_direction, round(pnl, 4), round(self._exp_budget, 4),
                 time.time(), closed_ts)
            )
            self._db.commit()
            logger.info(
                f"K15 experiment: {'WIN' if won else 'LOSS'} "
                f"pnl={pnl:+.2f} budget={self._exp_budget:.2f}"
            )
            self._exp_open = None
            if self._exp_budget <= 0:
                self._exp_active = False
                logger.info("K15 experiment: budget exhausted — stopped")
                return
```

### Step 2 — Try to enter new position

Skip if: not active, already have open position, k15 unavailable.

```python
if not self._exp_active or self._exp_open is not None:
    return

cached = self._cached_kronos
k15 = cached.get("prob_15min") if cached else None
if k15 is None:
    return

# Signal filter — skip near 0.5
if abs(k15 - 0.5) < 0.05:
    return

# Find 15-min market
market_15 = next((m for m in markets if "15" in m.get("ticker", "")), None)
if market_15 is None:
    return

ticker = market_15["ticker"]

# Candle progress from market close_time
try:
    close_str = market_15.get("close_time", "")
    close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
    candle_progress = (time.time() - (close_dt.timestamp() - 900.0)) / 900.0
except Exception:
    return

if not (0.03 <= candle_progress <= 0.10):
    return

# Orderbook
try:
    ob = self._router.get_orderbook(ticker)
    if not ob:
        return
    bid, ask, _ = self._parse_orderbook(ob)
except Exception:
    return

if ask == 0:
    return

spread_cents = ask - bid
if spread_cents > 3:
    return

direction = 1 if k15 > 0.5 else 0
entry_price_cents = ask if direction == 1 else (100 - bid)

# Candle ts
candle_ts = (
    close_dt.replace(second=0, microsecond=0) - timedelta(seconds=900)
).isoformat()

self._exp_open = {
    "candle_ts": candle_ts,
    "direction": direction,
    "k15": k15,
    "entry_price_cents": entry_price_cents,
}

self._db.execute(
    """INSERT OR IGNORE INTO experiment_trades
       (candle_ts, entered_at, direction, k15_raw,
        entry_price_cents, spread_cents, candle_progress)
       VALUES (?, ?, ?, ?, ?, ?, ?)""",
    (candle_ts, time.time(), direction, round(k15, 4),
     entry_price_cents, spread_cents, round(candle_progress, 4))
)
self._db.commit()
side = "YES→UP" if direction == 1 else "NO→DOWN"
logger.info(
    f"K15 experiment entry: {side} k15={k15:.3f} "
    f"price={entry_price_cents}¢ progress={candle_progress:.1%} "
    f"budget={self._exp_budget:.2f}"
)
```

---

## Call Site

In `_run_cycle()`, after `_resolve_gate_rejections()`:

```python
# 9. K15 experiment (paper, independent of main signal pipeline)
try:
    self._run_experiment(markets)
except Exception as exc:
    logger.debug(f"K15 experiment error: {exc}")
```

---

## Files Changed

| File | Change |
|---|---|
| `main.py` | `experiment_trades` table creation, `_exp_*` instance vars + budget restore, `_run_experiment()` method, call site in `_run_cycle()` |
| `tests/test_k15_experiment.py` | New test file — 6 tests |

---

## Tests

| Test | What it checks |
|---|---|
| `test_experiment_enters_on_strong_k15` | k15=0.70, progress=0.05, spread=2¢ → entry logged to DB |
| `test_experiment_skips_weak_k15` | k15=0.52 → no entry |
| `test_experiment_skips_wide_spread` | spread=5¢ → no entry |
| `test_experiment_skips_wrong_progress` | progress=0.40 → no entry |
| `test_experiment_resolves_win` | Closed candle matches direction → pnl positive, budget updated |
| `test_experiment_resolves_loss_and_tracks_budget` | Loss → pnl negative, budget decremented |

---

## Safety

- Paper only — no real orders. `_run_experiment` never calls `_router.place_order`.
- Entirely wrapped in try/except in `_run_cycle` — any bug is silently caught and logged at DEBUG.
- `candle_ts` PRIMARY KEY prevents double-entry on restarts.
- Budget restored from DB on restart — no artificial $100 reset.
