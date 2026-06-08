# K15 Ungated Paper Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed a $100 paper experiment in KronosV2 that trades raw k15 signal (no regime gate, no calibrator) to validate whether k15's proven Brier edge translates into live P&L.

**Architecture:** A synchronous `_run_experiment(markets)` method called at the end of `_run_cycle()`. Shares `_cached_kronos`, `_router`, `_store`, and the DB. Enters once per 15-min candle at 3-10% progress when k15 > 0.55 or < 0.45 and spread ≤ 3¢. Resolves on the next cycle after the candle closes. Budget persists across restarts by reading the last `budget_after` from DB.

**Tech Stack:** Python 3.11, SQLite, pytest, `unittest.mock`

---

## File Map

| File | Change |
|---|---|
| `main.py` | `_CREATE_EXPERIMENT_TRADES_TABLE` constant, DB creation in `__init__`, `_exp_*` instance vars + budget restore, `_run_experiment()` method, call site in `_run_cycle()` |
| `tests/test_k15_experiment.py` | New test file — 6 tests using the `patch.__init__` pattern from `test_main_candle_logger.py` |

---

## Task 1: Write failing tests

**Files:**
- Create: `tests/test_k15_experiment.py`

- [ ] **Step 1: Create the test file with helpers and 6 failing tests**

```python
"""Tests for KronosV2._run_experiment — $100 k15 paper experiment."""
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import main


_CREATE_EXPERIMENT_TRADES_TABLE = """
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
"""


def _make_markets(candle_progress: float = 0.05) -> list[dict]:
    """Build a markets list with one 15-min market at the given candle progress."""
    close_ts = time.time() + (1.0 - candle_progress) * 900.0
    close_dt = datetime.fromtimestamp(close_ts, tz=timezone.utc)
    return [{
        "ticker": "KXBTC15M-TEST",
        "close_time": close_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }]


def _make_df15(candle_open_dt: datetime, close_above_open: bool = True) -> pd.DataFrame:
    """3-row 15-min OHLCV. index[-2] is the just-closed candle (candle_open_dt)."""
    idx = pd.DatetimeIndex([
        candle_open_dt - timedelta(minutes=15),
        candle_open_dt,
        candle_open_dt + timedelta(minutes=15),
    ], tz="UTC")
    close_price = 95100.0 if close_above_open else 94900.0
    return pd.DataFrame(
        {"open": [95000.0] * 3, "high": [96000.0] * 3,
         "low": [94000.0] * 3, "close": [close_price] * 3,
         "volume": [1.0] * 3},
        index=idx,
    )


def _make_system(k15: float = 0.70, bid: int = 48, ask: int = 50,
                 budget: float = 100.0, exp_open: dict | None = None) -> main.KronosV2:
    """Minimal KronosV2 with experiment_trades table and required attrs."""
    with patch.object(main.KronosV2, "__init__", lambda self: None):
        system = main.KronosV2()

    db = sqlite3.connect(":memory:")
    db.execute(_CREATE_EXPERIMENT_TRADES_TABLE)
    db.commit()

    system._db = db
    system._exp_active = True
    system._exp_open = exp_open
    system._exp_budget = budget
    system._cached_kronos = {"prob_15min": k15, "prob": 0.6}
    system._store = MagicMock()
    system._store.get_ohlcv.return_value = None   # no resolved candle by default
    system._router = MagicMock()
    system._router.get_orderbook.return_value = {"yes": [], "no": []}
    system._parse_orderbook = MagicMock(return_value=(bid, ask, 100))
    return system


# ── Entry tests ───────────────────────────────────────────────────────────────

def test_enters_on_strong_bullish_k15():
    """k15=0.70 > 0.55, progress=5%, spread=2¢ → YES entry logged."""
    system = _make_system(k15=0.70)
    system._run_experiment(_make_markets(candle_progress=0.05))
    row = system._db.execute("SELECT direction, k15_raw FROM experiment_trades").fetchone()
    assert row is not None
    assert row[0] == 1   # YES→UP
    assert abs(row[1] - 0.70) < 0.001


def test_enters_on_strong_bearish_k15():
    """k15=0.28 < 0.45, progress=5% → NO entry logged."""
    system = _make_system(k15=0.28)
    system._run_experiment(_make_markets(candle_progress=0.05))
    row = system._db.execute("SELECT direction FROM experiment_trades").fetchone()
    assert row is not None
    assert row[0] == 0   # NO→DOWN


def test_skips_weak_k15():
    """k15=0.52 within ±0.05 of 0.5 → no entry."""
    system = _make_system(k15=0.52)
    system._run_experiment(_make_markets(candle_progress=0.05))
    count = system._db.execute("SELECT COUNT(*) FROM experiment_trades").fetchone()[0]
    assert count == 0


def test_skips_wide_spread():
    """Spread = ask - bid = 50 - 44 = 6¢ > 3¢ → no entry."""
    system = _make_system(k15=0.70, bid=44, ask=50)   # spread = 6¢
    system._run_experiment(_make_markets(candle_progress=0.05))
    count = system._db.execute("SELECT COUNT(*) FROM experiment_trades").fetchone()[0]
    assert count == 0


# ── Resolution tests ──────────────────────────────────────────────────────────

def test_resolves_win_and_updates_budget():
    """Open YES position on candle that closed UP → pnl positive, budget grows."""
    candle_open_dt = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    candle_ts = candle_open_dt.replace(second=0, microsecond=0).isoformat()

    exp_open = {
        "candle_ts": candle_ts,
        "direction": 1,
        "k15": 0.70,
        "entry_price_cents": 45,
    }
    system = _make_system(exp_open=exp_open, budget=100.0)

    # Insert the open position row so UPDATE can find it
    system._db.execute(
        "INSERT INTO experiment_trades "
        "(candle_ts, entered_at, direction, k15_raw, entry_price_cents, spread_cents, candle_progress)"
        " VALUES (?, ?, 1, 0.70, 45, 2, 0.05)",
        (candle_ts, time.time()),
    )
    system._db.commit()

    # Candle closed UP — df15.index[-2] == candle_open_dt
    system._store.get_ohlcv.return_value = _make_df15(candle_open_dt, close_above_open=True)

    system._run_experiment(_make_markets(candle_progress=0.40))  # new candle, no entry

    row = system._db.execute(
        "SELECT outcome, pnl, budget_after FROM experiment_trades WHERE candle_ts=?",
        (candle_ts,)
    ).fetchone()
    assert row[0] == 1           # outcome UP
    assert row[1] > 0            # pnl positive (won)
    assert row[2] > 100.0        # budget grew
    assert system._exp_budget > 100.0
    assert system._exp_open is None


def test_resolves_loss_decrements_budget():
    """Open YES position on candle that closed DOWN → pnl negative, budget shrinks."""
    candle_open_dt = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    candle_ts = candle_open_dt.replace(second=0, microsecond=0).isoformat()

    exp_open = {
        "candle_ts": candle_ts,
        "direction": 1,
        "k15": 0.70,
        "entry_price_cents": 45,
    }
    system = _make_system(exp_open=exp_open, budget=100.0)

    system._db.execute(
        "INSERT INTO experiment_trades "
        "(candle_ts, entered_at, direction, k15_raw, entry_price_cents, spread_cents, candle_progress)"
        " VALUES (?, ?, 1, 0.70, 45, 2, 0.05)",
        (candle_ts, time.time()),
    )
    system._db.commit()

    # Candle closed DOWN
    system._store.get_ohlcv.return_value = _make_df15(candle_open_dt, close_above_open=False)

    system._run_experiment(_make_markets(candle_progress=0.40))

    row = system._db.execute(
        "SELECT outcome, pnl, budget_after FROM experiment_trades WHERE candle_ts=?",
        (candle_ts,)
    ).fetchone()
    assert row[0] == 0           # outcome DOWN
    assert row[1] < 0            # pnl negative (lost)
    assert row[2] < 100.0        # budget shrank
    assert system._exp_budget < 100.0
    assert system._exp_open is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/test_k15_experiment.py -v 2>&1 | tail -20
```

Expected: all 6 FAIL — `_run_experiment` doesn't exist yet. Error should be `AttributeError: 'KronosV2' object has no attribute '_run_experiment'`.

---

## Task 2: Implement the experiment in main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add table constant**

After the `_CREATE_CANDLE_FEATURES_TABLE` constant (search for it), add:

```python
_CREATE_EXPERIMENT_TRADES_TABLE = """
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
"""
```

- [ ] **Step 2: Create the table in `__init__` and restore budget**

In `KronosV2.__init__`, after `self._db.execute(_CREATE_CANDLE_FEATURES_TABLE)` and its migration loop (around line 432-438), before `self._db.commit()`, add:

```python
        self._db.execute(_CREATE_EXPERIMENT_TRADES_TABLE)
```

Then after `self._db.commit()` (around line 438), before `self._running = False`, add:

```python
        # K15 paper experiment — $100 budget, persisted across restarts.
        _last_exp = self._db.execute(
            "SELECT budget_after FROM experiment_trades "
            "WHERE budget_after IS NOT NULL ORDER BY entered_at DESC LIMIT 1"
        ).fetchone()
        self._exp_budget: float = float(_last_exp[0]) if _last_exp else 100.0
        self._exp_open: dict | None = None
        self._exp_active: bool = self._exp_budget > 0
```

- [ ] **Step 3: Add `_run_experiment()` method**

Add this method to `KronosV2` after `_run_cycle` (search for `def _run_cycle`). Place it just before `def _process_market`:

```python
    def _run_experiment(self, markets: list) -> None:
        """$100 paper experiment: trade raw k15 with no regime gate."""
        # ── Step 1: Resolve open position ────────────────────────────────────
        if self._exp_open is not None:
            df15 = self._store.get_ohlcv("15min")
            if df15 is not None and len(df15) >= 2:
                closed_ts = (
                    df15.index[-2]
                    .to_pydatetime()
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
                        (btc_direction, round(pnl, 4),
                         round(self._exp_budget, 4), time.time(), closed_ts),
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

        # ── Step 2: Enter new position ────────────────────────────────────────
        if not self._exp_active or self._exp_open is not None:
            return

        cached = self._cached_kronos
        k15 = cached.get("prob_15min") if cached else None
        if k15 is None or abs(k15 - 0.5) < 0.05:
            return

        market_15 = next((m for m in markets if "15" in m.get("ticker", "")), None)
        if market_15 is None:
            return

        ticker = market_15["ticker"]
        try:
            close_str = market_15.get("close_time", "")
            close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            candle_progress = (time.time() - (close_dt.timestamp() - 900.0)) / 900.0
        except Exception:
            return

        if not (0.03 <= candle_progress <= 0.10):
            return

        try:
            ob = self._router.get_orderbook(ticker)
            if not ob:
                return
            bid, ask, _ = self._parse_orderbook(ob)
        except Exception:
            return

        if ask == 0 or (ask - bid) > 3:
            return

        direction = 1 if k15 > 0.5 else 0
        entry_price_cents = ask if direction == 1 else (100 - bid)
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
            "INSERT OR IGNORE INTO experiment_trades "
            "(candle_ts, entered_at, direction, k15_raw, "
            "entry_price_cents, spread_cents, candle_progress) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (candle_ts, time.time(), direction, round(k15, 4),
             entry_price_cents, ask - bid, round(candle_progress, 4)),
        )
        self._db.commit()
        side = "YES→UP" if direction == 1 else "NO→DOWN"
        logger.info(
            f"K15 experiment entry: {side} k15={k15:.3f} "
            f"price={entry_price_cents}¢ progress={candle_progress:.1%} "
            f"budget={self._exp_budget:.2f}"
        )
```

- [ ] **Step 4: Add call site in `_run_cycle()`**

In `_run_cycle()`, find the section after `_resolve_gate_rejections()` (search for `self._resolve_gate_rejections()`). Add immediately after its try/except block:

```python
        # 9. K15 paper experiment (independent of main signal pipeline)
        try:
            self._run_experiment(markets)
        except Exception as exc:
            logger.debug(f"K15 experiment error: {exc}")
```

- [ ] **Step 5: Run all 6 tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/test_k15_experiment.py -v 2>&1 | tail -20
```

Expected: all 6 PASS.

- [ ] **Step 6: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 599+ passing, 0 failures.

- [ ] **Step 7: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add main.py tests/test_k15_experiment.py && git commit -m "$(cat <<'EOF'
feat: k15 ungated $100 paper experiment embedded in KronosV2

Runs alongside main system in _run_cycle. Enters on k15 > 0.55 or
< 0.45 at 3-10% candle progress, spread ≤ 3¢. Fixed 1 contract.
Budget persists across restarts via experiment_trades table.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Restart service + verify

- [ ] **Step 1: Restart KronosV2**

```bash
launchctl kickstart -k gui/$(id -u)/com.kronos.v2
```

- [ ] **Step 2: Verify experiment_trades table exists and experiment is active**

```bash
sleep 15 && cd "/Users/ezrakornberg/Kronos V2" && python3 -c "
import sqlite3
conn = sqlite3.connect('trades.db')
tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
print('experiment_trades exists:', 'experiment_trades' in tables)
cols = [r[1] for r in conn.execute(\"PRAGMA table_info(experiment_trades)\").fetchall()]
print('columns:', cols)
conn.close()
"
```

Expected: `experiment_trades exists: True` with all 11 columns listed.

- [ ] **Step 3: Push**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git push origin main
```
