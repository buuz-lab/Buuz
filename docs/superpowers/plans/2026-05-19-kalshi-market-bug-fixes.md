# Kalshi 15-min/Hourly Market Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 bugs in `main.py` so the system trades the correct Kalshi BTC 15-min and hourly markets instead of the wrong daily series, with correct blackout logic, strike extraction, and timeframe propagation.

**Architecture:** All changes are isolated to `main.py` (5 of 7 bugs), one comment added to `fusion.py`, and two new scripts. No model, data, or test files change. Discovery was already run — findings embedded below.

**Tech Stack:** Python 3.11+, loguru, sqlite3, Kalshi REST API

---

## Discovery Findings (already run — do not re-run)

From live Kalshi API (run 2026-05-19):
- **15-min up/down series ticker**: `KXBTC15M`
  - Example market: `KXBTC15M-26MAY191545-45`
  - Reference/strike field: `floor_strike` (e.g., `76734.11` = BRTI at open time)
  - Trading cutoff field: `close_time` (e.g., `"2026-05-19T19:45:00Z"`)
  - `yes_sub_title` shows `"Target Price: $76,734.11"` — confirms `floor_strike` is correct

- **Hourly above/below series ticker**: `KXBTCD`
  - Example market: `KXBTCD-26MAY1916-T85799.99`
  - Reference/strike field: `floor_strike` (e.g., `85799.99` = the strike level)
  - Trading cutoff field: `close_time` (e.g., `"2026-05-19T20:00:00Z"`)

- Neither series has a `result_at_open` field. Both use `floor_strike` as the relevant threshold.
- The `close_time` field (UTC ISO-8601 with trailing `Z`) tells us when trading stops.

---

## Files Changed

- **Modify**: `main.py` — Bugs 3, 4, 5, 6, 7 + remove stale timing constants
- **Modify**: `btc_kalshi_system/signal/fusion.py` — Improvement C (comment only)
- **Already exists**: `scripts/discover_kalshi_markets.py` — Improvement A (written during planning)
- **Create**: `scripts/bootstrap_progress.py` — Improvement B

---

## Task 1: Verify Bugs 1 & 2 (already fixed — confirm, no code change)

**Files:**
- Read: `main.py:484–506`
- Read: `btc_kalshi_system/execution/router.py:30–43`
- Read: `.env` (for PAPER_TRADING)

- [ ] **Step 1: Confirm Bug 1 fix in `main.py`**

Verify lines 488–496 match exactly:
```python
import sys as _sys
_sys.excepthook = lambda exc_type, exc_val, exc_tb: logger.opt(exception=(exc_type, exc_val, exc_tb)).critical(
    "Uncaught exception — process is exiting"
)

try:
    system = KronosV2()
except Exception:
    logger.exception("Fatal error during KronosV2 initialisation — see traceback above")
    raise
```

- [ ] **Step 2: Confirm Bug 1 fix in `router.py`**

Verify `router.py:30–43` wraps `KalshiRawClient(...)` in try/except that calls `logger.critical(...)`. It already does — no change needed.

- [ ] **Step 3: Confirm Bug 2 (.env)**

Run:
```bash
grep PAPER_TRADING .env
```
Expected output:
```
PAPER_TRADING=true
```
If not `true`, set it to `true` now and stop. Do NOT set it to `false`.

---

## Task 2: Fix Bug 7 — remove double `update_market_context` call

**Files:**
- Modify: `main.py:155–164`

- [ ] **Step 1: Write a test that calls `_run_cycle` and counts `update_market_context` invocations**

No test file to add (main.py methods are not unit-tested). Instead, confirm the bug visually: in `_run_cycle()` starting around line 155, `update_market_context` is called inside the `if` block AND unconditionally after it.

- [ ] **Step 2: Apply fix**

In `main.py`, find the block:
```python
        now = time.time()
        ctx = self._get_market_context()
        if now - self._last_deepseek_refresh >= DEEPSEEK_REFRESH_SECONDS:
            self._fusion.update_market_context(ctx)
            self._last_deepseek_refresh = now
            logger.debug("DeepSeek context refreshed")

        # 3. Update fusion engine with latest context (idempotent if just refreshed)
        self._fusion.update_market_context(ctx)
```

Replace with:
```python
        now = time.time()
        ctx = self._get_market_context()
        if now - self._last_deepseek_refresh >= DEEPSEEK_REFRESH_SECONDS:
            self._last_deepseek_refresh = now
            logger.debug("DeepSeek context refreshed")
        # Always push the latest context once per cycle
        self._fusion.update_market_context(ctx)
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest -x -q
```
Expected: 195 passed (ignore the 2 pre-existing failures in test_circuit_breaker.py).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "fix: fetch market context once and update fusion engine exactly once per cycle"
```

---

## Task 3: Fix Bug 3 — `_get_active_markets()` queries wrong series

**Files:**
- Modify: `main.py` — `_get_active_markets()` method (~line 318)

- [ ] **Step 1: Apply fix**

Find and replace the entire `_get_active_markets` method:

Old:
```python
    def _get_active_markets(self) -> list[dict]:
        try:
            resp = self._router._raw._request(
                "GET", "/trade-api/v2/markets?series_ticker=KXBTC&status=open"
            )
            return resp.get("markets", [])
        except Exception as exc:
            logger.warning(f"Failed to fetch active KXBTC markets: {exc}")
            return []
```

New:
```python
    def _get_active_markets(self) -> list[dict]:
        # KXBTC15M = 15-min BTC up/down; KXBTCD = hourly BTC above/below
        series = [("KXBTC15M", "15min"), ("KXBTCD", "1h")]
        markets: list[dict] = []
        for series_ticker, market_type in series:
            try:
                resp = self._router._raw._request(
                    "GET", f"/trade-api/v2/markets?series_ticker={series_ticker}&status=open"
                )
                for m in resp.get("markets", []):
                    m["market_type"] = market_type
                    markets.append(m)
            except Exception as exc:
                logger.warning(f"Failed to fetch {series_ticker} markets: {exc}")
        if not markets:
            logger.info("No active 15-min or hourly BTC markets found")
        return markets
```

- [ ] **Step 2: Update the "no markets" message in `_run_cycle()`**

Find (around line 172):
```python
        if not markets:
            logger.info("No active KXBTC markets found")
            return
```

Replace with:
```python
        if not markets:
            logger.info("No active BTC markets found")
            return
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest -x -q
```
Expected: 195 passed.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "fix: query KXBTC15M and KXBTCD series instead of wrong KXBTC daily series"
```

---

## Task 4: Fix Bug 4 — timeframe hardcoded to "same_day"

**Files:**
- Modify: `main.py` — `_process_market()` method (~line 205)

- [ ] **Step 1: Apply fix**

Find in `_process_market()`:
```python
        timeframe = "same_day"
```

Replace with:
```python
        timeframe = market.get("market_type", "15min")
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest -x -q
```
Expected: 195 passed.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "fix: derive timeframe from market_type field instead of hardcoding same_day"
```

---

## Task 5: Fix Bug 5 — `_extract_strike()` fallback for up/down markets

**Files:**
- Modify: `main.py` — `_extract_strike()` method (~line 328)

Note: `floor_strike` is already the first field checked and IS present in both `KXBTC15M` and `KXBTCD` markets. This task adds the composite-price fallback for robustness (in case Kalshi ever returns a market with no strike fields).

- [ ] **Step 1: Apply fix**

Find the entire `_extract_strike` method:
```python
    def _extract_strike(self, market: dict) -> float | None:
        # Try common Kalshi market fields for the strike price
        for field in ("floor_strike", "cap_strike", "strike_price", "result_at_open"):
            val = market.get(field)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        # Parse from ticker: KXBTC-25JUN-T95000 → 95000.0
        ticker = market.get("ticker", "")
        for part in ticker.split("-"):
            if part.startswith("T") and part[1:].isdigit():
                return float(part[1:])
        return None
```

Replace with:
```python
    def _extract_strike(self, market: dict) -> float | None:
        for field in ("floor_strike", "cap_strike", "strike_price", "result_at_open"):
            val = market.get(field)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        # Parse from ticker: KXBTC-25JUN-T95000 → 95000.0
        ticker = market.get("ticker", "")
        for part in ticker.split("-"):
            if part.startswith("T") and part[1:].isdigit():
                return float(part[1:])
        # For up/down markets the threshold is "higher than current price",
        # so live BRTI price is the right substitute when no strike field is present.
        price = self._get_composite_price()
        if price > 0.0:
            logger.debug(
                f"No strike field found for {ticker} — using composite price {price:.2f} as fallback"
            )
            return price
        return None
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest -x -q
```
Expected: 195 passed.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "fix: fall back to composite price as strike when no strike field present in market"
```

---

## Task 6: Fix Bug 6 — replace hardcoded daily blackout with per-market logic

**Files:**
- Modify: `main.py` — timing constants (~lines 44–46), `_is_in_blackout()` method (~line 293), `_process_market()` call site (~line 195)

- [ ] **Step 1: Remove stale timing constants**

Find at the top of `main.py`:
```python
RESOLUTION_BLACKOUT_MINUTES = 15

# KXBTC resolves at 6:30 PM EDT (UTC-4 during DST) = 22:30 UTC
RESOLUTION_TIMES_EDT = [(18, 30)]
_EDT_OFFSET_HOURS = 4  # EDT = UTC-4
```

Replace with:
```python
# Per-market blackout windows: stop new entries this many seconds before close_time
_BLACKOUT_SECONDS = {"15min": 3 * 60, "1h": 10 * 60}
```

- [ ] **Step 2: Replace `_is_in_blackout()` method**

Find the entire method:
```python
    def _is_in_blackout(self) -> bool:
        now_utc = datetime.now(timezone.utc)
        blackout_seconds = RESOLUTION_BLACKOUT_MINUTES * 60
        for hour_edt, minute_edt in RESOLUTION_TIMES_EDT:
            # Convert EDT → UTC
            hour_utc = (hour_edt + _EDT_OFFSET_HOURS) % 24
            resolution_today = now_utc.replace(
                hour=hour_utc, minute=minute_edt, second=0, microsecond=0
            )
            delta = (resolution_today - now_utc).total_seconds()
            if 0 <= delta <= blackout_seconds:
                return True
        return False
```

Replace with:
```python
    def _market_is_in_blackout(self, market: dict) -> bool:
        """Return True if we are too close to this market's close_time to enter a new position."""
        close_time_str = market.get("close_time")
        if not close_time_str:
            return False
        try:
            close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            seconds_until_close = (close_dt - datetime.now(timezone.utc)).total_seconds()
            if seconds_until_close < 0:
                return True  # market already closed
            market_type = market.get("market_type", "15min")
            threshold = _BLACKOUT_SECONDS.get(market_type, 3 * 60)
            return seconds_until_close <= threshold
        except (ValueError, TypeError) as exc:
            logger.debug(f"Could not parse close_time '{close_time_str}': {exc}")
            return False
```

- [ ] **Step 3: Update call site in `_process_market()`**

Find:
```python
        # a. Resolution blackout
        if self._is_in_blackout():
            logger.info(f"In resolution blackout — skipping {ticker}")
            return
```

Replace with:
```python
        # a. Resolution blackout
        if self._market_is_in_blackout(market):
            logger.info(f"Too close to close_time — skipping {ticker}")
            return
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest -x -q
```
Expected: 195 passed.

- [ ] **Step 5: Verify no "same_day" or "RESOLUTION_TIMES_EDT" remain in `main.py`**

```bash
grep -n "same_day\|RESOLUTION_TIMES_EDT\|_EDT_OFFSET_HOURS\|_is_in_blackout\|RESOLUTION_BLACKOUT" main.py
```
Expected: no output (zero matches).

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "fix: replace daily EDT blackout with per-market close_time blackout for 15min/1h markets"
```

---

## Task 7: Improvement C — add comment in `fusion.py` clarifying strike for up/down markets

**Files:**
- Modify: `btc_kalshi_system/signal/fusion.py:77`

- [ ] **Step 1: Apply comment**

Find in `get_signal()`:
```python
        kronos_raw = self._kronos.run_monte_carlo(self._store, threshold=strike)
```

Replace with:
```python
        # For up/down markets, strike = BTC price at market open, so this computes
        # P(predicted_close > open_price) = P(price goes up) — exactly what we want.
        kronos_raw = self._kronos.run_monte_carlo(self._store, threshold=strike)
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest -x -q
```
Expected: 195 passed.

- [ ] **Step 3: Commit**

```bash
git add btc_kalshi_system/signal/fusion.py
git commit -m "docs: clarify that strike = open price for up/down markets in SignalFusionEngine"
```

---

## Task 8: Improvement B — create `scripts/bootstrap_progress.py`

**Files:**
- Create: `scripts/bootstrap_progress.py`

- [ ] **Step 1: Create the file**

```python
"""Prints bootstrap progress toward paper-trading thresholds."""
import sqlite3
from pathlib import Path

db_path = Path("trades.db")
if not db_path.exists():
    print("trades.db not found — system has not run yet.")
    raise SystemExit(0)

conn = sqlite3.connect(str(db_path))
total    = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
resolved = conn.execute("SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL").fetchone()[0]
wins     = conn.execute("SELECT SUM(outcome) FROM trades WHERE outcome IS NOT NULL").fetchone()[0] or 0
open_pos = total - resolved

print(f"Total trades logged    : {total}")
print(f"Open (not yet resolved): {open_pos}")
print(f"Resolved               : {resolved}  (need 500 for calibrator)")
if resolved:
    print(f"  Wins                 : {wins}")
    print(f"  Losses               : {resolved - wins}")
    print(f"  Win rate             : {wins / resolved * 100:.1f}%")
else:
    print("  Win rate             : —")
print(f"Edge tracker window    : {min(resolved, 50)} / 50  (need 30 for gate 4)")
print()
calibrator_pct = min(resolved / 500 * 100, 100)
edge_pct       = min(resolved / 30 * 100, 100)
print(f"Calibrator threshold   : [{('#' * int(calibrator_pct / 5)).ljust(20)}] {calibrator_pct:.0f}%")
print(f"Edge gate threshold    : [{('#' * int(edge_pct / 5)).ljust(20)}] {edge_pct:.0f}%")
if resolved >= 500:
    print("\nREADY TO GO LIVE — set PAPER_TRADING=false in .env and restart.")
```

- [ ] **Step 2: Run it to verify it works**

```bash
python3 scripts/bootstrap_progress.py
```
Expected: prints progress table (shows 0 resolved trades if system hasn't run yet, or actual counts if it has).

- [ ] **Step 3: Commit**

```bash
git add scripts/bootstrap_progress.py scripts/discover_kalshi_markets.py
git commit -m "feat: add bootstrap_progress and discover_kalshi_markets scripts"
```

---

## Task 9: Definition of Done — final verification

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest -q
```
Expected: 195 passed, 2 failed (the pre-existing circuit breaker failures — these are not caused by our changes).

- [ ] **Step 2: Confirm no hardcoded stale values remain**

```bash
grep -n "same_day\|RESOLUTION_TIMES_EDT\|KXBTC&\|_is_in_blackout\|_EDT_OFFSET_HOURS" main.py
```
Expected: no output.

- [ ] **Step 3: Run the discovery script**

```bash
python3 scripts/discover_kalshi_markets.py
```
Expected: prints market objects for `KXBTC15M` and `KXBTCD` showing `floor_strike` and `close_time` fields.

- [ ] **Step 4: Run bootstrap progress**

```bash
python3 scripts/bootstrap_progress.py
```
Expected: prints progress table without errors.

- [ ] **Step 5: Smoke-test startup (optional — requires Redis + Kalshi keys)**

```bash
timeout 15 python3 main.py 2>&1 | head -20
```
Expected within 10 seconds: lines containing `KronosV2 starting up` and `Running in PAPER TRADING mode`.
