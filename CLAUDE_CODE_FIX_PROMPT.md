# Claude Code Fix Prompt — KronosV2

Paste everything between the triple-backtick fences below directly into a Claude Code session
opened from inside the `Kronos V2` project directory.

---

```
You are working in the KronosV2 codebase — a BTC prediction-market trading system that trades
on Kalshi. The full system is already implemented and all 197 tests pass. Your job is to fix a
set of specific, known bugs and adapt the system for the actual market type being traded.

Read this entire prompt before touching any file.

---

## CONTEXT: WHAT THIS SYSTEM DOES

The system:
1. Streams BTC price from 4 exchanges into Redis (BRTI feed).
2. Every 5 minutes it runs the Kronos time-series model + XGBoost regime + DeepSeek LLM
   to produce a directional signal (up/down) and a probability.
3. It sizes positions with fractional Kelly, passes them through 6 pre-trade gates, then
   places (or simulates, in paper mode) orders on Kalshi.
4. When a Kalshi market resolves, it records the outcome, updates the calibrator and edge
   tracker, and writes to SQLite (trades.db).

---

## CONTEXT: THE TARGET MARKETS

We are trading **Kalshi BTC 15-minute and hourly up/down markets** — NOT the daily
close-above-strike (KXBTC) markets. Key differences:

- **Market type**: "Will BTC be higher/lower in the next 15 min (or 1 hour) than it is now?"
  There is NO fixed strike price. The reference is the BTC price AT MARKET OPEN. Kalshi stores
  this as a field in the market object — commonly `open_price`, `result_at_open`, or similar.
- **Resolution frequency**: Every 15 minutes or every hour, NOT once per day.
  This means we can accumulate 500+ resolved paper trades in days, not months.
- **Series tickers**: NOT `KXBTC` (that is the daily strike market). The 15-min and hourly
  up/down markets use a DIFFERENT series ticker. You MUST discover the correct one using the
  live Kalshi API before changing any code.
- **Timeframe field**: Should be `"15min"` or `"1h"` depending on which market, NOT `"same_day"`.
- **Blackout logic**: The current code has a single hardcoded 6:30 PM EDT blackout window.
  That is irrelevant for markets that resolve every 15 min / 1 hour. Each market should use
  its OWN resolution time from the Kalshi API to decide if we are too close to expiry.

---

## STEP 0 — DISCOVERY FIRST (mandatory before changing any code)

Before touching any file, run this script to learn what Kalshi actually returns:

```python
# Run: python3 scripts/discover_kalshi_markets.py
import os
from dotenv import load_dotenv
load_dotenv()
from btc_kalshi_system.execution.raw_http_client import KalshiRawClient
import json

c = KalshiRawClient()

# 1. List ALL BTC-related series to find the right series ticker
print("=== ALL BTC SERIES ===")
r = c._request("GET", "/trade-api/v2/series?status=active")
for s in r.get("series", []):
    if "BTC" in s.get("ticker", "").upper() or "BTC" in s.get("title", "").upper():
        print(json.dumps(s, indent=2))

# 2. Probe the series we already know about
print("\n=== KXBTC MARKETS (first 3) ===")
r = c._request("GET", "/trade-api/v2/markets?series_ticker=KXBTC&status=open&limit=3")
for m in r.get("markets", []):
    print(json.dumps(m, indent=2))
```

Save this as `scripts/discover_kalshi_markets.py` and run it. Study the output carefully:
- Note the EXACT series ticker for 15-min and hourly BTC up/down markets.
- Note which fields carry the reference/open price (the "strike" equivalent).
- Note the field that tells you WHEN the market closes (resolution time) — look for `close_time`,
  `expiration_time`, `close_date`, or similar.
- Note how to tell from a market object whether it is 15-min vs 1-hour.

Write down your findings as a comment at the top of `main.py` before writing any code.

---

## BUGS TO FIX (in order of priority)

### BUG 1 — Crash on startup produces empty log files

**Root cause**: `KronosV2.__init__()` can crash before any `logger.info()` call is reached.
When Python raises an uncaught exception it goes to `sys.stderr` — NOT through loguru's file
handler — so the log file stays empty. This has already been partially fixed (exception handler
added to `main()`), but verify the fix is complete:

In `main()`:
- Confirm there is a `try/except` around `KronosV2()` that calls `logger.exception(...)`.
- Confirm `sys.excepthook` is patched to route uncaught exceptions through loguru.

In `btc_kalshi_system/execution/router.py` (`KalshiClientRouter.__init__`):
- Confirm `KalshiRawClient(...)` is wrapped in a try/except that calls `logger.critical(...)`
  with a human-readable message before re-raising, so the user knows exactly what failed.

### BUG 2 — PAPER_TRADING is False (must be True for bootstrap)

Verify `.env` has `PAPER_TRADING=true`. Do NOT change it to false.

### BUG 3 — _get_active_markets() queries the wrong series

Currently: `"/trade-api/v2/markets?series_ticker=KXBTC&status=open"`

Fix: Query the correct series tickers for 15-min AND hourly BTC up/down markets (discovered
in Step 0). Return markets from BOTH series combined. Add a `market_type` field to each
returned dict (`"15min"` or `"1h"`) so downstream code can read it.

### BUG 4 — timeframe is hardcoded to "same_day"

In `_process_market(market, composite_price)`:

```python
timeframe = "same_day"   # ← WRONG
```

Fix: Derive the timeframe from the `market_type` field you added in Bug 3 fix (`"15min"` or
`"1h"`). Pass it through to the signal, checklist, position record, and SQLite row.

### BUG 5 — _extract_strike() won't work for up/down markets

For up/down markets there is no "above $X" strike. The reference is the BTC price when the
market opened. Kalshi stores this somewhere in the market object (discover the exact field
name in Step 0).

Fix `_extract_strike()` to:
1. First try the fields it already tries (`floor_strike`, `cap_strike`, `strike_price`,
   `result_at_open`).
2. THEN try whatever field(s) you discovered carry the open/reference price for up/down markets.
3. If STILL nothing, use `self._get_composite_price()` as a fallback — for an up/down market
   the threshold is "higher than current price", so the live BRTI price is the right substitute.
4. Log a debug line when the fallback is used so we can see it.

### BUG 6 — _is_in_blackout() uses a hardcoded single daily time

The current implementation blocks ALL trading within 15 minutes of 6:30 PM EDT. This is wrong
for markets that resolve every 15 min / 1 hour.

Replace `_is_in_blackout()` and `_process_market()`'s call to it with **per-market** blackout
logic:

```python
def _market_is_in_blackout(self, market: dict) -> bool:
    """
    Return True if we are too close to this specific market's resolution time.
    - 15-min market: block entry within 3 minutes of close.
    - 1-hour market: block entry within 10 minutes of close.
    Falls back to False (no block) if close_time cannot be parsed.
    """
```

Use the `close_time` / `expiration_time` field you found in Step 0. Parse it as UTC.
The `market_type` field (`"15min"` or `"1h"`) determines which threshold to use.

Remove the old `RESOLUTION_TIMES_EDT`, `_EDT_OFFSET_HOURS`, and `_is_in_blackout()` entirely.

### BUG 7 — Double market context fetch every cycle

In `_run_cycle()` the market context is fetched from Redis TWICE:

```python
if now - self._last_deepseek_refresh >= DEEPSEEK_REFRESH_SECONDS:
    ctx = self._get_market_context()            # fetch 1
    self._fusion.update_market_context(ctx)
    ...
ctx = self._get_market_context()                # fetch 2 — always runs
self._fusion.update_market_context(ctx)         # push twice if refresh ran
```

This has already been partially fixed. Verify the final version fetches once, checks the
15-min throttle, then calls `update_market_context` exactly once per cycle.

---

## SECONDARY IMPROVEMENTS (do these AFTER the bugs are fixed)

### IMPROVEMENT A — Verify the market context discovery script becomes permanent

Save `scripts/discover_kalshi_markets.py` (from Step 0) permanently in the repo so it can
be re-run any time Kalshi changes their API shape.

### IMPROVEMENT B — Add a bootstrap progress script

Create `scripts/bootstrap_progress.py`:

```python
"""Prints bootstrap progress toward paper-trading thresholds."""
import sqlite3
from pathlib import Path

db_path = Path("trades.db")
if not db_path.exists():
    print("trades.db not found — system has not run yet.")
    exit(0)

conn = sqlite3.connect(str(db_path))
total    = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
resolved = conn.execute("SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL").fetchone()[0]
wins     = conn.execute("SELECT SUM(outcome) FROM trades WHERE outcome IS NOT NULL").fetchone()[0] or 0
open_pos = total - resolved

print(f"Total trades logged   : {total}")
print(f"Open (not yet resolved): {open_pos}")
print(f"Resolved              : {resolved}  (need 500 for calibrator)")
print(f"  Wins                : {wins}")
print(f"  Losses              : {resolved - wins}")
print(f"  Win rate            : {wins/resolved*100:.1f}%" if resolved else "  Win rate: —")
print(f"Edge tracker window   : {min(resolved, 50)} / 50  (need 30 for gate 4)")
print()
calibrator_pct = min(resolved / 500 * 100, 100)
edge_pct       = min(resolved / 30 * 100, 100)
print(f"Calibrator threshold  : [{('#' * int(calibrator_pct/5)).ljust(20)}] {calibrator_pct:.0f}%")
print(f"Edge gate threshold   : [{('#' * int(edge_pct/5)).ljust(20)}] {edge_pct:.0f}%")
if resolved >= 500:
    print("\n✓ READY TO GO LIVE — set PAPER_TRADING=false in .env and restart.")
```

### IMPROVEMENT C — Sanity-check the Kronos threshold for up/down markets

In `SignalFusionEngine.get_signal()`, the call is:
```python
kronos_raw = self._kronos.run_monte_carlo(self._store, threshold=strike)
```

For an up/down market, `strike` will now be the current BTC price. This is correct —
`P(predicted_close > current_price)` IS the probability of "up". No code change needed,
just add a comment making this explicit so future readers understand it.

---

## CONSTRAINTS

- Do NOT change any test files unless a test breaks due to a bug fix.
- Do NOT change `PAPER_TRADING` in `.env` to false.
- Do NOT hardcode any new market-specific timing values until you have confirmed them from
  the live Kalshi API (Step 0).
- After all changes, run `python3 -m pytest` and confirm all tests still pass.
- Run `scripts/bootstrap_progress.py` and `scripts/discover_kalshi_markets.py` to verify
  the system is wired up correctly before finishing.

---

## DEFINITION OF DONE

1. `python3 -m pytest` — all 197+ tests pass.
2. `python3 scripts/discover_kalshi_markets.py` — prints real 15-min and hourly BTC market
   data from Kalshi, showing the fields used for strike/reference and close_time.
3. `python3 main.py` — starts up, writes `KronosV2 starting up` and `Running in PAPER TRADING
   mode` to the log within 10 seconds, then begins its 5-minute signal cycle.
4. After the first signal cycle, `python3 scripts/bootstrap_progress.py` shows > 0 total trades.
5. No hardcoded `"same_day"` timeframe anywhere in `main.py`.
6. No hardcoded `RESOLUTION_TIMES_EDT` list anywhere in `main.py`.
```
