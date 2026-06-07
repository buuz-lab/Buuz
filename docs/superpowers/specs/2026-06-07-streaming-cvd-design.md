# Streaming CVD + 15s Batch Refresh — Design Spec

**Date:** 2026-06-07  
**Scope:** Feature freshness within the 0–10% candle-progress trading window  
**Files touched:** `btc_kalshi_system/data/derivatives_feed.py`, `tests/data/test_streaming_cvd.py`, `tests/data/test_derivatives_feed.py`

---

## Problem

CVD is a flow signal — staleness compounds. At 60s refresh, CVD at decision time could reflect trade flow from a full minute ago. In a fast market, that's enough time for a sharp reversal to have occurred and resolved. The goal is CVD that's current to the last trade tick, not the last HTTP poll.

The remaining derivatives features (liquidations, OI delta, OKX spot imbalance) benefit from fresher data but don't require tick-level granularity — 15s batch is sufficient. Slow-moving features (funding rate, ETH direction, macro correlation) see no meaningful benefit from sub-60s refresh.

---

## Architecture

Two concurrent async tasks replace the single 60s batch loop:

**Task 1 — `StreamingCVDAccumulator.run()`**  
Persistent WebSocket connection to OKX (`wss://ws.okx.com:8443/ws/v5/public`, BTC-USDT-SWAP trades channel). Accumulates `(timestamp_ms, side, size)` ticks in a 15-min rolling `deque`. Recomputes `cvd_normalized`, `large_print_direction`, and `last_price` on every tick. Falls back to Kraken WS after 3 consecutive OKX reconnect failures. Reconnects with exponential backoff capped at 30s.

**Task 2 — `DerivativesFeed` batch loop at 15s**  
Fast-tier features fetched every cycle. Slow-tier features cached and re-fetched at most once per 60s. CVD read from accumulator properties — no HTTP trade fetch. Writes one `regime:features` Redis payload per cycle, identical format to today.

---

## StreamingCVDAccumulator

### State

| Field | Type | Purpose |
|---|---|---|
| `_trades` | `deque` | `(timestamp_ms, side, size)` entries; pruned to 15-min window on each tick |
| `_cvd` | `float` | Current `(buy_vol - sell_vol) / total_vol` over the window |
| `_large_print` | `float` | Direction of trades above large-print threshold |
| `_last_price` | `float` | Price of most recent tick; used for `basis_spread_pct` |
| `_last_tick_at` | `float` | `time.time()` of last received tick |

No lock needed — asyncio is single-threaded; batch and WS coroutines never interleave mid-computation.

### Properties

- `cvd_normalized → float` — current value from `_cvd`
- `large_print_direction → float` — current value from `_large_print`
- `last_price → float` — current value from `_last_price`
- `is_stale → bool` — `True` if `time.time() - _last_tick_at > 120` or deque has fewer than 5 entries (cold-start guard)

### `run()` coroutine

```
connect to OKX WS
subscribe: {"op": "subscribe", "args": [{"channel": "trades", "instId": "BTC-USDT-SWAP"}]}
loop:
    recv message
    parse ticks → append to deque → prune entries > 15 min old → recompute CVD fields
    on disconnect:
        exponential backoff: 2s, 4s, 8s, 16s, 30s (capped)
        after 3 consecutive failures → switch to Kraken WS
        if Kraken also fails → log error, sleep 30s, retry OKX
```

### CVD computation

Identical logic to the existing `_cvd_normalized()` method — ported to operate on the deque rather than a fetched trade list. `large_print_direction` reuses existing `_large_print_direction()` logic. No behaviour change, just a different data source.

---

## Batch Loop Changes

### Constants

```python
_REFRESH_INTERVAL = 15    # was 60 — 4x more frequent
_FEATURES_TTL     = 120   # was 360 — 8x the new interval; survives several failed cycles
# sleep offset unchanged: _REFRESH_INTERVAL - 10
```

### Slow-feature cache

`DerivativesFeed` gets `_last_slow_fetch: float = 0.0` and two cache variables (`_cached_funding: dict`, `_cached_eth_dir: float`). At the top of `_fetch_features()`:

```python
_refetch_slow = (time.time() - self._last_slow_fetch) >= 60
```

**Fast tier (every 15s):** `liq_net_norm`, `oi_delta_pct`, `okx_spot_imbalance`, CVD fields (from accumulator — zero cost).

**Slow tier (at most once per 60s):** `funding_rate`, `funding_rate_trend`, `eth_direction_15min`. Macro correlation (`MacroFeed`) already has its own 15-min internal cache — no change needed.

First cycle always runs the slow tier (since `_last_slow_fetch = 0.0`).

### Removed methods

`_fetch_trades_data()`, `_kraken_trades_data()`, and `_get_kraken_exchange()` are removed. The Kraken ccxt exchange instance is no longer needed in the batch path (Kraken is now a WS fallback inside the accumulator).

### DerivativesFeed.run() change

Starts the accumulator as a concurrent task:
```python
await asyncio.gather(self._batch_loop(), self._cvd_accumulator.run())
```

---

## Failure Modes

| Scenario | Behaviour |
|---|---|
| WS down, batch running | `is_stale` → True after 120s silence; batch sets `_cvd_stale=True` in Redis payload; fusion marks row stale; no trade fires; auto-recovers on WS reconnect |
| Batch down, WS running | Redis key expires after 120s; fusion LKG path activates; CVD continues accumulating in deque; valid value written immediately when batch recovers |
| Both down | Redis key expires → LKG → LKG age check triggers stale; no trades; auto-recovery |
| WS stale, batch healthy | `_cvd_stale=True` written every cycle; rest of features still fresh and logged; correct — stale CVD is stale regardless |

Cold-start guard (deque < 5 entries) is ported from existing batch stale logic into `is_stale` — behaviour unchanged.

---

## Testing

### `tests/data/test_streaming_cvd.py` (new)

- Inject buy/sell tick sequence → assert `cvd_normalized` matches manual calculation
- Inject ticks older than 15 min → assert pruned before CVD computation
- No ticks for 121s → assert `is_stale = True`
- Ticks after a gap → assert `is_stale` clears and CVD recomputes correctly
- Deque < 5 entries → assert `is_stale = True` (cold-start guard)
- Mixed large/small trades → assert `large_print_direction` correct

### `tests/data/test_derivatives_feed.py` (existing, minimal changes)

- Update TTL assertion: `110 <= ttl <= 120` (was `350 <= ttl <= 360`)
- Add: batch reads CVD from accumulator, not HTTP (mock accumulator, assert `_fetch_trades_data` no longer called)
- Add: slow-tier features are not re-fetched when called within 60s of last slow fetch

### `tests/data/test_derivatives_feed_okx_stale.py`

No changes — mocks at method level, unaffected by the restructuring.

---

## What Does Not Change

- `_write_features()` — same Redis key format
- `regime:features` key — same structure fusion reads
- `_cvd_stale` flag propagation path
- LKG path and age check
- All fusion code
- DeribitOptionsFeed — stays at 300s (PCR/skew deltas are per-cycle, changing interval would shift their scale)
- Gate 12 floor/cap logic
- All other gates

---

## Foundation for B (Mid-Candle Trading)

The streaming accumulator provides the infrastructure needed for eventual mid-candle signals. Once ticks are accumulating continuously, a mid-candle CVD reversal detector is a natural extension:

- At 40% candle progress, snapshot CVD direction vs candle-open CVD direction
- If sharply reversed and price hasn't reacted → potential mid-candle entry signal
- Requires a separate model trained on mid-candle entries (the `would_exit` data currently being collected is capturing Kalshi repricing at this window, which is the first piece of training data needed)

No code for B is built now — the accumulator architecture doesn't need to change when B is eventually built.
