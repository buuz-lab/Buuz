# Kronos V2 — SLY Lessons Phase 2 Design

**Date:** 2026-05-23  
**Scope:** Feature #6 (large_print_direction as 21st regime feature) + Dynamic Kelly (#11)  
**Status:** Approved, pending implementation

---

## Context

The SLY bot analysis produced a 12-item roadmap of signals that move BTC/Kalshi markets.
Tiers 1 and 2 were implemented in the 2026-05-23 session (commit `bd80bc0`) as the
20-feature expansion. This spec covers the two remaining buildable items before training:

- **#6 Large print direction** — institutional flow feature, was blocked on CVD volume
  verification (now cleared: all four exchanges use per-trade size)
- **#11 Dynamic Kelly** — runtime bet-size reduction in chop, dead tape, and losing streaks

Items #2 (Kalshi intra-cycle YES momentum) and #12 (slippage gate) are explicitly out of
scope. Items #10 and #12 are documented as post-training work in `handoff.md`.

**Timing:** Zero 20-feature training rows exist today. Both changes must land before
significant rows accumulate — the 21st feature must be present in all training data.

---

## Section 1 — `large_print_direction` (21st Regime Feature)

### What it measures
Net directional score from trades larger than 2× the session average size. Captures
institutional / smart-money flow. Range: [-1, 1]. Returns 0.0 when no large prints
exist in the sample window (graceful sparsity handling).

### Formula
```python
def _large_print_direction(self, trades: list[dict]) -> float:
    if not trades:
        return 0.0
    avg_size = sum(t["amount"] for t in trades) / len(trades)
    threshold = 2 * avg_size
    large = [t for t in trades if t["amount"] > threshold]
    if not large:
        return 0.0
    buy_vol = sum(t["amount"] for t in large if t["side"] == "buy")
    sell_vol = sum(t["amount"] for t in large if t["side"] == "sell")
    total = buy_vol + sell_vol
    return (buy_vol - sell_vol) / total if total > 0.0 else 0.0
```

### Data source
`derivatives_feed.py` already calls `self._exchange.fetch_trades(_SYMBOL, limit=500)`
every 5 minutes. The same trades list feeds `_cvd_normalized()` — no new API calls needed.

### Files changed

| File | Change |
|------|--------|
| `btc_kalshi_system/data/derivatives_feed.py` | Add `_large_print_direction(trades)` method; extend `_fetch_trades_data()` to return 3-tuple including it; add to `_fetch_features()` return dict |
| `btc_kalshi_system/signal/fusion.py` | Read `large_print_direction` from `regime:features` Redis dict; append to returned features dict as position 21 |
| `btc_kalshi_system/models/regime_model.py` | Append `"large_print_direction"` to `_FEATURE_ORDER` |
| `scripts/train_regime.py` | Append `"large_print_direction"` to `_FEATURE_COLS`; do NOT add to `_FEATURE_COLS_LEGACY` |
| `main.py` | Add `large_print_direction REAL` column to `trades` table schema and INSERT |
| `tests/signal/test_feature_order.py` | Update assertion from 20 → 21 features |
| `tests/data/test_derivatives_feed.py` | Add 5 new tests (all-buys, all-sells, no-large-prints, mixed, empty) |

### Invariants
- The 3-file feature order contract (`regime_model.py` / `train_regime.py` / `fusion.py`)
  must be identical. `python3 -m pytest tests/ -k "feature_order"` enforces this.
- `_FEATURE_COLS_LEGACY` stays at 6 features.
- `large_print_direction` is written to `regime:features` (the existing Redis key) —
  no new Redis keys needed.

---

## Section 2 — Dynamic Kelly

### Current state
`KellySizer.compute_size()` takes `(prob, market_price, current_exposure, same_timeframe_open)`.
Applies a fixed `CORRELATION_DISCOUNT = 0.7` for same-timeframe trades. No runtime
adaptation to market conditions.

### New multipliers

| Condition | Threshold | Shrink | Rationale |
|-----------|-----------|--------|-----------|
| Chop | `abs(range_breakout_flag) < 0.15` | `× 0.70` | No directional breakout = low conviction |
| Dead tape | `tape_speed_tpm < 0.20` | `× 0.80` | < 20 TPM = thin, uncommitted flow |
| Loss streak | `loss_streak >= 3` | `× 0.60` | Consecutive losses signal adverse conditions |

Multipliers are **multiplicative** and applied after existing logic. Worst case (all
three): `0.7 × 0.8 × 0.6 = 0.336×` base size.

### New constants (add to `kelly.py`)
```python
KELLY_CHOP_THRESHOLD = 0.15
KELLY_CHOP_SHRINK    = 0.70
KELLY_TAPE_THRESHOLD = 0.20
KELLY_TAPE_SHRINK    = 0.80
KELLY_STREAK_THRESHOLD = 3
KELLY_STREAK_SHRINK    = 0.60
```

### Loss streak tracking
Redis key: `trading:loss_streak` (integer counter).  
- **On win:** `redis.delete("trading:loss_streak")`  
- **On loss:** `redis.incr("trading:loss_streak")`

Updated in `main.py` at the outcome resolution block (where `outcome` is set from
market result). `PreTradeChecklist` reads the streak via its own Redis client before
calling Kelly.

### Interface change
```python
def compute_size(
    self,
    prob: float,
    market_price: float,
    current_exposure: float,
    same_timeframe_open: bool,
    regime_features: dict | None = None,  # new, optional
    loss_streak: int = 0,                 # new, optional
) -> float:
```

All new params are optional — existing callers and tests require no changes.

### Files changed

| File | Change |
|------|--------|
| `btc_kalshi_system/execution/kelly.py` | Add 6 constants; extend `compute_size()` with two new optional params; apply three multipliers after existing logic |
| `btc_kalshi_system/execution/pretrade_checklist.py` | Add Redis client to `__init__`; read streak before Kelly call; pass `regime_features` and `loss_streak` to `compute_size()` |
| `main.py` | Update streak counter (`INCR`/`DELETE`) at outcome resolution block |
| `tests/execution/test_kelly.py` | Add 7 new tests (each shrink fires, each doesn't fire at boundary, all three stack) |

---

## Section 3 — Post-Training Roadmap (no code)

Documented in `handoff.md`. See "Post-Training Roadmap" section.

- **#10 Meta-learning:** Extend `edge_tracker.py` after ~200 post-training trades to
  bucket accuracy by regime type and apply dynamic per-regime feature multipliers.
- **#12 Slippage gate:** Accumulate bid-ask spread data across cycles, then set a
  defensible threshold for Gate 6 on 15min markets. Revisit after go-live.

---

## Verification Checklist

1. `python3 -m pytest` — must pass (245+ tests; more with new ones)
2. `python3 -m pytest tests/ -k "feature_order"` — must show 21 features
3. Restart system; verify new trades have `large_print_direction IS NOT NULL`
4. Verify Kelly shrinks appear in logs on low-breakout or dead-tape markets
