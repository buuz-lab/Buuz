# Foundation Improvements — 2026-06-08

Recommendations from session 41 analysis. Ordered by priority: fix-before-deploy first, then data quality, then future architecture.

---

## Priority 1 — Fix Before Gate 15 Goes Live

### 1A. Cross-candle Redis contamination in Gate 15

**The bug:** `mid_candle:prob` Redis key has a 600s expiry. A 15-min candle is 900s. If the snapshot fires at 41% of candle N and `_run_cycle` processes candle N+1 at 2% progress (within 18s of the new candle opening), Gate 15 would read the previous candle's score. The trade is on a different market with different strike and different Kalshi pricing, but the model output is from the prior candle. This is a silent, wrong trade.

**The fix:** The Redis payload already includes `candle_ts` (`json.dumps({"prob": ..., "candle_ts": _in_progress_key})`). Gate 15 must verify it before using the score:

```python
# Gate 15 — Mid-candle model (40-60% window only)
if 0.40 <= candle_progress <= 0.60:
    _mc_raw = self._redis_client.get("mid_candle:prob")
    if _mc_raw:
        _mc = json.loads(_mc_raw)
        # Reject if score is from a different candle
        if _mc.get("candle_ts") != _current_candle_ts_iso:
            pass  # no gate — stale score, skip silently
        else:
            _mc_prob = _mc.get("prob")
            if _mc_prob is not None:
                if signal.direction == 1 and _mc_prob < 0.38:
                    return fail(15, f"Mid-candle model bearish ({_mc_prob:.2f}) vs YES entry")
                if signal.direction == 0 and _mc_prob > 0.62:
                    return fail(15, f"Mid-candle model bullish ({_mc_prob:.2f}) vs NO entry")
```

`_current_candle_ts_iso` is the ISO string of the current in-progress candle open time, derivable from `signal.regime_features["candle_progress"]` and the current time, or passed through `_candle_ticker_map`.

**Why it matters:** Without this, Gate 15 can fire on wrong data with 100% confidence. Silent wrong trades are worse than no gate.

---

### 1B. Cold accumulator filter in training query

**The bug:** The first snapshot after a service restart has very few ticks (we saw 52 on restart). `cvd_since_open` is computed from those ticks — 52 trades is not enough for a stable CVD reading. A few large sells skew it dramatically. The model trains on this as if it's a valid observation.

**The fix:** Add `tick_count_since_open > 200` to `_CANDLE_QUERY` in `scripts/train_mid_candle.py`:

```python
_CANDLE_QUERY = """
SELECT {cols}, btc_direction, candle_ts
FROM candle_features
WHERE btc_direction IS NOT NULL
  AND cvd_since_open IS NOT NULL
  AND kalshi_mid_candle_mid IS NOT NULL
  AND tick_count_since_open > 200
ORDER BY candle_ts ASC
"""
```

200 ticks at OKX BTC-USDT-SWAP perp = roughly 10–15 seconds of trading. Sufficient for a stable CVD reading. The cold-accumulator rows are noise, not signal.

**Cost:** Loses a few rows per restart. At our restart frequency, maybe 1-2 rows/week. Acceptable.

---

### 1C. Log k5_candle_ts in mid-candle snapshot

**The bug:** `k5_at_midcandle` is `_cached_kronos["prob"]` — the most recently cached k5. KronosBG runs every 5 minutes when a new 5-min candle closes. At 40% progress (6 minutes in), k5 may have refreshed once (at ~33%) or not at all if the BG loop was slow. If it hasn't refreshed, `k5_at_midcandle` reflects a computation from a prior 15-min candle's sub-candles — it could be up to 10 minutes stale.

You can't detect this without knowing *when* k5 was computed relative to the current candle's open.

**The fix:** Add `k5_candle_ts` to the mid-candle snapshot dict and schema:

```python
# In _candle_logger_loop mid-candle snapshot block:
_ck = self._cached_kronos
_k5_candle_ts = str(_ck["candle_ts"]) if _ck else None
...
self._mid_candle_snaps[_in_progress_key] = {
    ...
    "k5_at_midcandle":  _k5_mid,
    "k5_candle_ts":     _k5_candle_ts,  # NEW
    ...
}
```

Add `("k5_candle_ts", "TEXT DEFAULT NULL")` to `_CANDLE_FEATURES_COLUMN_MIGRATIONS`.

**Why:** At training time you can filter `WHERE k5_candle_ts >= candle_ts` to ensure k5 was computed from data within the current candle. More importantly, in post-analysis you can identify how much of the k5 importance is from fresh vs stale cache rows.

---

## Priority 2 — Data Quality Improvements

### 2A. Kalshi feature independence — wait for first train, then decide

**The concern:** `kalshi_drift_cents`, `kalshi_velocity`, `kalshi_brti_alignment`, `kalshi_mid_candle_spread`, and `kalshi_mid_candle_progress` are all Kalshi-derived. Unlike the regime model (where Kalshi contamination breaks Gate 5/8), the mid-candle model's Gate 15 is a pure direction filter — it doesn't compare `mid_candle_prob` against Kalshi price. So the circularity is less severe.

But there's a subtler problem: if `kalshi_drift_cents` dominates the model's importances, the model is just learning "follow Kalshi's mid-candle repricing." That replicates what the Kalshi price already tells you, not independent alpha. The independent alpha features are: `cvd_since_open`, `cvd_rate`, `k5_k15_delta_at_midcandle`, `cvd_brti_divergence`, `brti_velocity`, `tick_count_since_open`, `oi_delta_at_midcandle`.

**The fix:** Don't pre-optimize. After the first real train (200 rows), check the importances:
- If `kalshi_drift_cents` or `kalshi_velocity` > 15% importance: remove them — the model is following Kalshi, not adding signal
- If `cvd_brti_divergence` or `k5_k15_delta` lead: the model is genuinely finding independent alpha, keep all features

**If removing Kalshi features:** Update `_MID_CANDLE_FEATURES` in `mid_candle_model.py` to keep only:
```python
_MID_CANDLE_FEATURES = [
    "cvd_since_open", "cvd_rate", "tick_count_since_open",
    "brti_drift_since_open", "brti_velocity",
    "cvd_brti_divergence",
    "k5_at_midcandle", "k15_at_midcandle", "k5_k15_delta_at_midcandle",
    "oi_delta_at_midcandle",
    "kalshi_mid_candle_progress",  # timing context only, not price
]
```

---

### 2B. Verify `brti_at_candle_open` capture timing

**The concern:** BRTI at candle open is captured on the FIRST loop iteration that sees a new `_in_progress_key` (i.e., the first time the loop fires after a new 15-min candle starts). The loop sleeps 10 seconds between iterations, so the "open" capture can be up to 10 seconds late. During a volatile candle open, BTC can move $50–100 in 10 seconds. This adds noise to `brti_drift_since_open` — the baseline isn't exactly T=0.

**Severity:** Low. 10s timing slack on a 6-minute drift window is ~2.5% contamination. Not worth engineering complexity now.

**Future fix (if brti_drift becomes important):** Capture BRTI at the exact candle open by storing it in `_candle_ticker_map` alongside the ticker, reading it from the BRTI aggregator at candle detection time in `_process_market` or `_run_cycle`.

---

### 2C. tick_count normalization for time-of-day

**The concern:** OKX BTC-USDT-SWAP tick rate varies dramatically by session. An Asian session candle might produce 22,000 ticks at 2am UTC while a weekend US candle produces 5,000 at 2pm UTC. Raw `tick_count_since_open` conflates "high conviction" with "high-activity session."

**Severity:** Low. The model also has `hour_sin`/`hour_cos` implicitly via correlated features, and XGBoost can learn cross-feature interactions. Not worth complicating data collection now.

**Future fix:** Add `tick_rate_vs_expected` = `tick_count_since_open / historical_avg_ticks_at_this_hour_of_day`. Requires a rolling hourly tick baseline logged separately — a future session project.

---

## Priority 3 — Architecture for V2

### 3A. Real-time scoring instead of frozen snapshot

**The limitation:** The snapshot fires once at the first 40-60% crossing (~41%). If `_run_cycle` fires at 58%, the model score is 17% of candle progress stale — about 2.5 minutes old. During a fast-moving candle, the CVD picture can completely reverse in that window.

**V2 design:** Instead of scoring from the frozen snapshot, re-score on every `_run_cycle` during the 40-60% window using live accumulator state:

```python
# In _process_market, when 0.40 <= candle_progress <= 0.60:
if self._mid_candle_model is not None and _snap_ob:
    _live_snapshot = self._build_live_mid_snapshot(_in_progress_ts, ...)
    _live_prob = self._mid_candle_model.predict(_live_snapshot)["prob_up"]
    # Use _live_prob for Gate 15 directly, no Redis intermediate
```

**Training implication:** Training data would need to use the snapshot-at-decision-time, not a frozen 41% snapshot. This changes the training query. Keep the frozen snapshot approach for V1 training; switch to real-time for V2 after enough data to retrain.

**Why not now:** V1 frozen snapshot is consistent between training and inference (both use the 40-60% window snapshot). Real-time scoring requires a training data pipeline change too.

---

### 3B. Multiple snapshots per candle window

**The limitation:** A single snapshot at 41% provides one data point. The model can't tell if the CVD at 41% was building (last 5 ticks all buys) or fading (flat for 3 ticks then a few sells).

**V2 design:** Take 3 snapshots at ~40%, ~50%, ~60% progress. Log `cvd_delta_40_50` (CVD change from 40% to 50% snapshot) and `cvd_delta_50_60` as additional features. This captures acceleration vs deceleration of order flow within the entry window.

**Complexity:** Moderate. Requires multiple snap dicts per candle and a different training schema. V2 project.

---

### 3C. Gate 12 second window — specify the condition precisely

When Gate 12 allows the 40-60% window, it should only bypass the progress cap check — it must still apply the 3% floor and the 5% cap for high-vol/wide-spread conditions. Design:

```python
_IN_MID_WINDOW = 0.40 <= candle_progress <= 0.60
_mid_model_active = getattr(self, "_mid_candle_model_loaded", False)

if not _IN_MID_WINDOW:
    # Normal early-entry cap check
    if candle_progress < _PROGRESS_FLOOR:
        return fail(12, f"Below floor {_PROGRESS_FLOOR}")
    if candle_progress > _cap:
        return fail(12, f"Exceeds cap {_cap:.2f}")
elif not _mid_model_active:
    # Mid-window but no model loaded — block same as cap
    return fail(12, f"Mid-candle window requires model (not loaded)")
# else: mid-window + model active → continue to Gate 15
```

The 5% hard cap (high-vol + wide-spread) still applies to the mid-window via Gate 15's `prob < 0.38 / > 0.62` threshold — that threshold is conservative enough to act as an implicit volatility filter.

---

## Priority 4 — Monitoring & Observability

### 4A. Log mid-candle model miss rate

After Gate 15 goes live, log to `gate_rejections` (gate=15) the same way other gates do. This gives you:
- How often Gate 15 fires
- What the mid_candle_prob was when it blocked
- Whether the blocked trades would have won (counterfactual)

This is the data that tells you whether to tighten (0.38/0.62 → 0.40/0.60) or loosen the thresholds over time.

---

### 4B. Regime diversity check before Gate 15 deployment

Current data: all collected in Extreme Fear (F&G 8) sustained bear. The mid-candle model has never seen a bullish trending candle, a ranging session, or a genuine reversal. `cvd_brti_divergence = 1.0` on 19/20 rows means the divergence signal — one of the primary alpha sources — has essentially never fired.

Before deploying Gate 15, add a manual check:

```sql
SELECT
  CASE WHEN brti_drift_since_open > 50 THEN 'up_strong'
       WHEN brti_drift_since_open < -50 THEN 'down_strong'
       ELSE 'flat' END as regime,
  COUNT(*) as n,
  AVG(CAST(btc_direction AS FLOAT)) as pct_up
FROM candle_features
WHERE cvd_since_open IS NOT NULL AND btc_direction IS NOT NULL
GROUP BY 1;
```

If >85% of rows are `down_strong`, Gate 15 likely overfits to bear. Wait for at least 15% of rows in each bucket before going live.

---

## Summary — What to Build Next Session

| # | What | File | Priority |
|---|---|---|---|
| 1A | Gate 15 candle_ts validation | `pretrade_checklist.py` + main.py | Must-have before Gate 15 |
| 1B | `tick_count > 200` filter in training | `scripts/train_mid_candle.py` | Must-have before first train |
| 1C | Log `k5_candle_ts` in snapshot | `main.py` + schema migration | Do it now, costs nothing |
| 2A | Kalshi importance check → decide | Post first-train analysis | After 200 rows |
| 3C | Gate 12 second window exact logic | `pretrade_checklist.py` | Before Gate 15 |
| 4A | Gate 15 gate_rejections logging | `main.py` + `pretrade_checklist.py` | With Gate 15 |
| 4B | Regime diversity check | Manual SQL | Before Gate 15 deploy |

Items 1B and 1C can go in right now (data collection improvements). Items 1A, 3C, and 4A are Gate 15 integration — one build session when the data gate is hit (~June 10-11).
