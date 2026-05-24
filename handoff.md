# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min up/down markets). Forecast direction via Kronos + XGBoost regime classifier + DeepSeek gate, size with fractional Kelly, run 7 pre-trade gates.

**Current focus:** Accumulate 500 training-ready 21-feature rows (~June 2), train and deploy the RegimeModel, then flip `PAPER_TRADING=false` and go live (~June 5–7).

---

## Current Progress

**As of 2026-05-24 session 2: 12 training-ready 21-feature rows. System live and collecting.**

- `PAPER_TRADING=true` in `.env`
- **~54 trades/day. 500 rows by ~June 2.**
- Stats: 378 total trades, 207W/171L (54.7%), Net P&L: -$97.72
- System running on PID 55789 — confirm: `ps aux | grep "[Pp]ython.*main\.py"`
- Latest commit: `4072ee4` (feature/20-features-position-monitor branch — **restart main.py after merging**)
- Test suite: **275 passing**

**All phases complete:**
- Phase 0: CVD soft gate (Gate 7)
- Phase 1: 6→21 feature expansion
- Phase 2: PositionMonitor (mid-trade exit at T+5/T+10)
- Phase 2b: `large_print_direction` (21st feature) + Dynamic Kelly (chop/tape/streak shrinks)
- Bugfix: CVD ring buffer stale-timestamp detection
- Bugfix: DerivativesFeed reconnection on any fetch failure (not just 403)
- Phase 3a: P&L formula explicit direction branch (auditable, math unchanged)
- Phase 3b: CalibrationDriftMonitor (rolling 20-trade Brier score drift detection)
- Phase 3c: StratifiedEdgeTracker (per-regime edge observability, not yet gating)

**Go-live thresholds (both must be met):**
- ≥ 500 resolved trades total → calibrator (~May 27, nearly there)
- ≥ 500 new 21-feature training rows → regime model (~June 2)

**Timeline:**
| Date | Milestone |
|------|-----------|
| ~May 26–27 | 500 total trades → train calibrator |
| ~June 2–3 | 500 new 21-feature rows → `python3 scripts/train_regime.py` |
| ~June 2–3 | Deploy regime model → flip `REGIME_GATE2_ENFORCING=true` |
| ~June 5–7 | ~50 shadow trades observed → flip `PAPER_TRADING=false` |

---

## Architecture

**21-feature `_FEATURE_ORDER`** (identical in `regime_model.py`, `train_regime.py`, and `fusion._regime_features()` dict keys — mismatch silently corrupts training):
```
funding_rate, funding_rate_trend, oi_delta_pct, cvd_normalized, basis_spread_pct,
brti_volatility_1h, cvd_velocity, cvd_acceleration, brti_momentum_5min,
brti_momentum_15min, candle_progress, hour_sin, hour_cos, kalshi_implied_prob,
funding_window_proximity, trend_slope_1h, trend_r2_1h, hourly_sr_proximity,
range_breakout_flag, tape_speed_tpm, large_print_direction
```

**Feature sources:**
| Feature | Source |
|---------|--------|
| Features 1–6 | `derivatives_feed.py` → Redis `regime:features` |
| `cvd_velocity`, `cvd_acceleration` | Redis sorted set `regime:cvd_history` |
| `brti_momentum_*`, `candle_progress`, `hour_*`, `trend_*`, `hourly_sr_proximity`, `range_breakout_flag` | `fusion._regime_features()` from OHLCV |
| `tape_speed_tpm` | `store.get_raw_ticks(60)` |
| `large_print_direction` | `derivatives_feed.py` fetch_trades (net dir from prints > 2× avg size) |
| `kalshi_implied_prob` | `market_context["kalshi_mid_cents"]` / 100 |
| `funding_window_proximity` | UTC time proximity to 00/08/16h funding |

**Dynamic Kelly shrinks** (multiplicative, applied after existing cap):
| Condition | Shrink |
|-----------|--------|
| `abs(range_breakout_flag) < 0.15` | × 0.70 |
| `tape_speed_tpm < 0.20` | × 0.80 |
| `loss_streak >= 3` | × 0.60 |

Streak tracked in Redis key `trading:loss_streak` — cleared on win, incremented on loss in `main.py _check_resolutions`.

**Gate 7 (CVD soft gate):** `CVD_GATE_THRESHOLD = 0.3`. YES→UP with CVD < -0.3 fails. NO→DOWN with CVD > +0.3 fails.

---

## What Worked

- **3-file feature order contract enforced by test.** `python3 -m pytest tests/ -k "feature_order"` catches any mismatch between `regime_model.py`, `train_regime.py`, `fusion.py`.
- **fakeredis injection** for testing Redis-dependent code without a live Redis server.
- **TTL=600s + refresh every 240s** (not TTL=refresh) — gives headroom so `regime:features` never expires between writes.
- **LKG fallback** (`regime:features:lkg`, TTL=24h) — real stale data during outages rather than zeros.
- **CVD buffer two-mode stale detection:** count < 5 (cold) OR most recent timestamp > 360s old (feed gap). Both zero velocity and mark stale.
- **`zremrangebyscore` + `zremrangebyrank`** on CVD ring buffer — prevents stale timestamps accumulating across outages.
- **Per-side position cap Redis-backed** — survives multiple processes.
- **Dynamic Kelly streak shrink** — verified working: after 4-loss streak, Kelly dropped from ~$20 to ~$6.
- **DerivativesFeed re-resolve on any exception** — always closes dead exchange and re-resolves fresh instance on any failure, not just 403. Prevents feed staying broken indefinitely on timeouts/resets/rate limits.

## What Failed / Avoided

- **Blanket `MAX_POSITIONS_PER_TICKER=3`** — replaced by per-side cap.
- **20¢ entry price floor** — added then removed; sub-20¢ data too thin.
- **In-memory position count** — broke under multiple processes.
- **`floor_strike=0` accepted as valid** — made Kronos compute P(BTC > $0) ≈ 100%.
- **Circuit breaker in paper mode** — tripped at -$200, halting data collection.
- **Backfilling pre-instrumentation trades** — funding/OI/CVD not reconstructable.
- **CVD buffer freshness check at 180s** — false-positives on healthy cycles (feed writes every 240s). Use 360s.
- **Hardcoded timestamps in CVD test mocks** — caused freshness check to fire in tests. Always use `time.time()` in test setups for CVD entries.
- **403-only exchange failover** — only re-resolved on 403/Forbidden; timeouts and connection resets retried the same dead session object, leaving the feed silently broken for hours. Fixed: re-resolve on ANY exception.

---

## Files Touched This Session (2026-05-24)

| File | Change |
|------|--------|
| `btc_kalshi_system/data/derivatives_feed.py` | Removed 403-only branch; re-resolve exchange on ANY fetch exception |
| `btc_kalshi_system/portfolio/monitor.py` | P&L explicit direction branch in `resolve_trade()`; `deepseek_regime: str = "unknown"` added to `OpenPosition` |
| `btc_kalshi_system/signal/calibration_drift_monitor.py` | **New** — rolling 20-trade Brier score drift detector |
| `btc_kalshi_system/signal/stratified_edge_tracker.py` | **New** — per-regime rolling edge tracker (4 regimes, 50-trade deques) |
| `tests/portfolio/test_monitor.py` | 2 new NO-bet P&L tests |
| `tests/signal/test_calibration_drift_monitor.py` | **New** — 6 tests (incl. Redis restart test) |
| `tests/signal/test_stratified_edge_tracker.py` | **New** — 4 tests |
| `main.py` | Wire CalibrationDriftMonitor + StratifiedEdgeTracker; pass `deepseek_regime` to `OpenPosition` |
| `handoff.md` | exit_reason diagnosis + session update |

*(All prior session files documented in git log — see commits `befb381` and earlier.)*

---

## Path to Going Live / Pre-go-live Checklist

| Item | Status |
|------|--------|
| CalibrationDriftMonitor | ✅ COMPLETE — wired, tests passing |
| StratifiedEdgeTracker | 🔄 IN PROGRESS — wired for observability; not yet gating |
| Merge feature/20-features-position-monitor → main | ⬜ PENDING — restart main.py after merge |

---

## Next Steps

1. **Confirm StratifiedEdgeTracker parity with EdgeTracker on 50+ shared trades, then switch Gate 4 to use stratified edge for current regime.** After merging and running ~50 trades, compare `self._stratified_edge.summary()` vs `self._edge_tracker.current_edge()` — if they agree within noise, wire `is_above_threshold(signal.deepseek_regime)` into Gate 4.

2. **Wait for ~May 26–27:** Total resolved trades will hit 500 → train the calibrator. Check: `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL` — need ≥ 500.

3. **Monitor 21-feature row accumulation daily:**
   ```
   python3 scripts/regime_health_check.py
   ```
   Need `Training-ready (21-feature): 500`. Target ~June 2 at current rate.

4. **Train regime model when 21-feature rows ≥ 500:**
   ```
   python3 scripts/train_regime.py --dry-run
   ```
   Check Brier < 0.25, Kronos agreement > 55%. If sane, run without `--dry-run` → `models/regime.pkl`. Restart main.py. Gate 2 runs shadow mode by default — observe ~50 trades before flipping `REGIME_GATE2_ENFORCING=true`.

5. **After ~50 shadow trades with regime model live:** Flip `PAPER_TRADING=false` in `.env` and restart to go live.

6. **Post-go-live (deferred):** Kalshi intra-cycle YES momentum (needs new polling infra). Slippage gate for 15min markets (needs 200+ spread samples first). Gate 8 candle_progress gate deferred — only 33/384 trades have candle_progress populated, zero above 0.85.

---

## exit_reason Diagnosis (2026-05-24)

- **`regime_model._clf` is None** — `RegimeModel.__init__()` sets `_clf = None` and it stays None until `train_regime.py` is run; with only ~15 training-ready rows (need 500), it has never been trained, so `PositionMonitor._evaluate()` hits the `if self.regime_model._clf is None:` bootstrap branch, collects a snapshot, and returns early — `_execute_exit()` (where `exit_reason` is written) is never reached.
- **PositionMonitor IS scheduled** — `self._position_monitor.run()` is in the `asyncio.gather()` call at `main.py:246`, so the coroutine is running; it is not the issue.
- **Trades last long enough for T+5 to fire** — querying `trades.db`: avg time-remaining-in-15min-window when a trade enters is ~599s (well above the 300s T+5 threshold); 324/378 resolved trades (86%) had ≥300s remaining, so trade duration is not the blocker.

**Conclusion:** `exit_reason` will stay NULL until `train_regime.py` is run and the model is loaded. Expected behavior during the bootstrap accumulation phase.

---

## Context / Gotchas

- **Test suite: 259 pass.** `python3 -m pytest` from project root.

- **Feature order is a 3-file contract.** `regime_model.py` / `train_regime.py` / `fusion._regime_features()` must match exactly. Test: `python3 -m pytest tests/ -k "feature_order"`.

- **CVD ring buffer has TWO stale modes.** Cold (< 5 entries) and stale timestamp (most recent > 360s old). Both zero velocity/acceleration and set `stale=True`. The 360s threshold is intentional — feed writes every 240s, so 360s = missed one full cycle.

- **CVD test mocks must use `time.time()`**, not hardcoded epochs — otherwise freshness check fires.

- **`_FEATURE_COLS_LEGACY` stays at 6 features.** Do not add new features to it.

- **Do not add `_lkg` or `_lkg_written_at` to any feature list.** Corrupts XGBoost inputs.

- **Gate 6 guard `if signal.timeframe != "15min"` must stay.** Removing it blocks all 15min trades.

- **Gate 2 is in SHADOW mode** (`REGIME_GATE2_ENFORCING=false`). Do not flip until regime model has been live for ~50 trades.

- **PositionMonitor exit never calls `add_position()`.** Calls `remove_position()` first, then raw API. Do not "fix" this — it would break `MAX_POSITIONS_PER_TICKER_PER_SIDE=2`.

- **Kronos is blocking (2–3s on CPU).** Always `loop.run_in_executor(None, ...)`. Never call on event loop thread. Preload before asyncio starts (Apple Silicon segfault).

- **Two `brti_volatility_1h` implementations exist** — `DerivativesFeed` (Redis ticks) vs `fusion` (OHLCV pct_change). Do not consolidate.

- **Label semantics:** `y_up = int(direction == outcome)` = "did market close UP", not "did trade win". NO→DOWN win: `direction=0, outcome=1 → label=0`.

- **Loss streak Redis key:** `trading:loss_streak`. Integer. Cleared on win (`DELETE`), incremented on loss (`INCR`). Read by `PreTradeChecklist` before Kelly call.

- **Stale rows excluded from training.** `features_stale=1` rows are written with real values (0.0 fallback) but excluded from regime model training. Currently ~10 stale rows (~6%) — frozen since last restart, no new stale rows being generated.

- **`dump.rdb` and `trades.db.bak.*` must NOT be committed.**

- **`RANGING_SHRINK=0.7`, `_BOOTSTRAP_SHRINK=0.8`, `_UNCERTAINTY_SHRINK=0.5`** — do not equate.

- **DeepSeek returns `NEUTRAL_DEFAULT` on 402, not `SAFE_DEFAULT`.**

- **DerivativesFeed re-resolves on ANY exception** (commit `229b88b`). Prior to this fix, only 403/Forbidden triggered failover; timeouts/resets silently kept a dead session alive. If the feed goes quiet again, check `redis-cli ttl regime:features` — TTL of -2 means it expired and feed is down. Restart main.py to recover.

- **Restart procedure:** `ps aux | grep "[Pp]ython.*main\.py"` → kill PID → `cd /Users/ezrakornberg/Kronos\ V2 && python3 main.py > /tmp/kronos_restart.log 2>&1 &` — verify first DerivativesFeed log shows all 21 features including `large_print_direction`.

- **Feed health check:** `redis-cli ttl regime:features` should return 400–600. If -2, feed is down. `redis-cli get regime:features:lkg` shows LKG age via `_lkg_written_at` field.
