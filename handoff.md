# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min up/down markets). Forecast direction via Kronos + XGBoost regime classifier + DeepSeek gate, size with fractional Kelly, run 7 pre-trade gates.

**Current focus:** Accumulate 500 training-ready 21-feature rows (~June 2), train and deploy the RegimeModel, then flip `PAPER_TRADING=false` and go live (~June 5–7).

---

## Current Progress

**As of 2026-05-24 session 6: Deribit Options Feed COMPLETE — 6 new regime features (22–27) live, `deribit_stale=0` accumulation started.**

**Session 6 design decisions (implemented 2026-05-24):**
- **Feature expansion: 21 → 27 features.** Six new features added to `_FEATURE_ORDER` (features 22–27):
  - `atm_iv` — Deribit near-term at-the-money implied vol (interpolated between bracketing strikes, annualised %)
  - `iv_rv_spread` — ATM IV minus `brti_volatility_1h` (derived in `_get_market_context`, not written by the feed)
  - `pcr_oi` — Put/call ratio by open interest for the near expiry (neutral fallback = 1.0)
  - `term_structure_slope` — (far_atm_iv − near_atm_iv) / near_atm_iv; positive = contango, negative = backwardation
  - `skew_25d` — 25Δ put IV minus 25Δ call IV; negative = market hedging downside
  - `kalshi_spread_normalized` — Kalshi bid-ask spread in cents / 100; injected inline in `_process_market` via new `update_kalshi_spread()` on SignalFusionEngine
- **New file:** `btc_kalshi_system/data/deribit_options_feed.py` — isolated async feed, no auth, Deribit public REST
  - Redis: `options:features` (TTL 600s) + `options:features:lkg` (TTL 14400s = 4h)
  - Flat-interval retry on failure (same pattern as derivatives_feed); stateless REST, no reconnect complexity
  - On failure: skip write, let key expire, LKG survives, rows get `deribit_stale=1`
- **Stale policy: STRICT.** New `deribit_stale INTEGER DEFAULT 1` column in `trades.db`. Historical rows default to 1. `train_regime.py` adds `_EXTRA_FILTERS_27` requiring `deribit_stale = 0` alongside the existing NOT NULL checks. Old 21-feature retrain path unchanged.
- **Integration (Approach A):** `_get_market_context()` reads and merges `options:features` into the context dict (same pattern as `regime:derived_context`). `_deribit_lkg=True` marker added when LKG is used — triggers `deribit_stale=True` in `_regime_features()`.
- **ATM IV computation:** interpolate between two bracketing strikes; skip expiries with < 3 days to expiry; filter strikes with OI < 10.
- **Term structure:** compare ATM IV for nearest two valid expiries (both must have ≥ 3 days remaining).
- **25Δ skew:** use `spot × (1 ± 0.25 × atm_iv/100 × sqrt(T))` to approximate 25Δ strike locations, then look up nearest listed IV. `skew_25d = put_iv − call_iv`.
- **DeepSeek prompt:** add OPTIONS MARKET section between DERIVATIVES and SENTIMENT.
- **Feature order contract:** `_FEATURE_ORDER` in `regime_model.py`, `_FEATURE_COLS` in `train_regime.py`, and `_regime_features()` dict in `fusion.py` must all be updated consistently (existing test `test_feature_order` enforces this).

---

**As of 2026-05-24 session 5: DeepSeek enrichment complete — ~15 real signals now sent to DeepSeek V3. System live and collecting.**

- `PAPER_TRADING=true` in `.env`
- **~54 trades/day. 500 rows by ~June 2.**
- Stats: 378 total trades, 207W/171L (54.7%), Net P&L: -$97.72
- System running on PID 61960 — confirm: `ps aux | grep "[Pp]ython.*main\.py"` (**restart main.py after this merge to pick up session 5 changes**)
- Latest commit: merge of session 5 DeepSeek enrichment
- Test suite: **290 passing**
- gate_rejections verified (session 3): 2 rows written within first signal cycle post-restart, all 21 features captured.

**All phases complete:**
- Phase 0: CVD soft gate (Gate 7)
- Phase 1: 6→21 feature expansion
- Phase 2: PositionMonitor (mid-trade exit at T+5/T+10)
- Phase 2b: `large_print_direction` (21st feature) + Dynamic Kelly (chop/tape/streak shrinks)
- Bugfix: CVD ring buffer stale-timestamp detection
- Bugfix: DerivativesFeed reconnection on any fetch failure (not just 403)
- Phase 3a: P&L formula explicit direction branch (auditable, math unchanged)
- Phase 3b: CalibrationDriftMonitor (rolling 20-trade Brier score drift detection)
- Phase 3c: StratifiedEdgeTracker (per-regime edge observability, not yet gating) — ✅ FIXED session 4: `"unknown"` bucket added, CalibrationDriftMonitor ZeroDivision guard added
- Phase 3d: gate_rejections table — logs every blocked trade with full 21-feature vector + counterfactual outcome resolution ~15min later
- Session 5: DeepSeek enrichment — switch to V3 (deepseek-chat), ~15-signal prompt, Fear & Greed, volume ratio, composite price, derived context ring, recent outcomes

**Go-live thresholds (both must be met):**
- ≥ 500 resolved trades total → calibrator (~May 27, nearly there)
- ≥ 500 new 21-feature training rows → regime model (~June 2)
- ≥ 500 rows with `deribit_stale=0` → 27-feature model retrain (deferred; collect after Deribit feed is live)

**Timeline:**
| Date | Milestone |
|------|-----------|
| ~May 26–27 | 500 total trades → train calibrator |
| ~June 2–3 | 500 new 21-feature rows → `python3 scripts/train_regime.py` (21-feature model) |
| ~June 2–3 | Deploy 21-feature regime model → flip `REGIME_GATE2_ENFORCING=true` |
| ~June 5–7 | ~50 shadow trades observed → flip `PAPER_TRADING=false` |
| ~June 10+ | Deribit feed live long enough → retrain with 27-feature model when ≥500 `deribit_stale=0` rows |

---

## Architecture

**27-feature `_FEATURE_ORDER`** (identical in `regime_model.py`, `train_regime.py`, and `fusion._regime_features()` dict keys — mismatch silently corrupts training). Features 1–21 are live; features 22–27 added in session 6:
```
funding_rate, funding_rate_trend, oi_delta_pct, cvd_normalized, basis_spread_pct,
brti_volatility_1h, cvd_velocity, cvd_acceleration, brti_momentum_5min,
brti_momentum_15min, candle_progress, hour_sin, hour_cos, kalshi_implied_prob,
funding_window_proximity, trend_slope_1h, trend_r2_1h, hourly_sr_proximity,
range_breakout_flag, tape_speed_tpm, large_print_direction,
atm_iv, iv_rv_spread, pcr_oi, term_structure_slope, skew_25d,
kalshi_spread_normalized
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
| `atm_iv`, `pcr_oi`, `term_structure_slope`, `skew_25d` | `deribit_options_feed.py` → Redis `options:features` |
| `iv_rv_spread` | Derived in `_get_market_context()`: `atm_iv − brti_volatility_1h` |
| `kalshi_spread_normalized` | Inline in `_process_market()` via `update_kalshi_spread()` |

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

## Files Touched This Session (2026-05-24, session 6)

**Session 6 (Deribit Options Feed — features 22–27):**

| File | Change |
|------|--------|
| `btc_kalshi_system/data/deribit_options_feed.py` | **New** — async Deribit public REST feed; computes `atm_iv`, `pcr_oi`, `term_structure_slope`, `skew_25d`; writes `options:features` (TTL 600s) + `options:features:lkg` (TTL 14400s); retries on failure |
| `btc_kalshi_system/signal/fusion.py` | Added `update_kalshi_spread()`; added 6 new features (22–27) at bottom of `_regime_features()`; changed return type to `tuple[dict, bool, bool]` (adds `deribit_stale`); added `deribit_stale: bool` to `TradingSignal` |
| `btc_kalshi_system/models/regime_model.py` | Added 6 new keys to `_FEATURE_ORDER` (now 27 entries) |
| `scripts/train_regime.py` | Added 6 new keys to `_FEATURE_COLS` (now 27); added `_EXTRA_FILTERS_27`; updated `_build_query()` to accept `use_27` flag |
| `btc_kalshi_system/models/deepseek_parser.py` | Added OPTIONS MARKET section to `_PROMPT_TEMPLATE` and 6 corresponding format vars in `_build_prompt()` |
| `btc_kalshi_system/execution/position_monitor.py` | Updated `_regime_features()` unpack to 3-tuple |
| `main.py` | Import + instantiate `DeribitOptionsFeed`; add to `asyncio.gather()`; `update_kalshi_spread()` call before `update_kalshi_mid()`; `options:features` + LKG merge in `_get_market_context()`; `iv_rv_spread` derivation; 7 new `_TRADES_COLUMN_MIGRATIONS` entries; 7 new columns in `_record_trade_sqlite()` INSERT |
| `tests/data/test_deribit_options_feed.py` | **New** — 11 TDD tests for DeribitOptionsFeed (feed writes, LKG, expiry filtering, pcr_oi, interpolation, failure handling) |
| `tests/signal/test_fusion_deribit_features.py` | **New** — 11 TDD tests for fusion deribit features (27-key check, stale flags, kalshi_spread, pcr_oi default) |
| `tests/signal/test_feature_order.py` | Updated to 27 features; 3-tuple unpack |
| `tests/signal/test_regime_features.py` | Updated all `_regime_features()` unpacks to 3-tuple |
| `tests/models/test_regime_model.py` | Updated `_synthetic_features` to 27 cols; added 6 new keys to `_feature_dict()` |
| `handoff.md` | Session 6 update |

**Test suite: 312 passing (was 290).**

**`deribit_stale=0` rows begin accumulating from 2026-05-24. Do NOT retrain the 27-feature model until ≥500 `deribit_stale=0` rows are collected.**

---

**Session 4 (StratifiedEdgeTracker + CalibrationDriftMonitor bugfixes):**

| File | Change |
|------|--------|
| `btc_kalshi_system/signal/stratified_edge_tracker.py` | Added `"unknown"` to `REGIMES` — positions resolving with `deepseek_regime="unknown"` now tracked in their own bucket instead of silently dropped |
| `btc_kalshi_system/signal/calibration_drift_monitor.py` | Added `if not self._history: return` guard at top of `_recompute_window()` — prevents ZeroDivisionError when Redis partially writes `_KEY_TOTAL_COUNT` but `_KEY_HISTORY` is lost |
| `tests/signal/test_stratified_edge_tracker.py` | 2 new tests: `"unknown"` bucket records + appears in `summary()` |
| `tests/signal/test_calibration_drift_monitor.py` | 1 new test: `_recompute_window()` with empty history does not raise |

**Session 3 (gate_rejections):**

| File | Change |
|------|--------|
| `main.py` | `_CREATE_GATE_REJECTIONS_TABLE` + `_GATE_REJECTIONS_COLUMN_MIGRATIONS` (includes `aged_out INTEGER DEFAULT 0` migration); init at startup; write row on checklist failure in `_process_market`; `_resolve_gate_rejections()` with `aged_out=1` age-out (outcome stays NULL), `aged_out = 0` filter + `LIMIT 50` on resolution query; called from main loop |
| `tests/execution/test_gate_rejections.py` | **New** — 5 TDD tests: write-on-failure, win resolution, loss resolution, young-row skip, age-out flag |
| `handoff.md` | Session 3 update |

**gate_rejections design notes:**
- `outcome` is NULL for aged-out rows — use `WHERE aged_out = 0` to filter them out of analysis
- `aged_out = 0` filter on resolution SELECT prevents re-querying aged rows; `LIMIT 50` bounds API calls on first run
- New `aged_out` column arrives via `_GATE_REJECTIONS_COLUMN_MIGRATIONS` (idempotent ALTER TABLE), not in `_CREATE_GATE_REJECTIONS_TABLE` — safe on existing DBs

**Session 2:**

| File | Change |
|------|--------|
| `btc_kalshi_system/models/deepseek_parser.py` | Switch to deepseek-chat (V3); add `response_format`+`max_tokens`; replace prompt with 15-signal template; rewrite `_build_prompt` |
| `btc_kalshi_system/data/fear_greed.py` | **New** — Fear & Greed fetcher with Redis caching (TTL 1h) |
| `btc_kalshi_system/data/derivatives_feed.py` | Add `_fetch_volume_ratio()` + Fear & Greed call; write `volume_ratio_1h`, `fear_greed_value`, `fear_greed_label` to `regime:features` |
| `main.py` | Move `composite_price` before `update_market_context`; write `regime:derived_context` (TTL 120s) after each signal; extend `_get_market_context` to merge derived context, fear_greed nested dict, recent Kalshi outcomes |
| `tests/data/test_fear_greed.py` | **New** — 3 tests (cache hit, live fetch+cache write, failure→None) |
| `tests/models/test_deepseek_parser.py` | Update `_good_context()`; update `test_prompt_includes_market_context_values`; add 4 new prompt tests (CVD, fear_greed, recent_outcomes, graceful n/a) |
| `handoff.md` | Session 5 update |

**Prior session files documented in git log — see commits `befb381` and earlier.**

---

## Path to Going Live / Pre-go-live Checklist

| Item | Status |
|------|--------|
| CalibrationDriftMonitor | ✅ COMPLETE — wired, tests passing |
| StratifiedEdgeTracker | 🔄 IN PROGRESS — wired for observability; not yet gating |
| Merge feature/20-features-position-monitor → main | ✅ COMPLETE — fast-forward merge, main.py restarted, all 21 features confirmed |
| StratifiedEdgeTracker `"unknown"` bucket + CalibrationDriftMonitor guard | ✅ COMPLETE |

---

## Next Steps

0. ✅ **Deribit Options Feed implemented (session 6).** Restart main.py now: `ps aux | grep "[Pp]ython.*main\.py"` → kill PID → restart. Within 5 minutes verify: `redis-cli get options:features` returns a JSON dict with `atm_iv`, `pcr_oi`, `term_structure_slope`, `skew_25d`; `redis-cli ttl options:features` returns 400–600; `options:features:lkg` also populated. After one trade cycle, `trades.db` will have 7 new columns. **Do NOT retrain the 27-feature model until ≥500 `deribit_stale=0` rows accumulated — this is separate from the 21-feature retrain gate.**

1. **Wire StratifiedEdgeTracker into Gate 4 after ~50 trades.** After session 4 fixes land, run ~50 trades and check `self._stratified_edge.summary()`. Wire `is_above_threshold(signal.deepseek_regime)` into Gate 4 — if a regime has fewer than 1 recorded trade, `is_above_threshold` returns `False` (blocks). **Important:** do NOT compare `summary()` against `self._edge_tracker.current_edge()` for parity — they measure the same metric (realized edge) but over different populations (global vs per-regime). A difference is expected and not a bug. Instead, validate that `"unknown"` bucket has low count and non-`"unknown"` buckets are accumulating.

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

6. **Post-go-live (deferred):** Kalshi intra-cycle YES momentum (needs new polling infra). Slippage gate for 15min markets (needs 200+ spread samples first).

   **Gate 8 (candle_progress / UTC dark gate) — DEFERRED until more data:** Only 33/384 trades have `candle_progress` populated, zero above 0.85. Revisit once we have ≥200 trades with valid candle_progress values. Do not implement until data density justifies it.

---

## exit_reason Diagnosis (2026-05-24)

- **`regime_model._clf` is None** — `RegimeModel.__init__()` sets `_clf = None` and it stays None until `train_regime.py` is run; with only ~15 training-ready rows (need 500), it has never been trained, so `PositionMonitor._evaluate()` hits the `if self.regime_model._clf is None:` bootstrap branch, collects a snapshot, and returns early — `_execute_exit()` (where `exit_reason` is written) is never reached.
- **PositionMonitor IS scheduled** — `self._position_monitor.run()` is in the `asyncio.gather()` call at `main.py:246`, so the coroutine is running; it is not the issue.
- **Trades last long enough for T+5 to fire** — querying `trades.db`: avg time-remaining-in-15min-window when a trade enters is ~599s (well above the 300s T+5 threshold); 324/378 resolved trades (86%) had ≥300s remaining, so trade duration is not the blocker.

**Conclusion:** `exit_reason` will stay NULL until `train_regime.py` is run and the model is loaded. Expected behavior during the bootstrap accumulation phase.

---

## Context / Gotchas

- **Test suite: 290 pass.** `python3 -m pytest` from project root.

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

- **StratifiedEdgeTracker has 5 regimes:** `trending_up`, `trending_down`, `ranging`, `high_uncertainty`, `"unknown"`. Trades where DeepSeek context is stale (`OpenPosition.deepseek_regime = "unknown"`) go into the `"unknown"` bucket rather than being silently dropped. Without this, the global EdgeTracker and stratified totals diverge, making parity checks misleading, and Gate 4 would silently block any trade arriving with a stale regime.

- **StratifiedEdgeTracker measures realized edge, not calibration.** `current_edge(regime)` = `mean(outcome - market_price)`. Read it as "are we buying at prices that beat the market in this regime?" The `predicted_prob` field is stored in Redis but never used in any computation — do not treat `current_edge` as a calibration metric.

- **CalibrationDriftMonitor has a ZeroDivision guard in `_recompute_window()`.** If Redis partially writes (total_count written, history lost), `_history` can be empty while `_total_count % 20 == 0` on restart, triggering `_mean_brier` on an empty deque. Guard: `if not self._history: return` at the top of `_recompute_window()`.

- **`regime:derived_context` (TTL 120s)** — written by `_process_market` after each signal; DeepSeek reads it one cycle later via `_get_market_context`. One-cycle lag on momentum/trend/range data is intentional and acceptable.

- **Deribit feed uses LKG with 4h TTL** (`options:features:lkg`). When LKG is used, context dict carries `_deribit_lkg=True` — `_regime_features()` must detect this and set `deribit_stale=True`. Unlike `regime:features` (LKG rows still trade), Deribit LKG rows trade but are excluded from the 27-feature retrain (strict stale policy).

- **`deribit_stale INTEGER DEFAULT 1`** — ALL historical rows start as stale. The 27-feature retrain requires `features_stale=0 AND deribit_stale=0`. The 21-feature retrain (running first) only requires `features_stale=0`. Do not conflate the two stale flags.

- **`iv_rv_spread` is NOT written by `deribit_options_feed.py`** — it is a derived field computed in `_get_market_context()` from `ctx["atm_iv"] - ctx["brti_volatility_1h"]`. Both keys must be present; if either is missing, `iv_rv_spread` defaults to 0.0 and `deribit_stale=True`.

- **Deribit ATM IV is annualised percentage** (e.g., `55.2` = 55.2% annualised). Do not divide by 100 when writing to Redis — store as the raw percentage float. `_regime_features()` reads and uses it as-is.

- **Skip Deribit expiries with < 3 days to expiry** — front-month IV spikes near expiry due to theta, not regime. Standard expiry is Friday. Parse expiry from instrument name (e.g., `BTC-27JUN25-100000-C` → 27 Jun 2025).

- **`pcr_oi` neutral fallback = 1.0** (not 0.0). A ratio of 1.0 means equal put and call positioning — true neutral. 0.0 would imply zero put OI which is misleading.

- **`_FEATURE_COLS_LEGACY` in `train_regime.py` stays at 6 features.** Do not modify it. The legacy path is for very old rows — not related to the Deribit expansion.

- **`dump.rdb` and `trades.db.bak.*` must NOT be committed.**

- **`RANGING_SHRINK=0.7`, `_BOOTSTRAP_SHRINK=0.8`, `_UNCERTAINTY_SHRINK=0.5`** — do not equate.

- **DeepSeek returns `NEUTRAL_DEFAULT` on 402, not `SAFE_DEFAULT`.**

- **DerivativesFeed re-resolves on ANY exception** (commit `229b88b`). Prior to this fix, only 403/Forbidden triggered failover; timeouts/resets silently kept a dead session alive. If the feed goes quiet again, check `redis-cli ttl regime:features` — TTL of -2 means it expired and feed is down. Restart main.py to recover.

- **Restart procedure:** `ps aux | grep "[Pp]ython.*main\.py"` → kill PID → `cd /Users/ezrakornberg/Kronos\ V2 && python3 main.py > /tmp/kronos_restart.log 2>&1 &` — verify first DerivativesFeed log shows all 21 features including `large_print_direction`.

- **Feed health check:** `redis-cli ttl regime:features` should return 400–600. If -2, feed is down. `redis-cli get regime:features:lkg` shows LKG age via `_lkg_written_at` field.
