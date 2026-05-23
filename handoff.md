# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min
up/down markets). Forecast direction via Kronos + XGBoost regime classifier +
DeepSeek gate, size with fractional Kelly, run 6+ pre-trade gates.

**Current focus:** Accumulate 500 new 20-feature training rows (~June 2), train and
deploy the RegimeModel, then flip `PAPER_TRADING=false` and go live (~June 5–7).

---

## Current Progress

**As of 2026-05-23 ~18:30 UTC: 0 training-ready 20-feature rows. System is live and collecting.**

- `PAPER_TRADING=true` in `.env`
- **~49 trades/day. Expected 500 new 20-feature rows by ~2026-06-02.**
- Stats: 351 total trades / 0 training-ready 20-feature rows, 169W / 182L (48.1%), Net P&L: -$17.26
- System is **running** on PID 47293 — confirm with `ps aux | grep "[Pp]ython.*main\.py"`
- Latest commit: `bd80bc0`

**Implementation complete (2026-05-23).** Phase 0 (CVD gate), Phase 1 (6→20 feature
expansion), and Phase 2 (PositionMonitor) are all merged and live. System was restarted
at ~18:30 UTC. New trades will accumulate with all 20 features. The CVD ring buffer
requires 5 entries before `cvd_velocity`/`cvd_acceleration` are non-NULL — that means
the first ~20 minutes of trades will have `features_stale=1` and be excluded from
training. Existing 351 rows have NULLs for the 14 new columns and are excluded from
20-feature training (use `--legacy` to train on 6-feature subset).

**Go-live thresholds (both must be met):**
- ≥ 500 resolved trades total → calibrator (already met: 351 → will hit ~500 by ~May 27)
- ≥ 500 new 20-feature training rows → regime model (~June 2)

**Timeline:**
| Date | Milestone |
|------|-----------|
| May 23 (today) | Implementation complete, system restarted, 20-feature collection begins |
| ~May 26–27 | 500 total trades → train calibrator |
| ~June 2–3 | 500 new 20-feature rows → `python3 scripts/train_regime.py` |
| ~June 2–3 | Deploy regime model → flip `REGIME_GATE2_ENFORCING=true`, PositionMonitor exit becomes active |
| ~June 5–7 | ~50 shadow trades observed → flip `PAPER_TRADING=false` |

---

## Design Decisions Made This Session

### Feature expansion: 6 → 21 features

The full `_FEATURE_ORDER` (must be identical in `regime_model.py`, `train_regime.py`,
and returned dict from `fusion._regime_features()` — mismatch silently corrupts training):

```
funding_rate, funding_rate_trend, oi_delta_pct, cvd_normalized, basis_spread_pct,
brti_volatility_1h, cvd_velocity, cvd_acceleration, brti_momentum_5min,
brti_momentum_15min, candle_progress, hour_sin, hour_cos, kalshi_implied_prob,
funding_window_proximity, trend_slope_1h, trend_r2_1h, hourly_sr_proximity,
range_breakout_flag, tape_speed_tpm, large_print_direction
```

**New features and their sources:**
| Feature | Source | Notes |
|---------|--------|-------|
| `cvd_velocity` | Redis sorted set `regime:cvd_history` | Rate of change of CVD over 5 min |
| `cvd_acceleration` | Same | Delta of 5-min vs 10-min velocity |
| `brti_momentum_5min` | `store.get_ohlcv("5min")` | 1-candle pct return |
| `brti_momentum_15min` | Same | 3-candle pct return |
| `candle_progress` | `time.time() % 900 / 900` | Position in 15-min window [0,1] |
| `hour_sin` / `hour_cos` | UTC time | Cyclically encoded hour of day |
| `kalshi_implied_prob` | Orderbook mid-price | Set via `market_context["kalshi_mid_cents"]` before `get_signal()` |
| `funding_window_proximity` | UTC time | 1.0 at 00/08/16 UTC funding settlement, 0.0 at midpoint |
| `trend_slope_1h` | Last 12 5-min closes | Normalized linear regression slope |
| `trend_r2_1h` | Same | R² of that regression (trend quality) |
| `hourly_sr_proximity` | `store.get_ohlcv("1h")` | [0,1]: 0=at support, 1=at resistance |
| `range_breakout_flag` | `store.get_ohlcv("5min")` | Signed: +=bullish breakout, -=bearish |
| `tape_speed_tpm` | `store.get_raw_ticks(60)` | Tick count per minute / 100 |
| `large_print_direction` | `derivatives_feed.py` fetch_trades | Net dir score from prints > 2× avg size |

### Phase 0: CVD Soft Gate (Gate 7 in pretrade_checklist.py)
Based on: YES→UP with negative CVD = 32.3% win rate (handoff data). Statistical backing
from SLY bot analysis (28.6% win rate on opposing CVD trades).
- `CVD_GATE_THRESHOLD = 0.3` in `config.py`
- If `direction == 1` and `cvd_normalized < -0.3` → fail Gate 7
- If `direction == 0` and `cvd_normalized > +0.3` → fail Gate 7
- Uses `signal.regime_features` which already exists on `TradingSignal`

### Phase 2: PositionMonitor (mid-trade exit)
New coroutine in `btc_kalshi_system/execution/position_monitor.py`. Wakes every 60s.
At T+5 and T+10 per open position: re-runs `_regime_features()` + Kronos MC.
Exit condition: regime model AND Kronos both disagree with entry direction → submit
offsetting IOC order via `router.place_order()`.

**Critical rule:** `_execute_exit()` calls `portfolio_monitor.remove_position()` FIRST
then places the raw API call. It NEVER calls `add_position()`. This ensures the exit
does not count toward `MAX_POSITIONS_PER_TICKER_PER_SIDE=2`.

All T+5/T+10 snapshots (20 features + Kronos prob + exit_triggered) are written to
`trade_snapshots` table in `trades.db` regardless of whether exit fires. This is the
Approach C exit classifier training data.

Bootstrap mode: if `regime_model._clf is None`, PositionMonitor skips exit decision
but still writes snapshots. Do not skip snapshot collection.

### Dynamic Kelly (chop, tape, streak shrinks)
Three multiplicative shrinks applied after the existing Kelly cap:

| Condition | Threshold | Shrink | Rationale |
|-----------|-----------|--------|-----------|
| Chop | `abs(range_breakout_flag) < 0.15` | × 0.70 | No directional breakout = low conviction |
| Dead tape | `tape_speed_tpm < 0.20` | × 0.80 | < 20 TPM = thin, uncommitted flow |
| Loss streak | `loss_streak >= 3` | × 0.60 | Consecutive losses signal adverse conditions |

Worst case (all three active): `0.70 × 0.80 × 0.60 = 0.336×` base Kelly.

Loss streak tracked in Redis key `trading:loss_streak` (integer counter). Cleared on win, incremented on loss in `main.py _check_resolutions`. Read by `PreTradeChecklist` before each Kelly call.

New constants: `KELLY_CHOP_THRESHOLD`, `KELLY_CHOP_SHRINK`, `KELLY_TAPE_THRESHOLD`, `KELLY_TAPE_SHRINK`, `KELLY_STREAK_THRESHOLD`, `KELLY_STREAK_SHRINK` in `kelly.py`.

---

## What Worked (most recent first)

- **Phase 2b: large_print_direction + Dynamic Kelly (this session, commits `287d9e7` and `1dc0211`).** 21st regime feature from institutional flow (prints > 2× avg size). Three Dynamic Kelly shrinks: chop, dead tape, loss streak. 257 tests passing.

- **Phase 0/1/2 implementation (this session, commit `bd80bc0`).** CVD soft gate (Gate 7),
  20-feature expansion, PositionMonitor, 14 new DB columns, `trade_snapshots` table.
  245 tests passing. System live and collecting 20-feature rows.

- **SLY bot analysis (previous design session).** Validated CVD gate with real statistics
  (28.6% win rate on opposing-CVD trades). Provided roadmap of signals that actually
  move BTC/Kalshi markets.

- **Win rate by price script (`scripts/win_rate_by_price.py`, commit `a85c349`).**
  Run anytime: `python3 scripts/win_rate_by_price.py`. Flags: `--bucket 5`, `--dir yes/no`,
  `--min-trades N`. Current: 0–19¢ = 0% win, 60+¢ = 67–82%.

- **Entry price floor removed (commit `6f74f82`).** Sub-20¢ trades allowed.
  Monitor: if sub-20¢ win rate stays 0% after 20+ trades, re-add `MIN_ENTRY_PRICE_CENTS=20`.

- **Per-side position cap (commit `bcd3967`).** `MAX_POSITIONS_PER_TICKER_PER_SIDE=2`.
  `ticker_direction_count(ticker, direction)` is Redis-backed in `PortfolioMonitor`.

- **`floor_strike` as primary strike source (commit `ee2bc31`).** `> 0` guard rejects
  unset markets. Previously `floor_strike=0` made Kronos compute P(BTC > $0) ≈ 100%.

- **LKG feature fallback.** `regime:features:lkg` (TTL=24h). `_lkg=True` in context
  marks row as stale — excluded from training.

- **CVD oscillation documented.** CVD swung ±0.7 within 2h today. Kronos fires on
  price momentum only — doesn't read CVD. Gate 7 (CVD soft gate) is the immediate fix.

---

## What Failed / Avoided

- **Blanket `MAX_POSITIONS_PER_TICKER=3`.** Replaced by per-side cap.
- **20¢ entry price floor (added then removed).** Sub-20¢ data too thin (6 trades).
- **In-memory position count.** Broke under multiple processes. Always Redis-backed.
- **`floor_strike=0` accepted as valid.** Made Kronos compute P(BTC > $0) ≈ 100%.
- **Circuit breaker in paper mode.** Tripped at -$200, halting data collection.
- **Backfilling pre-instrumentation trades.** Rejected — funding/OI/CVD not reconstructable.
- **TTL == refresh interval.** Fixed via TTL=600s + overlapping 240s refresh.

---

## Post-Training Roadmap (no code yet)

These require the regime model to be trained and validated before implementation is useful.

### #10 — Per-feature-group accuracy tracking (meta-learning)
After ~200 post-training live trades, extend `edge_tracker.py` to bucket accuracy by
regime type (trending / ranging / high-vol). Track which of the 21 features are most
predictive per regime. XGBoost feature importance gives a first pass; dynamic per-regime
multipliers on Kelly or gate thresholds are the upgrade. Do not build before the model
has been live for at least 200 resolved trades — the signal will be too noisy.

### #12 — Slippage gate for 15min markets
Gate 6 is currently skipped for `timeframe == "15min"`. KXBTC15M bid-ask spread is
sometimes wide mid-cycle and can eliminate edge. Before setting a threshold, accumulate
spread observations across many cycles post-go-live. Revisit when you have 200+ spread
samples. Do not remove the Gate 6 guard in the meantime.

---

## What We Deferred From SLY Bot Analysis

### Skipped entirely (implement later, separate session):
- **Kalshi intra-cycle YES momentum** (SLY Tier 1 #2): Track YES price delta in first
  2 min of market appearance as smart-money signal. Needs new intra-cycle polling
  infrastructure that snapshots entry price when market first appears.

- **Large print direction** (SLY Tier 2 #6): ✅ **COMPLETE** (2026-05-23). Implemented as `large_print_direction` — the 21st regime feature. Net directional score from prints > 2× session average size. Source: `derivatives_feed.py` fetch_trades. Stored in `regime:features` Redis key and persisted to `trades.db`.

### Deferred to post-training (Tier 3):
- **Per-feature-group accuracy tracking / meta-learning** (SLY Tier 3 #10):
  Dynamic per-regime multipliers on feature groups. `edge_tracker.py` exists but
  needs regime model trained and ~200 post-training trades to calibrate.

- **Dynamic Kelly shrinks** (SLY Tier 3 #11): ✅ **COMPLETE** (2026-05-23). Three multiplicative shrinks — chop (×0.70 when `abs(range_breakout_flag) < 0.15`), dead tape (×0.80 when `tape_speed_tpm < 0.20`), loss streak (×0.60 when `loss_streak ≥ 3`). See Dynamic Kelly section above.

- **Slippage gate enhancement** (SLY Tier 3 #12): KXBTC15M bid-ask spread sometimes
  wide mid-cycle. Gate 6 currently skipped for 15min timeframe. Needs spread data
  across more cycles to set a defensible threshold.

---

## Files Touched / Created

### This session (2026-05-23, commit `bd80bc0`)
| File | Change |
|------|--------|
| `config.py` | Added `CVD_GATE_THRESHOLD = 0.3` |
| `btc_kalshi_system/execution/pretrade_checklist.py` | Added Gate 7 (CVD soft gate) |
| `btc_kalshi_system/data/derivatives_feed.py` | CVD ring buffer writes (`regime:cvd_history` sorted set, keep last 90) |
| `btc_kalshi_system/signal/fusion.py` | `_regime_features()` expanded to 20 features; `update_kalshi_mid()` method added |
| `btc_kalshi_system/models/regime_model.py` | `_FEATURE_ORDER` → 20 features |
| `main.py` | 14 new schema columns + `trade_snapshots` table, `kalshi_mid_cents` into context, PositionMonitor wired into asyncio.gather() |
| `scripts/train_regime.py` | `_FEATURE_COLS` → 20 features, `_FEATURE_COLS_LEGACY` for 6-feature fallback, `--legacy` flag |
| `btc_kalshi_system/execution/position_monitor.py` | **NEW.** Mid-trade exit coroutine |
| `tests/signal/test_feature_order.py` | **NEW.** Feature-order consistency test (3 assertions) |
| `tests/signal/test_regime_features.py` | **NEW.** 22 tests for new feature math |
| `tests/execution/test_gate7_cvd.py` | **NEW.** 8 tests for CVD soft gate |
| `tests/execution/test_position_monitor.py` | **NEW.** 7 tests for PositionMonitor + schema idempotency |

### Prior sessions (2026-05-23 earlier)
| File | Change |
|------|--------|
| `main.py` | `floor_strike` primary strike, per-side cap, circuit breaker paper mode fix |
| `btc_kalshi_system/portfolio/monitor.py` | `ticker_direction_count` Redis-backed |
| `btc_kalshi_system/portfolio/circuit_breaker.py` | Drawdown check gated on live mode |
| `scripts/monitor_trades.py` | **NEW.** Live SQLite polling monitor |
| `scripts/win_rate_by_price.py` | **NEW.** Win rate / P&L by price bucket |

---

## Next Steps

1. **Monitor 20-feature row accumulation.** Run `python3 scripts/regime_health_check.py`
   daily (or query directly):
   ```sql
   SELECT COUNT(*) FROM trades
   WHERE cvd_velocity IS NOT NULL AND brti_momentum_5min IS NOT NULL
     AND kalshi_implied_prob IS NOT NULL AND outcome IS NOT NULL;
   ```
   Need 500. Target: ~June 2.

2. **Train calibrator when total trades ≥ 500.** (~May 26–27 at current rate.)
   Update calibration in config or script — exact steps TBD based on calibrator implementation.

3. **Train the regime model when 20-feature rows ≥ 500.**
   `python3 scripts/train_regime.py --dry-run` — check Brier < 0.25, Kronos agreement > 55%.
   If sane, re-run without `--dry-run` → saves `models/regime.pkl`.
   Restart main.py → model auto-loads. Gate 2 runs in shadow mode by default.
   Flip `REGIME_GATE2_ENFORCING=true` after ~50 shadow trades.

4. **Early experimentation option:** If you want to train on the existing 351 6-feature rows:
   `python3 scripts/train_regime.py --legacy` — uses only original 6 features.
   Not recommended for deployment but useful for sanity-checking the pipeline.

5. **After regime model is live: implement deferred SLY features.**
   Priority order: (a) Kalshi intra-cycle YES momentum (new polling), (b) large print
   direction (after verifying per-trade volume flows through CVD — see memory flag),
   (c) dynamic Kelly with position scoring, (d) slippage gate for 15min markets.

---

## Context / Gotchas

- **Test suite invariant: 257 pass.** Run from project root: `python3 -m pytest`.

- **Feature order is a 3-file contract.** `_FEATURE_ORDER` in `regime_model.py`,
  `_FEATURE_COLS` in `train_regime.py`, and returned dict keys from `fusion._regime_features()`
  must be identical in the same order. There is a test for this: `python3 -m pytest tests/ -k "feature_order"`.

- **Existing 351 rows use 6 features.** `--legacy` flag in `train_regime.py` trains on
  them using only the original 6 features. Default training requires all new columns NOT NULL.

- **CVD ring buffer cold start.** If `regime:cvd_history` has fewer than 5 entries,
  `cvd_velocity` and `cvd_acceleration` = 0.0 and `features_stale=True`. Rows with
  stale=True are excluded from training. After restart, expect ~20 min before the buffer
  warms up (5 derivatives feed cycles).

- **PositionMonitor exit never calls `add_position()`.** It calls `remove_position()`
  first, then submits a raw API offsetting order. The offsetting order (e.g. buying NO
  to close a YES position) must NOT increment the NO side counter. This is by design.
  Do not "fix" this — it would break `MAX_POSITIONS_PER_TICKER_PER_SIDE`.

- **`kalshi_mid_cents` must be set in `market_context` before `get_signal()`.** The
  `_regime_features()` method reads it from `self._market_context`. If missing,
  `kalshi_implied_prob = 0.5` and `stale=True`. Set it in main.py after each orderbook fetch.

- **Kronos is blocking (2–3s on CPU).** Always offload via `loop.run_in_executor(None, ...)`.
  Never call on the event loop thread. Preload before asyncio starts (Apple Silicon segfault
  avoidance — do not refactor).

- **Check for stale processes before restarting.** `ps aux | grep "[Pp]ython.*main\.py"`.
  Kill all but newest. `pgrep -af main.py` gives false positives — use `ps aux`.

- **Per-side cap is Redis-backed.** `ticker_direction_count` reads `portfolio:open_positions`
  hash directly. Do not revert to in-memory count — breaks under multiple processes.

- **Label semantics.** `y_up = int(direction == outcome)`. This is "did market close UP",
  not "did trade win". For NO→DOWN wins: `direction=0, outcome=1 → label=0`.

- **LKG sentinel.** `_lkg=True` in market context dict. Never add `_lkg` or
  `_lkg_written_at` to the feature list — corrupts XGBoost inputs.

- **Gate 2 starts in SHADOW mode.** `REGIME_GATE2_ENFORCING=false` by default after
  loading a model. Only flip to `true` after observing ~50 shadow trades.

- **Gate 6 skipped for `timeframe == "15min"`.** Do not remove this guard.

- **Two `brti_volatility_1h` implementations exist.** `DerivativesFeed` (Redis ticks)
  vs `fusion._regime_features()` (5-min OHLCV pct_change). Persisted column is the
  fusion version. Do not consolidate after model training begins.

- **`dump.rdb` and `trades.db.bak.*` must NOT be committed.**

- **DeepSeek `NEUTRAL_DEFAULT` on 402, not `SAFE_DEFAULT`.**

- **`RANGING_SHRINK=0.7`, `_BOOTSTRAP_SHRINK=0.8`, `_UNCERTAINTY_SHRINK=0.5`** — do not equate.
