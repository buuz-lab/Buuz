# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min up/down markets). Forecast direction via Kronos + XGBoost regime classifier + DeepSeek gate, size with fractional Kelly, run 7 pre-trade gates.

**Current focus:** Monitor calibrator auto-retrain and trending_down shrink impact. Next milestone: deribit_stale=0 rows hit 500 → regime model retrain. Key active items: (1) calibrator now auto-retrains every 2h via cron; (2) trending_down gets 50% shrink (same as high_uncertainty) — empirical data showed 76.9% avg k15 confidence with 28.6% win rate (inverted signal); (3) regime model ETA ~5 days (230/500 deribit-clean rows as of 2026-05-30).

---

## Current Progress

**As of 2026-05-30 session 19: Calibrator system overhaul. trending_down shrink added. Calibrator retrained on k15 data (Brier 0.2708→0.2463). Auto-retrain cron wired. 390 tests pass.**

**Session 19: calibrator improvements + auto-retrain**

**Key findings driving changes:**
- k5 0.80+ bucket: avg calibrated prob 92%, actual win rate 52.4% — severe overconfidence (historical, from isotonic/passthrough era)
- k15 0.80+ bucket: avg k15 93%, actual win rate 39.4% — inverted signal at high confidence
- trending_down regime: k15 averaging 76.9% confidence, 28.6% win rate — k15 is an inverted signal in this regime
- k5 and k15 never disagree in direction (k15 only evaluated on trades that already passed k5 gates — structural, not informational)

**Changes — session 19:**

| File | Change |
|------|--------|
| `btc_kalshi_system/models/calibrator.py` | `_MIN_SAMPLES` lowered 300→100 — allows calibrator to fit on 129 available k15 rows (was blocking activation) |
| `btc_kalshi_system/signal/fusion.py` | `_TRENDING_DOWN_SHRINK = 0.5` added; `elif deepseek_regime == "trending_down": combined = 0.5 + (combined - 0.5) * _TRENDING_DOWN_SHRINK` — 50% shrink (same as high_uncertainty). Empirical basis: 76.9% avg k15 → 28.6% win rate is an inverted signal, not noise. |
| `tests/models/test_calibrator.py` | Updated 3 tests that hardcoded 300-sample threshold to use 100 (direct consequence of _MIN_SAMPLES change) |
| `scripts/auto_retrain_calibrator.py` | **New** — cron-driven calibrator auto-retrain. Three triggers: emergency (Brier > 0.25 on last 50 k15 rows), row-based (+50 k15 rows), time-based (7 days). Marker at `models/calibrator_last_trained.json`. |
| `models/calibrator.pkl` | Retrained on 129 k15 rows; Brier improved 0.2708→0.2463; high-confidence squashing now reflects k15 empirical inversion (raw=0.90→cal=0.43) |
| crontab | Added `0 */2 * * *` entry for `auto_retrain_calibrator.py` |

**Calibrator transform curve (current):**
```
raw=0.50 → cal=0.517
raw=0.60 → cal=0.548
raw=0.70 → cal=0.577
raw=0.80 → cal=0.456   ← inversion begins
raw=0.90 → cal=0.433   ← high confidence = below-50% output
raw=0.95 → cal=0.441
```
The inversion above ~0.78 reflects that k15 high-confidence outputs are empirically anti-correlated with wins. Combined with the trending_down shrink, very high-confidence trades in trending_down will rarely pass Gate 5.

**Watch for:** calibrator Brier on next auto-retrain runs — if Brier stays above 0.25 the emergency trigger will fire every 2h. Monitor `logs/auto_retrain_calibrator.log`. Consider raising `_EMERGENCY_BRIER_THRESHOLD` to 0.26 if it causes churn without improvement.

**Commits:** `bc583b5` (calibrator changes), `97fa222` (auto_retrain_calibrator.py)

---

**As of 2026-05-29 session 18: Deep signal analysis. Regime-aware Gate 5 deployed (ranging requires 15% edge, high_uncertainty requires 8%). Confirmed regime model cannot be trained yet (215 clean rows vs 500 needed). 394+ tests pass.**

**Session 18: signal analysis + regime-aware Gate 5**

**Key findings:**

| Finding | Detail |
|---------|--------|
| k15_raw is anti-correlated with P(up) | Spearman = −0.12. k15_raw > 0.8 → P(up) = 38% (n=52). Signal is partially inverted at high conviction. |
| Regime model has never been trained | `regime_model.pkl` does not exist. 0/630 trades have active regime prob. The 0.4 regime weight in fusion is dead weight. |
| All regimes losing money | ranging −$0.10/trade (n=474), high_uncertainty −$0.84/trade (n=119), trending_up −$1.63/trade (n=17), trending_down −$0.43/trade (n=20) |
| Ranging dead zone | calibrated_prob 0.55–0.65 → 23% win rate (36 trades). Above 0.65 → 60% win rate (126 trades). |
| All deribit-clean data is 5 days old | deribit_stale=0 rows only since 2026-05-25. 215 total vs 500 needed for regime model. ~5 weeks away at 60 clean trades/day. |
| Do not restart from 145 "cleanest" rows | Dirty vs clean win rate nearly identical (48.9% vs 47.9%). Outcome labels and kronos_raw are valid across all rows. Restarting would push calibrator activation 2+ weeks further. |
| S/R proximity signal is flat | hourly_sr_proximity win rates 45–52% across all buckets. 24h high/low is a range-position indicator, not true S/R. Not actionable. |

**Changes — session 18:**

| File | Change |
|------|--------|
| `btc_kalshi_system/execution/pretrade_checklist.py` | **Gate 5 regime-aware**: ranging → `min_required = max(base_min, 0.15)`; high_uncertainty → `max(base_min, 0.08)`. Targets the 0.55–0.65 cal_prob dead zone in ranging (23% win rate). Failure message now includes regime. Tests added for all four branches. |

**Regime model status:** Cannot train until ≥500 clean rows (features_stale=0 AND deribit_stale=0). Currently 215. ETA ~5 weeks. Do NOT attempt to train early — underfitting at 215 rows produces confidently wrong regime labels and will make Gate 2 harmful when enforced.

**Calibrator status:** k15_raw count = 118. Needs 300 to activate. At ~40 k15 trades/day → activates in ~4–5 days. Run `python3 scripts/train_calibrator.py` when count hits 300.

---

**As of 2026-05-29 session 17: Calibrator replaced: IsotonicRegression → quadratic LogisticRegression on [raw, raw²]. train_calibrator.py now trains on kronos_raw_15min. 394 tests pass.**

**Session 17: quadratic logistic calibrator**

**Root cause:** `IsotonicRegression(increasing=True)` cannot learn the inverted-U relationship in k15_raw: Spearman correlation of −0.12, k15_raw > 0.8 → P(up) = 0.38. Monotone-increasing constraint forced the calibrator to assign high calibrated values to high raw inputs even when they predicted DOWN.

**Changes:**

| File | Change |
|------|--------|
| `btc_kalshi_system/models/calibrator.py` | Replace `IsotonicRegression` with `LogisticRegression` fitting on `[raw, raw²]` features. Rename internal `_iso` → `_model`. `transform()` calls `predict_proba` and clips to `[0,1]`. `save()`/`load()` use `"model"` key; `load()` falls back to `"iso"` for backward compatibility with existing `.joblib` files. |
| `scripts/train_calibrator.py` | Query now selects `kronos_raw_15min` with `AND kronos_raw_15min IS NOT NULL`. Prints warning if fewer than 200 rows have `kronos_raw_15min` (data is sparse). |
| `tests/models/test_calibrator.py` | Updated `test_monotonicity_guard_reverts_worse_fit` to use `_model` instead of `_iso`. Added `test_calibrator_uses_logistic_regression`, `test_inverted_signal_calibrated_below_half`, `test_passthrough_still_works_below_min_samples_logistic`. |

**Why LogisticRegression on [raw, raw²]:** quadratic features let the model fit a parabola in calibrated probability — capturing both monotone and inverted-U shapes without a monotonicity constraint. LogisticRegression's L2 regularization prevents overfitting on the ~300-sample window.

**Backward compatibility:** `Calibrator.load()` falls back to reading the `"iso"` key if `"model"` is absent. Any existing `calibrator.pkl` saved with the old isotonic code will load and behave as passthrough (IsotonicRegression object stored in `_model`, which does not have `predict_proba`). Re-run `python3 scripts/train_calibrator.py` after deploying to replace the file with a logistic-trained model.

**Commits:** session 17

---

**As of 2026-05-28 session 16: Gate 10 (DeepSeek trend conflict) added. Gate 8 made confidence-aware. Monitor P&L formula fixed. Main loop converted from fixed 300s timer to event-driven on BG loop completion. 382 tests pass.**

**Session 16: accuracy investigation + timing fix**

**Root causes identified (2026-05-28, 32% win rate day):**
1. `trending_down + YES` trades firing repeatedly (5 consecutive losses at 28–34¢) — no gate blocked directional conflict with DeepSeek regime.
2. Gate 8 threshold widened to 0.25 in session 15 to protect one high-confidence case (k15=0.89 at 29¢). Side effect: YES bets at 28–34¢ with moderate k15 confidence no longer blocked (opposing=0.22 < 0.25).
3. Monitor P&L formula wrong for NO direction trades: treated `fill_price_cents` as the YES price, swapping win/loss payout ratios. Showed −$2149; true P&L is −$171 (Kalshi shows −$145; $26 gap = fees).
4. Main loop ran on a fixed 300s wall-clock timer, independent of when BG loop finished MC. ~7.7% of cycles fired while BG loop was mid-computation (23s window), reading the previous candle's k15 instead of the freshest one.

**Changes:**

| File | Change |
|------|--------|
| `btc_kalshi_system/execution/pretrade_checklist.py` | **Gate 10** (new): `trending_down + YES` or `trending_up + NO` → hard block (failed_gate=10). **Gate 8 confidence-aware threshold**: `signal_confidence = abs(calibrated_prob - 0.5)`; ≥0.30 → threshold 0.25, ≥0.15 → 0.15, <0.15 → 0.10. OI-squeeze compound still applies (÷4). |
| `main.py` | `_cache_updated_event = asyncio.Event()` added to `__init__`. BG loop calls `self._cache_updated_event.set()` after writing cache. `_main_loop` now `await self._cache_updated_event.wait()` + `.clear()` instead of fixed 300s sleep. |
| `scripts/monitor.sh` | P&L formula fixed: `kelly_contracts*(100-fill_price_cents)/100` on wins, `-kelly_contracts*fill_price_cents/100` on losses. Direction-independent (fill_price_cents is always the price paid). |
| `tests/execution/test_pretrade_checklist.py` | 9 new tests: 4 confidence-aware Gate 8 scenarios, 5 Gate 10 scenarios. |
| `tests/execution/test_gate7_cvd.py` | Fixed `_make_signal` default `deepseek_regime` from `"trending_up"` → `"ranging"` (Gate 10 correctly fires on trending_up+NO). |
| `tests/test_main_bg_kronos.py` | 2 new tests: BG loop sets event after MC, main loop fires on event not timer. |

**Gate 8 confidence-aware threshold detail:**

| k15_cal range | distance from 0.5 | Gate 8 threshold |
|---|---|---|
| ≥0.80 or ≤0.20 | ≥0.30 | **0.25** (session 15 regression case preserved) |
| 0.65–0.79 or 0.21–0.35 | 0.15–0.29 | **0.15** |
| 0.35–0.65 | <0.15 | **0.10** |

Uses `signal.calibrated_prob` (the k15-calibrated, bootstrap-shrunk combined prob) as the confidence proxy.

**Gate 10 retroactive impact on 2026-05-28:**
- Would have blocked 5 trades (trending_down + YES): 1 WIN, 4 LOSS → net positive
- Gate 8 confidence tiers would have blocked 4 additional losses at 28–38¢ fills

**Event-driven main loop:**
- Old: `asyncio.sleep(max(0, SIGNAL_INTERVAL_SECONDS - elapsed))` — fires on 300s wall clock
- New: `await self._cache_updated_event.wait(); self._cache_updated_event.clear()` — fires only after BG loop posts fresh MC result
- Cadence unchanged (~5 min, one per 5-min candle close). Race window eliminated: `_run_cycle` can never fire while BG loop is mid-computation.

**P&L accounting (correct formula):**
```sql
SUM(CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
         WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0 END)
```
`fill_price_cents` is always the price paid (NO fill for NO trades, YES fill for YES trades). The formula is direction-independent. True all-time P&L as of session 16: **−$171** (vs Kalshi −$145; gap = fees).

**Commits:** session 16

---

**As of 2026-05-27 session 15: Gate 8 threshold widened 0.08→0.25. Gate 8b denominator /0.20→/0.30. Bootstrap floor bug fixed. Depth check capped to available (no longer hard-fails). Terminal monitor script added. 371 tests pass.**

**Session 15: Gate 8 + Gate 8b recalibration + depth cap**

**Root cause:** k15=0.89 at 29¢ YES was blocked by `"Kelly size rounds to 0 contracts after Kalshi Kelly multiplier"`. Two compounding bugs:
1. Gate 8 threshold (0.08) was tuned when k15 was flat at 0.558. With isotonic passthrough (k15_cal ≈ k15_raw), high-confidence k15 calls are reliable through moderate Kalshi disagreement. 0.08 = "Kalshi must only be 58% to block us" — too sensitive for a well-calibrated signal.
2. Gate 8b denominator `/0.20` zeroed Kelly at opposing≥0.20, hitting the Gate 2 rounding-to-0 path before Gate 8 could fire. Bootstrap floor check then failed because `is_bootstrap and kelly_dollars > 0` evaluated on the post-multiplier (zeroed) value.

**Changes:**

| File | Change |
|------|--------|
| `config.py` | `KALSHI_CONSENSUS_THRESHOLD` 0.08 → 0.25 (Kalshi must price ≥75% against us to hard-block) |
| `btc_kalshi_system/execution/pretrade_checklist.py` | Gate 8b denominator `/0.20` → `/0.30`; save `_pre_mult_kelly_dollars` before multiplier for bootstrap floor; depth check caps to `available_contracts` instead of hard-failing |
| `tests/execution/test_pretrade_checklist.py` | 6 existing tests updated for new thresholds; 3 new tests: `test_high_confidence_k15_passes_when_kalshi_disagrees_moderately`, `test_gate2_depth_capped_to_available`, `test_gate2_zero_depth_still_fails` |

**Gate 8 validation (live results after fix):**
- 02:04 YES→UP 29¢: WIN ✓ (the exact blocked case, now passes)
- 02:09 YES→UP 51¢: WIN ✓
- 02:19 YES→UP: LOSS ✗ — but CVD=-0.81, Gate 7 shadow would have blocked correctly (CVD now 4/4 when opposing)

**Depth cap behavior change:** Previously, if Kelly wanted N contracts and orderbook had M < N, Gate 2 hard-failed (`"Insufficient depth: need N, M available"`). Now: caps `kelly_contracts = available_contracts`, recalculates `kelly_dollars`, and proceeds. `available_contracts == 0` still hard-fails.

**Bootstrap floor bug fixed:** Gate 8b multiplier can zero `kelly_dollars`. The floor check `is_bootstrap and kelly_dollars > 0` must use the pre-multiplier value. Fixed by saving `_pre_mult_kelly_dollars = kelly_dollars` before multiplier application.

**k15 calibration status:** ~171 resolved samples, isotonic passthrough active. `k15_cal ≈ k15_raw`. k15 is the effective signal for direction and sizing. k5 stored for analysis only.

**Gate 7 CVD status:** 4/4 correct when opposing. 02:19 LOSS (CVD=-0.81) is the clearest example. Still shadow mode — not activated as hard gate yet.

**Terminal monitor script added:** `scripts/monitor.sh` — color-coded bash script, refreshes every 30s. Sections: BG Loop, Gate Rejections, Trades, P&L, Regime snapshot (CVD/LP/funding/fear_greed), Last Activity. Run with: `bash "/Users/ezrakornberg/Kronos V2/scripts/monitor.sh"`

**Commits:** `ac47663` (Gate 8/8b + bootstrap fix), `9288e2e` (depth cap)

---

**As of 2026-05-27 session 14: k15_calibrated_prob + candle_progress logging added. Deribit _MIN_DAYS_TO_EXPIRY lowered 3→1. Two pre-existing test bugs fixed. 368 tests pass.**

**Session 14: k15 vs k5 timing analysis instrumentation**

Added two new columns to `gate_rejections` and one to `trades` to enable a proper k5/k15 timing comparison in the future.

**Why:** Current k15 data (109 resolved rows) shows k15 outperforming k5 overall (54.1% vs 40.4% directional accuracy) and strongly in the divergence case (k5YES/k15NO → k15 right 67.5%). But the data doesn't yet support the k5/k15 timing design (k5 at t=0, k15 at t+5) because:
- Only 5 rows at t=0 (the entire design premise has 5 data points)
- 89% of data is high_uncertainty + ranging; zero trending_up rows
- All rows are from gate_rejections (Kelly→0 blocked trades) — a different population than what would fire under k15-primary
- `k15_calibrated_prob` cannot be reconstructed post-hoc (calibrator state changes every 25 resolutions)

**New columns:**

| Column | Table(s) | Purpose |
|--------|----------|---------|
| `k15_calibrated_prob` | `gate_rejections`, `trades` | calibrator.transform(kronos_raw_15min) at signal time. The only moment this can be captured. Enables full k15-primary counterfactual: direction, edge, kelly. |
| `candle_progress` | `gate_rejections` | Denormalized from features JSON for direct timing-bucket queries without JSON parsing. |

**Key query for future analysis:**
```sql
SELECT
  CASE WHEN candle_progress < 0.15 THEN 't=0'
       WHEN candle_progress < 0.55 THEN 't+5'
       ELSE 't+10' END as timing,
  CASE WHEN signal_prob >= 0.5 AND k15_calibrated_prob >= 0.5 THEN 'agree-YES'
       WHEN signal_prob < 0.5  AND k15_calibrated_prob < 0.5  THEN 'agree-NO'
       WHEN signal_prob >= 0.5 AND k15_calibrated_prob < 0.5  THEN 'k5YES-k15NO'
       ELSE 'k5NO-k15YES' END as signal,
  COUNT(*) as n,
  ROUND(100.0*AVG(outcome),1) as win_pct
FROM gate_rejections
WHERE k15_calibrated_prob IS NOT NULL AND outcome IS NOT NULL AND aged_out=0
GROUP BY timing, signal ORDER BY timing, n DESC;
```

**Also: Deribit _MIN_DAYS_TO_EXPIRY lowered 3→1** — 3 days was filtering out valid weekly options near the roll, causing unnecessary `deribit_stale=1` rows during those periods. 1 day only skips same-day expiry (where theta spikes are actually problematic).

**Pre-existing test bugs fixed (both from session 13):**
- `test_does_not_rerun_mc_on_same_candle`: expected 1 MC call per candle, should be 2 (k5 + k15)
- `test_fill_price_from_second_fetch`: `capture_record` mock missing `**kwargs` for `kronos_raw_15min`

**Files changed:**

| File | Change |
|------|--------|
| `main.py` | `_GATE_REJECTIONS_COLUMN_MIGRATIONS`: added `k15_calibrated_prob`, `candle_progress`; `_TRADES_COLUMN_MIGRATIONS`: added `k15_calibrated_prob`; `_process_market`: compute `_k15_cal = calibrator.transform(k15_raw)` and `_candle_prog` before all 3 gate_rejections INSERTs; `_record_trade_sqlite`: new `k15_calibrated_prob` param + column |
| `btc_kalshi_system/data/deribit_options_feed.py` | `_MIN_DAYS_TO_EXPIRY` 3→1 |
| `tests/execution/test_gate_rejections.py` | `_GATE_REJECTIONS_DDL` updated to include all migration columns |
| `tests/test_main_bg_kronos.py` | MC call count 1→2; `capture_record` gains `**kwargs` |

**Commit:** `e0be718`

---

**As of 2026-05-26 session 12 (post-deploy): Edge-flip shadow mode live. Gate 2a price floor added. KronosEngine candle_freq param added. 368 tests pass.**

**Session 12: Edge-flip direction shadow mode (Gate 9)**

Discovered that the fusion sets `direction = 1 if combined_prob >= 0.5 else 0` without considering market price. When Kronos says 55.8% YES but the market prices YES at 72–76¢, the edge for YES is negative (Kelly→0, blocked). The positive edge is actually on the NO side (NO at 24–28¢, win_prob=44.2%, edge=+16–20%). The system was never evaluating this.

Historical data confirmed: 50–60% Kronos prob bucket had only 31.6% win rate (19 trades) — below 50% — consistent with the system fighting markets that have already priced in the move.

Fix (shadow mode, not live): After any Gate 2 "Kelly rounds to 0" failure, compute whether the flipped direction has positive edge AND meets the ≥20¢ price floor. If so, insert a `gate_rejections` row with:
- `direction = flipped_dir` (opposite of fusion signal)
- `failed_gate = 9` (edge-flip shadow gate)
- `shadow = 2` (excluded from training; distinct from shadow=1 Gate 7 rows)
- `failed_reason` records original direction prob vs price, flipped direction edge

Resolution logic unchanged — outcomes tracked same as other gate_rejections rows. Once win rate is confirmed positive over ~50+ resolved shadow rows, the fix goes live by adding market-price-aware direction selection before the checklist.

**Session 12: Gate 2a minimum price floor (20¢)**

Depth failures at 2–9¢ markets: Kelly requested 100–400 contracts (even a $8 bet at 2¢ = 400 contracts). Historical data: 0W/10L at ≤18¢ fill price. Added `_MIN_TRADE_PRICE_CENTS = 20` check in `pretrade_checklist.py` before Kelly runs. Commit `072e087`.

**Session 12: KronosEngine configurable candle_freq**

Prediction horizon mismatch: `run_monte_carlo()` used 5-min candles and predicted `y_timestamp = last_candle + 5min`, but 15-min Kalshi markets settle at the 15-min BRTI close. Added `candle_freq: str = "5min"` parameter (default preserves existing behavior). Background loop and 1h track can pass correct freq when ready. Commit `d36b56a`.

**k15 shadow column (session 13)**

`kronos_raw_15min` column added to `trades` and `gate_rejections`. Background loop computes both 5min and 15min MC on every 5min candle close; `prob_15min` cached alongside `prob`. All trade and rejection INSERTs log both. Enables A/B calibration comparison over time.

**Caveat if switching k15 to primary signal:** The background loop triggers on 5min candle closes and `_resample()` always appends the live in-progress candle, so the 15min OHLCV input refreshes every 5 minutes (not just on 15min closes) — no structural staleness problem. However, the hard-abort threshold in `_process_market` is `_cache_age > 600` (10 min). If the background loop ever misses a 5min trigger, a 15min-primary system could hit this threshold mid-candle. Raise it to `900` (one full 15min period) before going live with k15 as primary. The 360s watchdog in `_regime_watchdog` should also be raised to `~600s` accordingly.

**k15/k5 signal architecture — tentative design (session 13)**

**Core design: k15 picks direction, k5 sizes the bet (and vice versa at t=0)**

| Time in 15-min window | Direction signal | Size modifier |
|---|---|---|
| t=0 (progress < 0.15) | k5 primary | full Kelly if k15 agrees; half Kelly if k15 disagrees |
| t+5 (0.15–0.55) | k15 primary | full Kelly if k5 agrees; half Kelly if k5 disagrees |
| t+10 (progress > 0.55) | block or very tight edge only | candle_progress data: 27% win rate at 0.7 |

**Why this design:**
- k15 at t=0 is one 15-min bar stale (background loop hasn't finished for the new candle yet); k5 is fresher
- k15 at t+5 has just updated with the bar that closed at market open — most informative k15 gets
- Agreement amplifies (full Kelly); disagreement doesn't block the trade, it just reduces conviction (half Kelly)
- No momentum missed: disagreement costs size, not the entry

**Implementation notes:**
- Direction change at t+5: replace `cached["prob"]` with `cached["prob_15min"]` for signal direction computation in `_process_market`, gated on `candle_progress >= 0.15`
- Size modifier: add an `agreement_mult` (1.0 or 0.5) to the Kelly shrink chain in `pretrade_checklist.py`
- `candle_progress` is already computed in fusion.py and available in `signal.regime_features`
- Gate rejections log `candle_progress` via `json_extract(features, '$.candle_progress')` — no schema change needed for analysis

---

**Uptrend cases and how to adjust**

Four scenarios when BTC enters an uptrend:

**Case 1 — k15 bullish, k5 bullish (agree)**
Both signals see the trend. Full Kelly at both t=0 and t+5. This is the ideal case — no adjustment needed. Expect this after the first full 15-min bar of the uptrend has closed and k15 has updated.

**Case 2 — k15 bullish, k5 bearish (k5 sees an intra-bar pullback)**
This is the design working correctly. k15 leads the trend direction; k5 is reacting to short-term noise within the uptrend. Enter at half Kelly. If k15 is right (trend holds), momentum is captured. Risk is limited by the smaller size.

**Case 3 — k15 bearish, k5 bullish at t+5 (the dangerous case)**
k5 is catching a trend reversal but k15 is still bearish from the bar that closed before the reversal started. Under the design, k15 controls direction at t+5 → we either skip or enter the wrong way (NO). This is where the design can fail.

*Mitigations:*
- At t=0 in the same market period, k5 is primary → you still enter YES at half Kelly (k15 disagreeing reduces size but doesn't block). This captures the early momentum even if t+5 misfires.
- After one 15-min bar of uptrend, k15 will have flipped bullish and the t+5 signal corrects itself. The failure window is at most ONE market period (15 min).
- DeepSeek override: if DeepSeek classifies `trending_up` AND k5 is strongly bullish (≥ 0.80), treat t+5 as k5-primary instead of k15-primary. This is the cleanest adjustment and uses an existing signal.
- If k5 is very strongly bullish (≥ 0.85) while k15 is bearish at t+5, consider 75% Kelly instead of 50% rather than full reversal — softer than overriding k15 entirely.

**Case 4 — Both bearish, trend reversal mid-period**
Both signals are wrong. This is a regime-change problem independent of the k5/k15 design. The existing circuit breaker and direction win-rate tracker are the right mitigations here.

**Case 5 — Early trend, k5 has fired at t=0 but k15 hasn't updated yet**
k5 enters at t=0 (half Kelly, k15 still bearish). By t+5, k15 has updated with the first bullish 15-min bar → k15 now bullish → t+5 entry at full or half Kelly depending on k5. The system naturally scales up as the trend becomes clear. This is the expected happy path for catching early momentum.

**Monitoring signals that precede k15 catching up:**
When these flip, the transition from Case 3 to Case 1 is imminent:
- `large_print_direction` turns strongly positive (> 0.6)
- `cvd_normalized` turns positive and sustained
- `volume_ratio_1h` climbing above 0.8x
- `funding_rate_trend` turning positive
- k5 consistently ≥ 0.70 for 2+ consecutive cycles
- DeepSeek flips to `trending_up`

**Key validation needed before implementing:**
The design assumes k15 will flip bullish within one 15-min bar of a real uptrend starting. This needs to be confirmed with live trending_up data — it's the only regime missing from the k15 dataset. Until then, this design should not be implemented. Use the current shadow logging to accumulate that data.

---

**k15 investigation findings + integration roadmap (session 13)**

94 resolved gate rejections with k15 data collected (all 2026-05-27, thin-volume Extreme Fear overnight). Key results:

| | k5 | k15 |
|---|---|---|
| Directional accuracy (94 rows) | 53.2% | 66.0% |
| Accuracy when they agree (66 rows) | 54.5% | 54.5% |
| Accuracy when they diverge (36 rows) | **30.6%** | **69.4%** |

In all 36 divergence cases: k5 was bullish, k15 was bearish. k15 was right 25/36 = 69.4%. k5 is near-coinflip when they agree; k5 is a strong contra-indicator when they diverge.

**Critical data gap:** zero k15 rows in `trending_up` regime (36 historical rows exist, none with k15). This is the regime where k15 being overly conservative would hurt — must see this before any gate goes live.

**Session 13 schema addition:** `kronos_raw` added to `gate_rejections`. Now stores raw k5 MC (pre-calibration) alongside `kronos_raw_15min` for apples-to-apples comparison. `signal_prob` (calibrated) was the old k5 proxy — use `kronos_raw` for all future k5 vs k15 analysis.

---

**k15 integration options**

**Option A — k15 as hard confirmation gate (recommended)**
Require `kronos_raw_15min >= 0.5` (for YES direction) before the trade proceeds. Adds as a new gate between signal fusion and the existing checklist.
- Eliminates the 36-case divergence pattern (30% win rate)
- Reduces trade frequency ~30% (only fires when both agree)
- Doesn't require regime model retrain
- Implementation: one condition in `_process_market` before `result = self._checklist.run(...)`, logged as a new gate_rejections row with `failed_gate=10` or similar
- **Code change:** raise `_cache_age > 600` → `> 900` and watchdog `360s` → `600s` in `main.py` first

**Option B — k15 as a regime model feature**
Add `kronos_raw` and `kronos_raw_15min` to the regime model's X matrix. The model learns when their disagreement signals regime uncertainty.
- Requires ~200+ trades with k15 populated (currently 3)
- Not viable until ~late June at current trade rate
- Combine with Option A: gate first, feature later

**Option C — weighted ensemble**
Composite signal = `0.6 * kronos_raw_15min + 0.4 * kronos_raw`. Replace `cached["prob"]` with composite before calibration step in `_kronos_background_loop`.
- Softer than a hard gate — doesn't eliminate borderline divergence cases
- Harder to interpret and audit
- Not recommended as first step; could layer on top of Option A later

---

**Suggested schedule**

| Date | Action | Condition |
|------|--------|-----------|
| Now → May 30 | Accumulate k15 data across regimes | Need `trending_up` rows + 400+ total resolved k15 gate rejections |
| ~May 30–31 | Implement Option A (k15 hard gate) | 66%+ accuracy holds across regimes; trending_up data confirms k15 isn't systematically late in bullish markets |
| ~June 1–3 | Regime model retrain | 500 training-ready rows with `deribit_stale=0` (currently ~102, accumulating ~73/day) |
| Post-retrain | Add `kronos_raw` + `kronos_raw_15min` as features | Only if enough trades have k15 populated (~100+); otherwise skip and add in next retrain |

**Regime model retrain is independent of the k15 gate** — they can ship in the same deploy (~June 1–3) but neither blocks the other.

**Pre-gate-live checklist:**
- [ ] At least one `trending_up` session with k15 data — confirm k15 accuracy holds (target ≥55%)
- [ ] 400+ resolved k15 rows across ≥2 distinct regimes
- [ ] Raise `_cache_age` threshold `600 → 900` and watchdog `360 → 600` in `main.py`
- [ ] Add `failed_gate=10` path to gate_rejections with shadow=0

---

**Session 11 bootstrap floor: Gate 2 chicken-and-egg deadlock resolved**

Regime model needs trades to accumulate training data, but Kelly rounds to 0 on thin edges in bootstrap mode (stacked chop/tape/direction_win_rate shrinks), blocking trades entirely. Fixed by adding an `is_bootstrap` flag to `PreTradeChecklist.run()`. When `is_bootstrap=True` (i.e., `regime_model._clf is None`), `kelly_dollars > 0`, and `25 ≤ trade_price_cents ≤ 75`:
- Floor forces `kelly_contracts = 1` at all three rounding checkpoints (initial sizing, Gate 8b, drift shrink)
- Price range guard excludes bad risk/reward extremes (>75¢ or <25¢ prices)
- Gate 8, Gate 7, circuit breaker all still apply — only Gate 2 gets floored
- `is_bootstrap = (self._regime_model._clf is None)` set in `main.py` before both checklist calls

Commit `0e7e137`. 3 new tests added.

**Session 11 hotfix: DerivativesFeed OKX exchange stuck as None**

`_fetch_trades_data` and `_fetch_volume_ratio` both catch all exceptions internally and fall back to Kraken, so `run()` never saw a failure and never called `_resolve_exchange()` after OKX was nulled out. Fixed by adding a guard at the top of the `run()` inner loop: if `self._exchange is None`, re-resolve before fetching. Commit `46b7143`.

---

**As of 2026-05-26 session 11: Regime-adaptive trading fixes COMPLETE. 362 tests pass.**

**Session 11: Regime-adaptive trading implementation**

| File | Change |
|------|--------|
| `config.py` | Added `KALSHI_CONSENSUS_THRESHOLD = 0.08`, `CALIBRATOR_MODEL_PATH = "models/calibrator.pkl"` |
| `btc_kalshi_system/execution/pretrade_checklist.py` | Gate 8 hard block (+ OI squeeze compound at 2%), Gate 8b continuous Kelly multiplier, drift 50% Kelly shrink, `fresh_kalshi_mid`/`is_drifting`/`direction_win_rate` params, `kalshi_mid_at_block` on `ChecklistResult` |
| `btc_kalshi_system/execution/kelly.py` | Added `direction_win_rate` param; 40% shrink when rolling 30-trade win rate < 45% |
| `btc_kalshi_system/models/calibrator.py` | `_MIN_SAMPLES` 500→300; monotonicity guard in `fit()`; persist `_prev_brier` in save/load |
| `btc_kalshi_system/models/regime_model.py` | Added `btc_24h_return` to `_FEATURE_ORDER` (Feature 28) |
| `btc_kalshi_system/signal/calibration_drift_monitor.py` | Added `reset_baseline()` |
| `btc_kalshi_system/signal/fusion.py` | Added `drift_monitor` param; bootstrap shrink 0.4 when drifting; added `btc_24h_return` to `_regime_features()` |
| `btc_kalshi_system/signal/direction_win_rate_tracker.py` | **New** — per-direction rolling 30-trade win rate via Redis sorted sets |
| `main.py` | Calibrator load-on-startup; refit every 25 resolutions (rolling 300 rows, correct y_up labels, save, reset_baseline); drift monitor label fix (y_up not outcome); drift_monitor wired into fusion; DirectionWinRateTracker wired; Gate 8 wired into both checklist calls; btc_24h_return schema + record; `kalshi_mid_at_block` gate_rejections migration; `numpy`/`os` top-level imports |
| `scripts/train_regime.py` | Added `btc_24h_return` to `_FEATURE_COLS`; `_EXTRA_FILTERS_28`; `use_28` flag in `_build_query()` |
| `scripts/train_calibrator.py` | **New** — standalone calibrator training script with --dry-run, Brier comparison |
| `scripts/auto_retrain.py` | `_ROW_TRIGGER_DELTA` 500→200 |
| `.gitignore` | Added `models/calibrator.pkl` |

**Test count:** 362 (338 baseline + 24 new)

**Key new constants:**
- `KALSHI_CONSENSUS_THRESHOLD = 0.25` — Gate 8 fires when Kalshi prices ≥25% against our direction (raised from 0.08 in session 15)
- `CALIBRATOR_MODEL_PATH = "models/calibrator.pkl"` — calibrator persistence path

**Key new behavior:**
- **Gate 8** fires when Kalshi consensus ≥8% against direction; OI squeeze compounds to 2% threshold for NO→DOWN bets; blocks logged to `gate_rejections` with `failed_gate=8`
- **Calibrator** now uses `y_up = int(direction == outcome)` labels; rolling 300-row window; persists to disk; refit every 25 resolutions; monotonicity guard prevents degraded refit
- **Drift monitor** now records `y_up` not `outcome`; wired to bootstrap shrink (0.4 when drifting) in fusion; `reset_baseline()` called after each calibrator refit
- **DirectionWinRateTracker** fires 40% Kelly shrink when rolling 30-trade per-direction win rate < 45%
- **Feature 28 (btc_24h_return)**: 24h BTC price return added to 3-file contract (fusion/regime_model/train_regime)
- `auto_retrain.py` trigger now fires every 200 new rows (was 500)

**Notes:**
- `models/` directory created by first calibrator save — `models/calibrator.pkl` is in `.gitignore`
- Gate 8 blocks query: `SELECT COUNT(*), ROUND(100.0*AVG(outcome),1) as win_pct FROM gate_rejections WHERE failed_gate=8 AND outcome IS NOT NULL AND aged_out=0`
- btc_24h_return stale path: defaults 0.0 + features_stale=True when len(df1h) < 25; excluded from training via `_EXTRA_FILTERS_28`

---

**As of 2026-05-26 session 10: Regime-adaptive trading design COMPLETE. Implementation pending (see spec). 338 tests pass.**

**Session 10: Regime-adaptive trading analysis + design**

BTC dropped ~20% (May 20→76k). Market bounced (mean-reversion). Kronos stayed bearish. Win rate dropped to 37–42% on May 23–26, losing $347 on May 23 alone. Root cause analysis identified 3 compounding failures and 15 specific bugs/missing features. Full spec at `docs/superpowers/specs/2026-05-26-regime-adaptive-trading-design.md`.

**Three root causes identified:**

1. **Calibrator has never worked correctly — 4 bugs, never activated.** `_MIN_SAMPLES=500` too high (460 trades accumulated, never crosses threshold). Wrong training labels: `outcome` (trade win) used instead of `y_up = int(direction == outcome)` (market direction). For NO trades these are opposite, making isotonic regression learn contradictory signals. No persistence: calibrator resets to passthrough on every launchd restart — all in-memory fits lost. No rolling window: query fetches ALL 460 rows, blending two contradictory regimes.

2. **No Kalshi consensus gate.** When Kalshi prices UP and Kronos bets DOWN: 23.9% trade win rate on 46 trades (May 23–26). When Kalshi agrees with direction: 49–60% win rate. Zero false positives on the 248 good-day (May 20–22) trades at every threshold tested. This is the single strongest available signal — missing entirely.

3. **Drift detection fires but nothing acts on it.** `CalibrationDriftMonitor` correctly detected the May 23 regime shift (Brier 0.21 → 0.46). But `is_drifting()` is not wired to Kelly, bootstrap shrink, or trade suppression anywhere in the codebase. Additionally, the drift monitor itself has the same label bug as the calibrator.

**Additional findings:**
- `bootstrap_shrink=0.8` has no interaction with drift detection — extreme directional bets fire at full strength during known-bad regime
- Continuous Kalshi disagreement → win rate gradient: 63% win at 0¢ opposition, 12–22% win at 5–10¢ opposition → should be a continuous Kelly multiplier, not just a hard gate
- No per-direction rolling win rate tracker: drift monitor needs 60 trades to fire; a 30-trade per-direction tracker fires 2× faster
- `CalibrationDriftMonitor.record()` also has the label bug (records `outcome` not `y_up`)
- `models/` directory doesn't exist — calibrator save would fail silently even after fix
- No `train_calibrator.py` script (unlike `train_regime.py`) — no way to manually verify calibration before go-live
- `btc_24h_return` missing as a feature: all 27 features max out at 1h lookback; BTC's 24h crash context is invisible to the regime model. `brti:candles:1h` stores candles permanently (no TTL), so 6+ days of data already available.
- OI rising + NO→DOWN: 14.3% win rate on 14 trades — cleaner short-squeeze indicator than CVD alone
- Kalshi accuracy (54.9%) vs Kronos accuracy (18.4%) in losing period — market is 3× better predictor right now

**Planned changes (pending implementation):**

| Layer | Change | Files |
|-------|--------|-------|
| 1 | Fix calibrator label bug: use `y_up = int(direction==outcome)` | `main.py` |
| 1 | Add rolling 300-row window + `features_stale=0` filter to calibrator training query | `main.py` |
| 1 | Lower `_MIN_SAMPLES` 500→300 | `calibrator.py` |
| 1 | Add calibrator persistence: `CALIBRATOR_MODEL_PATH`, save after refit, load on startup | `config.py`, `calibrator.py`, `main.py` |
| 1 | Refit cadence: every 25 resolutions via Redis counter (not every trade) | `main.py` |
| 1 | Monotonicity guard: revert refit if new Brier > old Brier on training data | `calibrator.py` |
| 1 | New `scripts/train_calibrator.py` (mirrors `train_regime.py`, dry-run, Brier report) | new file |
| 1b | Fix drift monitor label bug: record `y_up` not `outcome` | `main.py` |
| 1b | Add `CalibrationDriftMonitor.reset_baseline()` — call after each calibrator refit | `calibration_drift_monitor.py` |
| 1b | Bootstrap shrink 0.8→0.4 when `is_drifting()=True` in fusion | `fusion.py` |
| 2 | Gate 8 hard block: Kalshi consensus at 8% threshold (OI squeeze compound at 2%) | `pretrade_checklist.py`, `config.py` |
| 2 | Gate 8b continuous Kelly multiplier: `max(0, 1 - opposing_margin/0.30)` (was /0.20, widened session 15) | `pretrade_checklist.py` |
| 2 | Drift monitor → Kelly: 50% shrink when `is_drifting()` | `pretrade_checklist.py` |
| 2 | Gate 8 logged to `gate_rejections` with `failed_gate=8`; fresh second-fetch mid used | `main.py`, `pretrade_checklist.py` |
| 3 | New `DirectionWinRateTracker`: 30-trade rolling per-direction win rate → 40% Kelly shrink when <45% | new file, `kelly.py`, `main.py` |
| 3 | `btc_24h_return` as Feature 28 (3-file contract: `regime_model.py`, `train_regime.py`, `fusion.py`) | 3 files + `main.py` |
| 3 | `auto_retrain.py` `_ROW_TRIGGER_DELTA` 500→200 | `scripts/auto_retrain.py` |

**Deployment order:** Gate 8 + drift→Kelly wiring first (immediate bleeding stop, no model changes) → calibrator fixes → drift monitor fixes + direction tracker → Feature 28 (before June 3 regime train).

**Key constants added:**
- `CALIBRATOR_MODEL_PATH = "models/calibrator.pkl"` in config.py
- `KALSHI_CONSENSUS_THRESHOLD = 0.08` in config.py (raised to 0.25 in session 15)

**Key gotchas for implementation:**
- `y_up = int(direction == outcome)` for the correct label. direction=0, outcome=0 → y_up=1 (market UP); direction=0, outcome=1 → y_up=0 (market DOWN = NO win). This is `(directions == outcomes).astype(float)` in numpy.
- `fresh_kalshi_mid = (result2_best_bid_cents + result2_best_ask_cents) / 200.0` — pass to checklist from the second orderbook fetch data, not from `signal.regime_features["kalshi_implied_prob"]`
- Feature 28 default = 0.0 when len(df1h) < 25; set `stale=True`
- `use_28=True` implies `use_27=True` in `_build_query()` — 28-feature model is strict superset
- `reset_baseline()` must clear `_KEY_BASELINE`, `_KEY_ALERT_COUNT`, `_KEY_TOTAL_COUNT` AND `_KEY_HISTORY` in Redis, and reset `_history` deque and `_total_count=0` in memory
- Gate 8 blocks: write `failed_gate=8` to `gate_rejections` — same logging path as other gates (see existing `_process_market` gate rejection write in main.py)
- `models/` directory doesn't exist — use `os.makedirs("models", exist_ok=True)` before any save
- Do NOT add `_lkg`, `_lkg_written_at`, or `_deribit_lkg` to feature lists — corrupts XGBoost

---

**As of 2026-05-26 session 9: launchd persistence COMPLETE. Kronos now auto-starts on login and auto-restarts on crash. 338 tests pass.**

**Session 9: launchd persistence**

| File | Change |
|------|--------|
| `scripts/launch_kronos.sh` | **New** — wrapper script: `cd` to project dir, `source .env`, `exec python3 main.py`. Required because launchd does not source shell profiles or `.env`. |
| `~/Library/LaunchAgents/com.kronos.v2.plist` | **New** — launchd agent: `RunAtLoad=true`, `KeepAlive=true`, `ThrottleInterval=15s`. Starts on login, restarts on crash. Stdout/stderr → `logs/launchd_stdout.log` / `logs/launchd_stderr.log`. |
| `main.py` | Added `shadow` column comment in `_GATE_REJECTIONS_COLUMN_MIGRATIONS` and at shadow INSERT call — makes the training exclusion rule explicit: any training query must filter `WHERE shadow = 0` to avoid duplicating trades already in `trades.db`. |

**Manage the launchd agent:**
```bash
launchctl load   ~/Library/LaunchAgents/com.kronos.v2.plist   # start + register (persists across reboots)
launchctl unload ~/Library/LaunchAgents/com.kronos.v2.plist   # stop + deregister
launchctl list | grep kronos                                    # check status (PID + exit code)
```

**What persists vs what doesn't:**
- Survives: terminal close, process crash, Mac wake from sleep
- Survives on next login: reboots, logouts (re-registers on login)
- Does NOT survive: Mac fully asleep (process suspended, not dead — resumes on wake)
- For true 24/7 (lid closed): System Settings → Battery → disable "Prevent automatic sleeping when on power adapter" (keeps Mac awake on AC power, lid open or closed)

---

**As of 2026-05-25 session 8: Background Kronos MC loop + second orderbook fetch COMPLETE. Trades now execute at T+180ms instead of T+23s. 338 tests pass.**

**Session 8: Background Kronos MC loop + second orderbook fetch**

| File | Change |
|------|--------|
| `btc_kalshi_system/signal/fusion.py` | `get_signal()` gains `kronos_raw: float \| None = None` parameter — when provided, skips `run_monte_carlo()` call entirely; when None (default), calls MC as before (backward compat for tests) |
| `main.py` | `__init__`: `self._cached_kronos: dict \| None = None`; new `_kronos_background_loop()` coroutine; added to `asyncio.gather()`; cache staleness watchdog added to `_regime_watchdog()` (fires at >360s); `_process_market`: reads `_cached_kronos` before `get_signal()` (None → INFO skip, >600s → ERROR skip), logs strike delta at DEBUG, passes `kronos_raw=cached["prob"]` to `get_signal()`, second orderbook fetch after first checklist passes, re-runs full checklist on fresh prices, uses `result2`+fresh `fill_price_cents` for order placement and DB record |
| `tests/test_main_bg_kronos.py` | **New** — 9 TDD tests: background loop (populates cache, no-rerun on same candle, survives MC exception), process_market cache guard (None skip, stale skip, kronos_raw passthrough), second fetch (abort on failure, abort on checklist fail, fill_price from second fetch) |
| `tests/signal/test_fusion_kronos_raw.py` | **New** — 3 TDD tests: skips MC when kronos_raw provided, calls MC when None, calibrator receives provided value |
| `tests/execution/test_gate_rejections.py` | `_make_trader()` adds `trader._cached_kronos` with valid cache so existing gate rejection tests reach the checklist |

**Architecture — `_cached_kronos` dict structure:**
```python
{
    "prob":       float,          # P(close > strike) from run_monte_carlo
    "candle_ts":  pd.Timestamp,   # timestamp of the 5-min candle that triggered this run
    "computed_at": float,         # time.time() when MC finished
    "strike":     float,          # 15-min reference price used as MC threshold
}
```

**Background loop behavior:** Polls `store.get_ohlcv("5min")` every 10s; detects new candle via `last_candle_ts`; runs `run_monte_carlo` in `asyncio.to_thread`; writes a new dict (never mutates in place); logs at INFO. Errors (MC failure, OHLCV insufficient) caught per-iteration; loop never exits.

**Second orderbook fetch behavior:** After first checklist passes, fetches orderbook again (~90ms). Both paper and live mode abort if this fails. Re-runs full `checklist.run()` on fresh bid/ask/contracts. If second checklist fails, logs INFO with gate number and returns. `fill_price_cents` and `result2` (kelly $, contracts) from second fetch are used for order placement, `OpenPosition`, and SQLite record.

**Test suite: 338 passing (was 326).**

---

**As of 2026-05-25 session 7 (continued): Gate 7 (CVD soft gate) converted to shadow mode — no longer blocks trades. Shadow rows written to `gate_rejections` with `shadow=1` so win-rate tracking continues. 326 tests pass.**

**Session 7 (continued): Gate 7 shadow mode**

| File | Change |
|------|--------|
| `btc_kalshi_system/execution/pretrade_checklist.py` | Gate 7 block removed entirely — checklist always passes regardless of CVD |
| `main.py` | Shadow Gate 7 check added after checklist pass: writes `gate_rejections` row with `shadow=1` when CVD would have triggered, then lets trade proceed. `_GATE_REJECTIONS_COLUMN_MIGRATIONS` extended with `("shadow", "INTEGER DEFAULT 0")` |
| `tests/execution/test_gate7_cvd.py` | Tests updated: "blocks" cases replaced with "no longer blocks" assertions; aligned/mild-CVD pass cases preserved |

**Why disabled:** Live data showed Gate 7 was blocking trades that won 61.9% of the time overall — and 80% on the YES→UP side specifically. The original 32.3% win-rate calibration was from a different market regime. Gate 7 was net-negative in the current regime.

**Shadow query for win-rate tracking:**
```sql
SELECT CASE direction WHEN 1 THEN 'YES→UP' ELSE 'NO→DOWN' END as side,
  COUNT(*) as would_have_blocked,
  ROUND(100.0*SUM(outcome)/COUNT(*),1) as win_pct
FROM gate_rejections
WHERE failed_gate=7 AND shadow=1 AND outcome IS NOT NULL AND aged_out=0
GROUP BY direction
```
Historical real blocks (before shadow mode) remain in the table with `shadow=0`.

---

**As of 2026-05-25 session 7: Multi-source derivatives feed COMPLETE. DerivativesFeed now queries OKX + Hyperliquid + Kraken Futures in parallel; `okx_partial=True` only when all three fail. Kraken spot fallback added for `_fetch_volume_ratio`. Training filter excludes `okx_stale` rows. 326 tests pass.**

**Session 7 (2026-05-25): Multi-source derivatives feed**

| File | Change |
|------|--------|
| `config.py` | Added `HYPERLIQUID_BASE_URL`, `KRAKEN_FUTURES_BASE_URL` constants |
| `btc_kalshi_system/data/derivatives_feed.py` | `_prev_oi` changed `float` → `dict[str, float]` with `"okx"`, `"hyperliquid"`, `"kraken_futures"` keys; OKX logic extracted to `_fetch_okx_funding_and_oi()`; `_fetch_funding_and_oi()` replaced with 3-source parallel gather + averaging; new `_fetch_hyperliquid_funding_and_oi()` (1h→8h normalization); new `_fetch_kraken_futures_funding_and_oi()` (annualized→8h via ÷1095); `_coinglass_funding_and_oi()` updated to use `_prev_oi["okx"]`; `_fetch_volume_ratio()` now falls back to Kraken spot; `_get_kraken_exchange()` lazy-init helper added; `_kraken_trades_data()` uses `_get_kraken_exchange()` |
| `scripts/train_regime.py` | `_EXTRA_FILTERS_20` + `_EXTRA_FILTERS_27` now include `AND COALESCE(okx_stale, 0) = 0` |
| `tests/data/test_derivatives_feed.py` | `make_feed()` updated to use dict `_prev_oi`; `test_coinglass_fallback_when_okx_funding_oi_fails` rewritten to test `_coinglass_funding_and_oi` directly; 6 new tests for HL fetcher, KF fetcher, 3 multi-source fusion scenarios, volume ratio Kraken fallback |

**Kraken Futures funding rate note:** `fundingRate` in their API is annualized. Divide by 1095 (= 365 × 3 funding periods/day) to get 8h equivalent. Verified: live rate 0.0349 → 0.0000319 per 8h (in line with OKX's typical 0.0001–0.0003 range). Test mock uses 0.1095 / 1095 = 0.0001 for clarity.

**Test suite: 326 passing (was 318).**

---

**As of 2026-05-25 session 6 (continued): Deribit Options Feed COMPLETE and verified. First `deribit_stale=0` trade confirmed at 2026-05-25T05:40Z (`atm_iv=30.9`). Both feeds healthy. Accumulation underway: 1/500 fresh rows.**

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
- **~54 trades/day. 21-feature rows: 500 by ~June 2. 27-feature rows: 500 by ~June 3–4.**
- Stats: 402 total trades, 401 resolved, 388 training-ready (21-feature), 1 deribit_stale=0 row
- System running — check PID: `ps aux | grep "[Pp]ython.*main\.py"`
- Latest commit: merge of session 5 DeepSeek enrichment
- Test suite: **312 passing**
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
| ~June 3–4 | 500 `deribit_stale=0` rows → retrain with 27-feature model (first fresh row: 2026-05-25T05:40Z) |

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

- **Background MC loop — trades now execute at T+180ms instead of T+23s.** Decoupling Kronos from the trading cycle eliminates the 23s order-placement delay. Second orderbook fetch resolves the stale-price problem simultaneously. Both paper and live mode abort on second-fetch failure.
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
- **Stale orderbook on order placement (RESOLVED session 8).** Was: orderbook fetched at T+0, Kronos ran ~23s, limit order placed at T+0 price. Fix: background MC loop + second orderbook fetch right before placement. Both paper and live mode abort on second-fetch failure. First fetch kept for feature injection only.
- **403-only exchange failover** — only re-resolved on 403/Forbidden; timeouts and connection resets retried the same dead session object, leaving the feed silently broken for hours. Fixed: re-resolve on ANY exception.
- **`iv_rv_spread` direct subtraction (unit mismatch)** — `atm_iv` is annualised % (~31), `brti_volatility_1h` is dimensionless tick CV (~0.001). Direct subtraction ≈ `atm_iv`, making the spread useless. Fix: annualise brti_vol before subtracting: `brti_vol × sqrt(8760) × 100`. Gives ~22% spread (31% IV − 9% RV). 3 existing DB rows were written with the correct formula (all `deribit_stale=0` rows post-restart used the fixed code). Bug identified and fixed 2026-05-25.
- **`train_regime load_dataset` missing `use_27=True`** — with `_FEATURE_COLS` at 27 entries, the default `_build_query(legacy)` call used `_EXTRA_FILTERS_20` (no `deribit_stale` gate), which would have included rows where features 22–27 are NULL. XGBoost would learn those features as "always missing" at train time while inference supplies real values — silent model corruption. Fix: `_build_query(legacy, use_27=not legacy)`. The retrain correctly returns 0 rows until ≥500 `deribit_stale=0` rows accumulate.
- **`auto_retrain` + `regime_health_check` local feature lists** — both scripts had hardcoded feature column lists (6 and 21 entries) that would `KeyError` when `model.get_regime()` was called after the 27-feature model deployed (it does `features[k] for k in _FEATURE_ORDER`). Fixed to import `_FEATURE_ORDER` from `regime_model.py` directly. Also: `_TRAINING_READY_FILTER` in `auto_retrain` now requires `deribit_stale=0` so the row trigger fires on the correct count (~500 Deribit-fresh rows) rather than the 21-feature count (~388), which would have caused `train_regime.py` to immediately fail with a confusing error on June 2.

---

## Files Touched This Session (2026-05-24, session 6)

**Session 6 (continued): OKX stale flag**

| File | Change |
|------|--------|
| `btc_kalshi_system/data/derivatives_feed.py` | `_coinglass_funding_and_oi()` + `_fetch_funding_and_oi()` return 4-tuple with `okx_partial: bool`; `_fetch_features()` embeds `_okx_partial=True` when partial; `run()` pops flag before writing; `_write_features(okx_partial=)` skips LKG update on partial, always writes CVD ring buffer |
| `btc_kalshi_system/signal/fusion.py` | `TradingSignal.okx_stale: bool = False` field added; `_regime_features()` returns 4-tuple `(features, stale, deribit_stale, okx_stale)`; `generate_signal()` unpacks and passes `okx_stale` to signal |
| `btc_kalshi_system/execution/position_monitor.py` | `_regime_features()` unpack updated to 4-tuple |
| `main.py` | `_TRADES_COLUMN_MIGRATIONS`: `okx_stale INTEGER DEFAULT 0` added; trades INSERT includes `okx_stale` column and value |
| `tests/data/test_derivatives_feed_okx_stale.py` | **New** — 3 tests: LKG not updated on partial, `_okx_partial` embedded in primary key, LKG written on success |
| `tests/signal/test_fusion_okx_stale.py` | **New** — 3 tests: `okx_stale` true on LKG, true on partial, false when fresh |
| `tests/data/test_derivatives_feed.py` | Existing fallback tests updated for 4-tuple return |
| `tests/signal/test_*.py` | All `_regime_features()` 3-tuple unpacks updated to 4-tuple |

**Test suite: 318 passing (was 312).**

**Session 6 (continued): trade_snapshots schema fix**

| File | Change |
|------|--------|
| `main.py` | `_CREATE_TRADE_SNAPSHOTS_TABLE`: 6 new Deribit columns added (after `tape_speed_tpm`, before `kronos_prob`); `_TRADE_SNAPSHOTS_COLUMN_MIGRATIONS` list added; migration loop added in `__init__` — trade_snapshots: 6 new Deribit columns added for T+5/T+10 analytics |
| `btc_kalshi_system/execution/position_monitor.py` | `_write_snapshot()` cols list: 6 new Deribit columns added (after `tape_speed_tpm`, before `kronos_prob`) — trade_snapshots: 6 new Deribit columns added for T+5/T+10 analytics |
| `tests/execution/test_position_monitor.py` | Inline `CREATE TABLE trade_snapshots` in `_make_db()` and `_make_position_monitor()` updated to include 6 new columns |

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

**Immediate (this week):**

1. **Monitor calibrator auto-retrain.** Check `logs/auto_retrain_calibrator.log` after each 2h cron run. Emergency trigger fires when Brier > 0.25 on last 50 k15 rows — current Brier is 0.2527, just above threshold. If it fires every run without improving, raise `_EMERGENCY_BRIER_THRESHOLD` to 0.26 in `scripts/auto_retrain_calibrator.py`. Marker file at `models/calibrator_last_trained.json` — check it exists after first real run.

2. **Monitor trending_down shrink impact.** With `_TRENDING_DOWN_SHRINK=0.5`, combined signals in trending_down will be much more conservative. Track Gate 5 rejections: `SELECT failed_gate, COUNT(*) FROM gate_rejections WHERE deepseek_regime='trending_down' AND DATE(timestamp) >= date('now','-7 days') GROUP BY failed_gate`. Expect more Gate 5 rejections in trending_down as signals get squeezed toward 0.5.

3. **Monitor Gate 5 ranging filter impact.** Track: `SELECT deepseek_regime, failed_gate, COUNT(*) FROM gate_rejections WHERE failed_gate=5 AND DATE(timestamp) >= date('now','-7 days') GROUP BY deepseek_regime`. Win rate on ranging trades passing Gate 5 should move from 47% toward 60%.

4. **Investigate high_uncertainty oversizing.** −$0.84/trade at 51% win rate means Kelly is ignoring something. Check whether high_uncertainty fills are systematically at unfavorable prices (spread eating edge). Query: `SELECT AVG(fill_price_cents), AVG(pnl_dollars) FROM trades WHERE deepseek_regime='high_uncertainty' AND outcome IS NOT NULL`.

**Medium-term (~4–6 weeks):**

4. **Train regime model when clean rows ≥ 500.** `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL AND features_stale=0 AND deribit_stale=0 AND atm_iv IS NOT NULL`. Run `python3 scripts/train_regime.py --dry-run` first. Deploy in shadow mode (REGIME_GATE2_ENFORCING=false), observe ~50 trades, then enforce.

5. **Segmented calibrators after regime model is deployed.** Train a separate logistic calibrator on ranging-only rows. Ranging will have 600+ k15 rows by then. This is the next calibration improvement after the global logistic calibrator activates.

---

**Historical next steps (preserved for reference):**

0. ✅ **Deribit Options Feed live and verified (session 6, 2026-05-24).** `options:features` writing correctly (TTL 400–600s). First `deribit_stale=0` trade confirmed 2026-05-25T05:40Z (`atm_iv=30.9`). Both feeds healthy. **Do NOT retrain the 27-feature model until ≥500 `deribit_stale=0` rows accumulated — separate from the 21-feature retrain gate.** Monitor: `sqlite3 trades.db "SELECT COUNT(*) FROM trades WHERE deribit_stale=0"`

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

6. ✅ **Second orderbook fetch + stale price fix COMPLETE (session 8).** Trades now use fresh prices from second fetch for order placement, Kelly sizing, and DB record. Both paper and live mode abort on second-fetch failure.

7. ✅ **Background Kronos MC loop COMPLETE (session 8).** `_kronos_background_loop` runs in `asyncio.gather()`. Trades now execute at T+180ms instead of T+23s.

8. **Post-go-live (deferred):** Kalshi intra-cycle YES momentum (needs new polling infra). Slippage gate for 15min markets (needs 200+ spread samples first).

---

## Kronos Monte Carlo — Timing Analysis & Decoupling Plan

### Measured timing (2026-05-25, 4 cycles, steady-state)

| Step | Latency |
|------|---------|
| Orderbook fetch (Kalshi REST) | 76–98ms (~90ms typical) |
| Kronos Monte Carlo (100 paths, CPU) | 22,463–27,456ms (~23s typical) |
| Everything else (Redis, checklist, Kelly, order submit) | <200ms |
| **Total cycle wall time** | **~23s** |
| Cycle interval | 300s |
| CPU idle per cycle | ~277s |

Monte Carlo is 100 sequential `predictor.predict()` calls at ~230ms each. Runs in `asyncio.to_thread` so it doesn't block WebSocket feeds, but it blocks trade placement by ~23s every cycle.

### Why the current architecture is wrong

The 300s cycle does: fetch orderbook → run MC (23s) → place order at **T+0 price**.

Two problems:
1. **Stale price**: limit order uses the T+0 orderbook snapshot. By T+23 the market has moved. See Next Step 6.
2. **Wasted idle time**: MC finishes at T+23, then the system sleeps for ~277s. Meanwhile OHLCV ticks in and the signal goes stale. MC could have run again.

### Correct architecture: background MC loop

Decouple MC from the trading cycle entirely:

```
Background task (_kronos_background_loop):
  loop forever:
    wait for new 5-min candle (poll store.get_ohlcv("5min"), detect timestamp change)
    run MC (~23s) → store (prob, candle_timestamp) as self._cached_kronos
    # no sleep needed — next iteration immediately checks for new candle

Main 5-min cycle (_process_market):
  fetch orderbook #1 (~90ms) → inject kalshi_mid + spread into fusion
  read self._cached_kronos  ← instant, no 23s wait
  if cache stale (>10min old): skip cycle, log warning
  run gates on fresh signal
  fetch orderbook #2 (~90ms) → fresh prices for gates + order placement
  place order at fresh ask price
```

**Effect:** trades placed at **T+90ms** instead of T+23s. A 250× reduction in trade latency from cycle start to order submission. Stale orderbook bug also resolved because the second fetch happens right before placing.

**Why once per candle is sufficient:** OHLCV only changes every 5 minutes. Running MC more than once per candle is identical computation on identical input — wasted CPU. The background loop naturally paces itself: 23s of work, ~277s waiting for the next candle.

### MC efficiency options (if 23s per run needs to be reduced)

| Option | Speedup | Cost |
|--------|---------|------|
| Reduce `n_paths` 100→50 | ~2× (→12s) | Noisier prob estimates (±5% vs ±3%) |
| Reduce `n_paths` 100→25 | ~4× (→6s) | Noticeably noisier (±7%) |
| `torch.no_grad()` wrapper | ~10–15% | None — pure win if not already wrapped |
| `ThreadPoolExecutor` across paths | 2–4× (GPU-free) | PyTorch releases GIL during tensor ops; needs hardware testing |
| `sample_count=100` in one call | Potentially 5–10× | The predictor comment says "averages paths internally" — test if a single call with `sample_count=100` gives same variance as 100 calls with `sample_count=1`. If yes, this is the biggest win. |

With the background loop, the 23s per run stops being on the critical path entirely, so efficiency tuning becomes optional rather than urgent. Recommended order: (1) implement background loop first, (2) try `torch.no_grad()` as a free win, (3) only optimize further if CPU load becomes a concern.

### Implementation notes

- `_cached_kronos` should store `{"prob": float, "candle_ts": pd.Timestamp, "computed_at": float}` — `candle_ts` is the timestamp of the candle that triggered the MC run; `computed_at` is `time.time()` when MC finished.
- Cache freshness check in `_process_market`: if `time.time() - cached["computed_at"] > 600`, skip and log — something broke in the background loop.
- Background loop must handle `ValueError: Insufficient OHLCV data` (not enough candles on startup) — catch and retry after 60s.
- `_kronos_background_loop` added to `asyncio.gather()` in `run()` alongside existing feed tasks.
- Strike for the cached MC run: use `self._get_15min_reference_price()` — same as what `_process_market` computes. If strike changes between MC run and cycle (unlikely intra-candle), the cached prob is still directionally valid.

   **Gate 8 (candle_progress / UTC dark gate) — DEFERRED until more data:** Only 33/384 trades have `candle_progress` populated, zero above 0.85. Revisit once we have ≥200 trades with valid candle_progress values. Do not implement until data density justifies it.

---

## exit_reason Diagnosis (2026-05-24)

- **`regime_model._clf` is None** — `RegimeModel.__init__()` sets `_clf = None` and it stays None until `train_regime.py` is run; with only ~15 training-ready rows (need 500), it has never been trained, so `PositionMonitor._evaluate()` hits the `if self.regime_model._clf is None:` bootstrap branch, collects a snapshot, and returns early — `_execute_exit()` (where `exit_reason` is written) is never reached.
- **PositionMonitor IS scheduled** — `self._position_monitor.run()` is in the `asyncio.gather()` call at `main.py:246`, so the coroutine is running; it is not the issue.
- **Trades last long enough for T+5 to fire** — querying `trades.db`: avg time-remaining-in-15min-window when a trade enters is ~599s (well above the 300s T+5 threshold); 324/378 resolved trades (86%) had ≥300s remaining, so trade duration is not the blocker.

**Conclusion:** `exit_reason` will stay NULL until `train_regime.py` is run and the model is loaded. Expected behavior during the bootstrap accumulation phase.

---

## Context / Gotchas

- **Test suite: 338 pass.** `python3 -m pytest` from project root.

- **`_cached_kronos` is None on startup for ~23s.** `_process_market` skips silently (INFO log: "not yet populated") until the background loop completes its first MC run. This is expected behavior — no trades fire during this startup window.

- **Cache staleness thresholds:** `_process_market` skips if `computed_at > 600s` (ERROR log). `_regime_watchdog` fires an OS notification if `computed_at > 360s`. The background loop refreshes every new 5-min candle (~300s cycle), so >360s means the loop missed at least one candle.

- **Strike delta logged at DEBUG only.** The background loop may compute MC at a slightly different strike than the current market strike (if BRTI drifted since the last candle). This delta is logged for diagnostics but does NOT gate the trade — the cached prob is still directionally valid within a candle.

- **`get_signal()` `kronos_raw=None` fallback is test-only.** In production, `_process_market` always passes `kronos_raw=cached["prob"]`. The None path calls `run_monte_carlo()` inline — never trigger this in production (would block the cycle for ~23s and create concurrent MC).

- **Second checklist re-runs full Kelly sizing.** `result2.kelly_dollars` and `result2.kelly_contracts` come from the second checklist call with fresh prices. `OpenPosition`, `place_order`, and `_record_trade_sqlite` all use `result2`, not `result` from the first checklist.

- **Feature order is a 3-file contract.** `regime_model.py` / `train_regime.py` / `fusion._regime_features()` must match exactly. Test: `python3 -m pytest tests/ -k "feature_order"`.

- **CVD ring buffer has TWO stale modes.** Cold (< 5 entries) and stale timestamp (most recent > 360s old). Both zero velocity/acceleration and set `stale=True`. The 360s threshold is intentional — feed writes every 240s, so 360s = missed one full cycle.

- **CVD test mocks must use `time.time()`**, not hardcoded epochs — otherwise freshness check fires.

- **`_FEATURE_COLS_LEGACY` stays at 6 features.** Do not add new features to it.

- **Do not add `_lkg` or `_lkg_written_at` to any feature list.** Corrupts XGBoost inputs.

- **Gate 6 guard `if signal.timeframe != "15min"` must stay.** Removing it blocks all 15min trades.

- **Gate 2 is in SHADOW mode** (`REGIME_GATE2_ENFORCING=false`). Do not flip until regime model has been live for ~50 trades.

- **Gate 2 "Kelly rounds to 0" — possible calibrator underfitting (small sample, treat with caution).** Analysis of 39 blocked trades (17 DOWN, 22 UP) using Kalshi mid as fill proxy showed the two sides behave oppositely:
  - **DOWN blocks (+5.6¢ avg P&L per trade):** Buying NO at ~83¢ avg fill, 87.5% win rate → positive EV. Kelly rounds to 0 because the *calibrated* probability is close to the market price, but the actual win rate substantially exceeds it. Likely cause: calibrator is still underfitted (sub-500 training trades) and is dampening Kronos's DOWN signal toward 50%, making computed edge look thin when the true edge is real.
  - **UP blocks (−4.9¢ avg P&L per trade):** Buying YES at ~87¢ avg fill, 81.8% win rate → negative EV. Gate 2 correctly blocking these.
  - **Sample is small (17 DOWN, 22 UP resolved).** Do not act on this yet. Revisit after the calibrator is trained on ≥500 trades — if DOWN "Kelly rounds to 0" blocks still show positive EV after calibrator training, the Kelly floor or calibration curve may need adjustment.

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

- **Gate 7 is shadow-only** (`shadow=1` in `gate_rejections`). It no longer blocks trades. The original 32.3% YES→UP win-rate calibration was regime-specific; live data showed 80% win rate on blocked YES→UP trades and 56% on NO→DOWN, both net-negative for the gate. **Sample sizes are small (5 YES→UP, 16 NO→DOWN resolved blocks)** — shadow tracking is kept running specifically because the data may not be conclusive yet. Re-evaluate once shadow rows accumulate (target: 50+ per side). Gate could be reimplemented as-is, recalibrated to a different threshold, or replaced with a different regime stat entirely (e.g. `large_print_direction` instead of CVD). Do NOT re-enable without fresh regime-specific data.

- **`gate_rejections.shadow` column** — added via migration (`_GATE_REJECTIONS_COLUMN_MIGRATIONS`). Historical real blocks have `shadow=0` (default). Shadow observations have `shadow=1`. Always filter by `shadow` when analyzing gate effectiveness to avoid mixing the two populations.

- **k15_raw is anti-correlated with P(up): Spearman = −0.12.** High k15_raw (>0.8) → P(up) = 38% (n=52 clean rows). The logistic [raw, raw²] calibrator can learn this inverted-U. Isotonic could not.

- **Regime model has NEVER been trained.** `regime_model.pkl` does not exist. Do not attempt training below 500 deribit-clean rows — produces confidently wrong labels. All clean rows are from 2026-05-25 onward.

- **Gate 5 is now regime-aware.** ranging: `min_required = max(spread+0.005, 0.15)`. high_uncertainty: `max(spread+0.005, 0.08)`. trending_up/down: unchanged. The 0.15 threshold filters the 0.55–0.65 cal_prob dead zone in ranging (23% win rate on 36 trades). Do not reduce the 0.15 threshold without fresh win-rate data showing the dead zone has resolved.

- **Do not retrain calibrator on kronos_raw (5min).** Session 17 changed `train_calibrator.py` to query `kronos_raw_15min`. The old isotonic model trained on 5min raw is now replaced. Mixing 5min and 15min raw in one calibrator would corrupt it.

- **Dirty rows (features_stale=1 or deribit_stale=1) are valid for calibrator training.** The stale flags indicate bad feature values for the regime model, not bad kronos_raw or outcome labels. The calibrator only uses kronos_raw_15min + outcome — use all rows, not just clean ones.

- **`dump.rdb` and `trades.db.bak.*` must NOT be committed.**

- **`_RANGING_SHRINK=0.7`, `_BOOTSTRAP_SHRINK=0.8`, `_UNCERTAINTY_SHRINK=0.5`, `_TRENDING_DOWN_SHRINK=0.5`** — do not equate. trending_down and high_uncertainty use the same shrink factor (0.5) but for different empirical reasons: high_uncertainty is noisy, trending_down has an inverted k15 signal (76.9% avg confidence → 28.6% win rate).

- **DeepSeek returns `NEUTRAL_DEFAULT` on 402, not `SAFE_DEFAULT`.**

- **DerivativesFeed re-resolves on ANY exception** (commit `229b88b`). Prior to this fix, only 403/Forbidden triggered failover; timeouts/resets silently kept a dead session alive. If the feed goes quiet again, check `redis-cli ttl regime:features` — TTL of -2 means it expired and feed is down. Restart main.py to recover.

- **Multi-source derivatives feed (session 7):** `_fetch_funding_and_oi()` now queries OKX, Hyperliquid, and Kraken Futures in parallel; averages whichever succeed. `okx_partial=True` only when ALL three fail. Hyperliquid reports 1h funding (multiply × 8 for 8h equiv). Kraken reports annualized funding (divide by 1095 for 8h equiv). `_prev_oi` is now a dict `{"okx": float, "hyperliquid": float, "kraken_futures": float}` — each source tracks its own prev OI for delta computation. `_coinglass_funding_and_oi` is preserved but no longer in the main fallback chain; it can be called directly for debugging.

- **Restart procedure (launchd):** `launchctl unload ~/Library/LaunchAgents/com.kronos.v2.plist && launchctl load ~/Library/LaunchAgents/com.kronos.v2.plist` — launchd will restart automatically on crash. Logs in `logs/launchd_stdout.log` and `logs/launchd_stderr.log`. Old manual procedure (direct python3) is superseded.

- **Feed health check:** `redis-cli ttl regime:features` should return 400–600. If -2, feed is down. `redis-cli get regime:features:lkg` shows LKG age via `_lkg_written_at` field.

---

## Future Roadmap

Ordered by data requirements. Each item has a clear trigger condition before implementation begins.

### Phase 1 — Calibrator activates ✅ COMPLETE (2026-05-30)

- ✅ Calibrator retrained on 129 k15 rows; Brier improved 0.2708→0.2463; passthrough=False
- ✅ `_MIN_SAMPLES` lowered 300→100 to allow fitting on available k15 data
- ✅ `auto_retrain_calibrator.py` added — triggers every 2h via cron (`0 */2 * * *`); +50 row, 7-day, and Brier>0.25 emergency triggers
- ✅ `_TRENDING_DOWN_SHRINK=0.5` added in fusion.py — addresses empirical inverted-signal in trending_down
- Monitor: does Gate 5 ranging filter improve win rate toward 60%? If not, the signal quality problem is deeper than calibration.

### Phase 2 — Regime model trains (~early July)
**Trigger:** `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL AND features_stale=0 AND deribit_stale=0 AND atm_iv IS NOT NULL` ≥ 500.

- Run `python3 scripts/train_regime.py --dry-run`. Check Brier < 0.25 and feature importances make sense (funding_rate, cvd_normalized, brti_momentum should rank high).
- Deploy in shadow mode (REGIME_GATE2_ENFORCING=false). Observe ~50 trades. Gate 2 disagreement rate should be < 35% — if higher, the model is noisy and needs more data before enforcing.
- Once enforcing, the 0.4 regime weight in fusion.py becomes real. Expect calibrated_prob distribution to widen slightly.
- Add `kronos_raw_15min` and `kronos_raw` as features 29–30 in the NEXT retrain after this one (not this one — too soon, not enough k15 rows per regime to be meaningful).

### Phase 3 — Segmented calibrators (~August)
**Trigger:** ≥ 500 k15_raw rows per regime for ranging (trending regimes may never hit this and should stay on global calibrator).

- Train a separate logistic [raw, raw²] calibrator on ranging-only rows.
- In `fusion.py`, route `calibrator.transform()` through a `SegmentedCalibrator` wrapper that checks `deepseek_regime` and dispatches to the right model, falling back to global for regimes without enough data.
- Expect the ranging calibrator to learn a more aggressive compression (closer to 0.5) than the global one.
- high_uncertainty: train segmented calibrator only if ≥ 400 k15 rows accumulate. Otherwise keep on global.

### Phase 4 — MC path weighting with S/R (~late 2026, if ever)
**Trigger:** hourly_sr_proximity win rate shows a consistent pattern across ≥ 200 trades per bucket. Currently flat (45–52% across all buckets). Requires redefining S/R from 24h high/low (range position) to pivot-point or volume-cluster based levels.

- The correct approach: post-process `predicted_closes` array in `run_monte_carlo()` by downweighting paths that would require breaking through a strong S/R level. Clean architectural change that doesn't touch the Kronos model.
- Do not pursue until the SR signal is demonstrably non-flat. Current data does not support it.

### Phase 5 — Go live (PAPER_TRADING=false)
**Prerequisites before flipping:**
- [ ] Regime model trained and deployed (Gate 2 enforcing for ≥50 trades without degrading win rate)
- [ ] Calibrator active on k15 data (not passthrough)
- [ ] Ranging win rate ≥ 52% over 100+ trades after Gate 5 regime-aware filter
- [ ] high_uncertainty per-trade P&L positive or flat (currently −$0.84/trade)
- [ ] 14-day rolling win rate > 52% overall

### Signals to watch continuously
These don't require implementation — just monitoring queries to run weekly:

| Signal | Query | Action trigger |
|--------|-------|----------------|
| Gate 5 ranging blocks | `SELECT COUNT(*) FROM gate_rejections WHERE failed_gate=5 AND deepseek_regime='ranging' AND DATE(timestamp) >= date('now','-7 days')` | If < 5/day, ranging signal improved; consider lowering 0.15 threshold |
| high_uncertainty P&L | `SELECT AVG(pnl_dollars) FROM trades WHERE deepseek_regime='high_uncertainty' AND outcome IS NOT NULL AND DATE(timestamp) >= date('now','-14 days')` | If still < −0.50/trade after calibrator activates, consider suppressing regime entirely |
| Gate 2 shadow disagreement | `SELECT COUNT(*) FROM trades WHERE regime_prob IS NOT NULL AND ((direction=1 AND regime_prob < 0.5) OR (direction=0 AND regime_prob >= 0.5))` | If > 40% after regime model deploys, do not enforce Gate 2 yet |
| k15 calibration health | `python3 scripts/train_calibrator.py --dry-run` | Run weekly once calibrator is active; Brier should trend downward |
