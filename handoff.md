# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min
up/down markets). Forecast direction via Kronos + XGBoost regime classifier +
DeepSeek gate, size with fractional Kelly, run 6 pre-trade gates. **Current
focus:** accumulate 500 training-ready rows, then train and deploy the RegimeModel.
The pipeline is fully instrumented — just waiting on data volume.

---

## Current Progress

**As of 2026-05-22 ~09:10 UTC: 226 total trades / 17 training-ready rows (3% of 500).
142W / 83L (63%). Net P&L: +$307.01.**

- `PAPER_TRADING=true` in `.env`
- **Commit `4f8ff3f` pushed to `origin/main`.** Regime training pipeline hardened
  this session (see Files Touched below).
- **System running as PID 20513** with 23-column schema.
- **Bootstrap clock:** ~15 days to 500 training-ready rows at current resolved rate
  (~32 trades/day). Check with `python3 scripts/regime_health_check.py`.
- **Auto-retraining wired:** `scripts/auto_retrain.py` can be added to crontab
  (every 6 hours). It fires automatically when +500 new rows resolve, 14 days
  elapse, or rolling accuracy drops below 55%.

**Go-live thresholds (both must be met):**
- ≥ 500 resolved trades total — for calibrator
- ≥ 500 training-ready rows (`features_stale=0 AND funding_rate IS NOT NULL AND
  outcome IS NOT NULL`) — for regime model training

---

## What Worked

- **Per-trade feature snapshot via TradingSignal.** Added `regime_features: dict`
  and `features_stale: bool` to the `TradingSignal` dataclass. Persisted values
  are exactly what the model was fed — no train/serve skew possible.
- **Idempotent ALTER TABLE migration.** `_TRADES_COLUMN_MIGRATIONS` in `main.py`
  runs every startup, swallowing duplicate-column errors.
- **`features_stale` flag.** When Redis `regime:features` is missing/expired,
  fusion uses 0.0 fallbacks for inference but tags the row stale. Training filters
  on `features_stale=0`.
- **Label semantics.** `up_label = int(direction == outcome)` — NOT `outcome`.
  The `outcome` column means "did this trade win," which is inverted for short
  trades. All evaluation code in this repo uses this formula consistently.
- **Soft-launch Gate 2.** `REGIME_GATE2_ENFORCING=false` default. Disagreements
  are logged but not blocked. Avoids sudden ~30% drop in trade frequency.
- **3-fold walk-forward CV** in train_regime.py for evaluation confidence.
  Fold windows operate on `X_train` only — held-out `--test-size` rows stay
  completely outside CV. Warns if Brier std > 0.05.
- **`return 0.0` in `_funding_rate_trend` fallback.** When the 4-hour window
  can't be computed, returns neutral (not a noisy delta from history[0]).
- **`_FRESH_FILTER` constant in regime_health_check.py / auto_retrain.py.**
  Requires all 6 feature columns NOT NULL (not just `funding_rate`) to prevent
  NaN passthrough to inference.

---

## What Failed / Avoided

- **Backfilling the 200+ pre-instrumentation trades.** Rejected — funding rate /
  OI / CVD / basis history not reliably reconstructable. Pre-migration rows stay
  training-invisible (NULL features), used only for calibrator + edge tracker.
- **Re-reading Redis at `_record_trade_sqlite`.** Would reintroduce train/serve
  skew if a cycle spans a DerivativesFeed refresh. Signal carries the snapshot.
- **Consolidating the two `brti_volatility_1h` implementations.** `DerivativesFeed`
  uses Redis ticks; `fusion._regime_features()` uses 5-min OHLCV pct_change std.
  The persisted column is the fusion version. Do NOT consolidate after training
  begins — it would invalidate any trained model.
- **Using `old = history[0]` as funding_rate_trend fallback.** Made the lookback
  window non-deterministic (silently stretched to 80+ hours). Fixed to `return 0.0`.

---

## Files Touched / Created

### This session (commit `4f8ff3f`, 2026-05-22)

| File | Change |
|------|--------|
| `scripts/train_regime.py` | Feature variance gate (warn+exit if >2 near-zero std features, `--force` to override). 3-fold expanding walk-forward CV replacing single split. Feature importance logging post-save with single-feature-dominance warning. `--max-rows N` flag for rolling-window training. |
| `scripts/regime_health_check.py` | **NEW.** Diagnostic: training progress + feature variance stats, staleness rate, zero-variance flags, rolling accuracy/Brier on deployed model (DEGRADED if acc<55% or Brier>0.25). Run as `python3 scripts/regime_health_check.py`. |
| `scripts/auto_retrain.py` | **NEW.** Cron-driven retraining. Emergency (acc<55%), row-based (+500 rows), time-based (14 days) triggers. Rolling window: uses all data until 1500 rows, then last 1200. `--force` / `--dry-run` flags. Marker: `models/last_trained.json`. |
| `btc_kalshi_system/data/derivatives_feed.py` | `_funding_rate_trend` fallback: `return 0.0` instead of `old = history[0]`. Updated docstring. |
| `tests/data/test_derivatives_feed.py` | +1 test: `test_funding_rate_trend_returns_zero_when_no_entry_older_than_window`. |

### Prior session (commit `d3933be`, 2026-05-21)

| File | Change |
|------|--------|
| `btc_kalshi_system/signal/fusion.py` | `TradingSignal` carries `regime_features` + `features_stale`. Gate 2 soft-launch via `config.REGIME_GATE2_ENFORCING`. |
| `btc_kalshi_system/models/regime_model.py` | `train()` accepts `**xgb_kwargs`. |
| `main.py` | 23-column schema, idempotent ALTER TABLE, regime feature persistence, `RegimeModel.load()` with FileNotFoundError fallback. |
| `config.py` | `REGIME_MODEL_PATH`, `REGIME_GATE2_ENFORCING`. |

---

## Next Steps

1. **Monitor training-ready row accumulation.** Run `python3 scripts/regime_health_check.py`
   daily. At ~32 trades/day, expect 500 rows around 2026-06-07. The script will
   print estimated days and flag any pipeline issues early.

2. **(Optional) Add auto_retrain to crontab.** Copy the crontab line from the top
   of `scripts/auto_retrain.py`. It will fire automatically when 500 new rows
   resolve. Test with `python3 scripts/auto_retrain.py --dry-run` first.

3. **Train the model when ready.** `python3 scripts/train_regime.py --dry-run`
   previews Brier / accuracy / Kronos-agreement. If sane (Brier < 0.25, Kronos
   agreement > 55%), re-run without `--dry-run` to save `models/regime.pkl`.
   Restart KronosV2 — it auto-loads on init.

4. **Observe ~50 trades in shadow Gate 2.** Log lines like
   `Gate 2 disagreement: kronos_direction=... regime_direction=...` will appear.
   Verify disagreement rate is 20-40%, not 50%+.

5. **Flip `REGIME_GATE2_ENFORCING=true` in `.env` and restart.** Activates real
   Gate 2 enforcement. Now both go-live thresholds can be evaluated against the
   fully trained pipeline.

---

## Context / Gotchas

- **Test suite invariant: 204 pass** (was 203; +1 funding_rate_trend test added
  this session). Run from project root: `python3 -m pytest`.
- **Kronos preload rule.** Apple Silicon segfault avoidance: preload Kronos in
  `KronosV2.__init__()` before asyncio, `map_location="cpu"`, `set_num_threads(1)`
  BEFORE `from_pretrained`. Do not refactor.
- **Label = `int(direction == outcome)`, NOT `outcome`.** `outcome` means "did
  this trade win," inverted for short trades. All evaluation code uses this formula.
- **Two `brti_volatility_1h` implementations exist.** `DerivativesFeed` (Redis
  ticks) vs `fusion._regime_features()` (5-min OHLCV pct_change). Persisted column
  is the fusion version. Do not consolidate after model training begins.
- **CV fold windows must stay out of the held-out test set.** Folds are computed
  on `n_cv = n_total - args.test_size` rows, not all rows — prevents fold 3 from
  overlapping the held-out set.
- **`_TRAINING_READY_FILTER` in auto_retrain.py uses the looser 3-condition
  filter** (`features_stale=0, funding_rate IS NOT NULL, outcome IS NOT NULL`) to
  match train_regime.py's actual query — not the strict 6-column filter.
- **Gate 2 starts in SHADOW mode after loading a model.** Set
  `REGIME_GATE2_ENFORCING=true` only after observing ~50 trades. Default `false`.
- **`dump.rdb` is in git but must NOT be staged in commits.** Stage code files
  explicitly. Same for `trades.db.bak.*`.
- **`prev_oi` in `DerivativesFeed` is an instance attribute.** One false-zero
  `oi_delta_pct` on cold start is expected and harmless.
- **Calibrator is independent.** Uses only `kronos_raw + outcome`, not regime
  features. Hits its 500-sample threshold separately.
- **DeepSeek `NEUTRAL_DEFAULT` on 402, not `SAFE_DEFAULT`.**
- **`_RANGING_SHRINK = 0.7`, `_BOOTSTRAP_SHRINK = 0.8`, `_UNCERTAINTY_SHRINK =
  0.5`** in `fusion.py`. Bootstrap shrink is intentionally lighter than uncertainty
  shrink. Do not equate.
- **15-min reference price = last completed 15-min BRTI candle close.** Do not
  revert to `composite_price`.
- **Gate 6 is skipped for `timeframe == "15min"`.**
- **RSA-PSS Kalshi signing, sign path-only.**
