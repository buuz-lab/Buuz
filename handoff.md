# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min
up/down markets). Forecast direction via Kronos + XGBoost regime classifier +
DeepSeek gate, size with fractional Kelly, run 6 pre-trade gates. **Current
focus:** accumulate 500 training-ready rows, then train and deploy the RegimeModel.
The pipeline is fully instrumented — just waiting on data volume.

---

## Current Progress

**As of 2026-05-23 ~06:10 UTC: 89 training-ready rows. System is live and running.**

- `PAPER_TRADING=true` in `.env`
- **~43 trades/day resolved rate. Expected to hit 500 training-ready rows ~2026-06-02 (~9.5 days).**
- Stats at this session: 304 total trades / 89 training-ready rows, 177W / 125L (59%), Net P&L: +$289.63.
- System is **running** (PID changes on restart — check `pgrep -f main.py`).
- **OKX outage 14:20–21:00+ PDT 2026-05-22.** Watchdog fired ERROR continuously. Trades during
  this window fired on stale (zero) features. LKG fix addresses this for future outages.

**Go-live thresholds (both must be met):**
- ≥ 500 resolved trades total — for calibrator
- ≥ 500 training-ready rows (`features_stale=0 AND funding_rate IS NOT NULL AND
  outcome IS NOT NULL`) — for regime model training

---

## What Worked (most recent first)

- **Last-Known-Good (LKG) feature fallback.** During exchange outages `regime:features`
  expires and was returning all-zero features to Gate 2. Fix: `_write_features()` now
  also writes `regime:features:lkg` (TTL=24h) with a `_lkg_written_at` timestamp.
  `_get_market_context()` tries LKG when the primary key is expired, logs the age, and
  injects `_lkg=True` sentinel. `fusion._regime_features()` treats `_lkg=True` as stale
  so those rows are excluded from training — same as zero-fallback rows. Gate 2 now runs
  on real (aged) features instead of zeros during multi-hour OKX outages.
  Triggered by: 14:20–21:00+ PDT OKX outage on 2026-05-22.
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
- **Loguru hardening (`enqueue=True, catch=True`).** File sink no longer silently
  drops itself on write/rotation errors. `enqueue=True` makes writes non-blocking
  and thread-safe under asyncio.
- **TTL=600s for `regime:features`.** Doubled from 300s — tolerates one full
  missed refresh cycle without expiring.
- **Overlapping refresh in `DerivativesFeed.run()`.** After a successful write,
  sleeps 240s instead of 300s. Key is always renewed with 60s to spare, even if
  OKX fetch takes extra time.
- **`_regime_watchdog` coroutine in `KronosV2`.** Checks `regime:features` TTL
  every 60s. Fires macOS notifications (via `osascript`) and logs WARNING/ERROR
  if TTL drops below 90s or key is fully expired. Wired into `asyncio.gather`.

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
- **TTL == refresh interval (both 300s).** Any asyncio scheduling drift or slow
  OKX call caused the key to expire before renewal. Root cause of all staleness
  incidents observed. Fixed via TTL=600s + overlapping refresh.
- **Loguru silent sink death.** At 05:07 UTC all 4 exchange feeds disconnected
  simultaneously + Kalshi API timeout. Log rotation failed during this storm and
  loguru silently dropped the file sink at 05:16 UTC. System traded blind for 7
  hours with no log record. Fixed via `catch=True` + `enqueue=True`.

---

## Files Touched / Created

### This session (2026-05-23 — per-ticker position cap)

| File | Change |
|------|--------|
| `btc_kalshi_system/portfolio/monitor.py` | Added `ticker_position_count(ticker) -> int` method — counts open positions for a specific ticker. |
| `main.py` | Added `MAX_POSITIONS_PER_TICKER = 3` constant. Added gate before pre-trade checklist: skips market if `ticker_position_count(ticker) >= 3`. |

**Why:** Without this cap the system was averaging-down into every price dip on the same
ticker, stacking 4+ positions (43x44¢ → 83x25¢ → 106x20¢ → 149x13¢) and concentrating
$80+ of exposure on a single bet. The existing `has_timeframe_position` check only applies
a Kelly correlation discount (0.7×) — it never blocks entry. The `MAX_TOTAL_EXPOSURE_DOLLARS`
cap didn't trigger because each trade was small in dollar terms at low prices. The "3 trades
per market" behavior that previously existed was emergent from the exposure math at typical
prices (~50¢), not an explicit rule. This commit makes it explicit.

**Commit:** `2e7480f`

### This session (2026-05-22, late evening — Coinglass + Kraken fallbacks)

| File | Change |
|------|--------|
| `btc_kalshi_system/data/derivatives_feed.py` | Split `_fetch_features()` into `_fetch_funding_and_oi()` (OKX→Coinglass REST fallback) and `_fetch_trades_data()` (OKX→Kraken ccxt fallback). Added `_coinglass_funding_and_oi()` and `_kraken_trades_data()`. Constants `_KRAKEN_SYMBOL`, `_COINGLASS_BASE` added. `_kraken_exchange` lazy attribute in `__init__`. |
| `config.py` | Added `COINGLASS_API_KEY: str = os.getenv("COINGLASS_API_KEY", "")`. |
| `tests/data/test_derivatives_feed.py` | +2 tests: Coinglass fallback called when OKX funding/OI fails; Kraken fallback called when OKX trades fail, with correct CVD value asserted. `make_feed()` updated with `_kraken_exchange`, `_ccxt_async`, `_prev_oi` attributes. |

### This session (2026-05-22, evening — LKG fix)

| File | Change |
|------|--------|
| `btc_kalshi_system/data/derivatives_feed.py` | Added `_LKG_TTL = 86_400`. `_write_features()` now also writes `regime:features:lkg` (24h TTL) with `_lkg_written_at` timestamp. |
| `main.py` | `_get_market_context()`: when `regime:features` expired, tries `regime:features:lkg`; pops timestamp, logs age, injects `_lkg=True` sentinel, returns LKG dict instead of `{}`. |
| `btc_kalshi_system/signal/fusion.py` | `_regime_features()`: staleness check expanded to `stale = not ctx or ctx.get("_lkg", False)` so LKG rows are still excluded from training. |
| `tests/data/test_derivatives_feed.py` | +1 test: `test_lkg_key_written_on_successful_write` — verifies LKG key exists, TTL~86400, all 6 feature keys present, `_lkg_written_at` populated. |

### This session (2026-05-22, afternoon)

| File | Change |
|------|--------|
| `main.py` | Loguru file sink: added `enqueue=True, catch=True`. Added `_regime_watchdog` coroutine (TTL check every 60s, macOS notifications via `osascript` on WARNING/ERROR). Wired watchdog into `asyncio.gather`. Added `import subprocess`. |
| `btc_kalshi_system/data/derivatives_feed.py` | `_FEATURES_TTL` 300→600. `run()` overlapping refresh: sleep 240s (not 300s) after successful write so key never expires before renewal. |

### Prior session (commit `4f8ff3f`, 2026-05-22 morning)

| File | Change |
|------|--------|
| `scripts/train_regime.py` | Feature variance gate, 3-fold walk-forward CV, feature importance logging, `--max-rows N` flag. |
| `scripts/regime_health_check.py` | **NEW.** Diagnostic script for training progress, staleness rate, model health. |
| `scripts/auto_retrain.py` | **NEW.** Cron-driven retraining with emergency/row/time triggers. |
| `btc_kalshi_system/data/derivatives_feed.py` | `_funding_rate_trend` fallback: `return 0.0`. |
| `tests/data/test_derivatives_feed.py` | +1 test for funding_rate_trend zero fallback. |

### Prior session (commit `d3933be`, 2026-05-21)

| File | Change |
|------|--------|
| `btc_kalshi_system/signal/fusion.py` | `TradingSignal` carries `regime_features` + `features_stale`. Gate 2 soft-launch. |
| `btc_kalshi_system/models/regime_model.py` | `train()` accepts `**xgb_kwargs`. |
| `main.py` | 23-column schema, idempotent ALTER TABLE, regime feature persistence, `RegimeModel.load()` fallback. |
| `config.py` | `REGIME_MODEL_PATH`, `REGIME_GATE2_ENFORCING`. |

---

## Next Steps

1. **Restart the bot** to pick up this session's changes (loguru hardening,
   overlapping refresh, watchdog). Confirm watchdog is logging TTL checks and
   that no macOS notifications fire immediately after startup.

2. **Monitor training-ready row accumulation.** Run `python3 scripts/regime_health_check.py`
   daily. Expect 500 rows ~2026-06-07 at current rate.

3. **(Optional) Add auto_retrain to crontab.** Copy the crontab line from the top
   of `scripts/auto_retrain.py`. Test first with `python3 scripts/auto_retrain.py --dry-run`.

4. **Train the model when ready.** `python3 scripts/train_regime.py --dry-run`
   previews Brier / accuracy. If sane (Brier < 0.25, Kronos agreement > 55%),
   re-run without `--dry-run` to save `models/regime.pkl`. Restart — it auto-loads.

5. **Flip `REGIME_GATE2_ENFORCING=true` in `.env` and restart** after observing
   ~50 shadow trades with a 20-40% disagreement rate (not 50%+).

---

## Context / Gotchas

- **Test suite invariant: 207 pass.** Run from project root: `python3 -m pytest`.
- **Bot must be restarted** for this session's changes to take effect.
- **Watchdog uses whatever redis client is available on existing KronosV2 instance
  variables** — do not add a second redis connection. Check `self._store._redis`
  or equivalent.
- **macOS notifications use `osascript`** — WARNING fires with sound "Basso",
  ERROR fires with sound "Sosumi". If notifications stop appearing, check that
  Terminal/Python has notification permissions in macOS System Settings.
- **Kronos preload rule.** Apple Silicon segfault avoidance: preload Kronos in
  `KronosV2.__init__()` before asyncio, `map_location="cpu"`, `set_num_threads(1)`
  BEFORE `from_pretrained`. Do not refactor.
- **Label = `int(direction == outcome)`, NOT `outcome`.** `outcome` means "did
  this trade win," inverted for short trades. All evaluation code uses this formula.
- **Two `brti_volatility_1h` implementations exist.** `DerivativesFeed` (Redis
  ticks) vs `fusion._regime_features()` (5-min OHLCV pct_change). Persisted column
  is the fusion version. Do not consolidate after model training begins.
- **CV fold windows must stay out of the held-out test set.** Folds are computed
  on `n_cv = n_total - args.test_size` rows, not all rows.
- **`_TRAINING_READY_FILTER` in auto_retrain.py uses the looser 3-condition
  filter** (`features_stale=0, funding_rate IS NOT NULL, outcome IS NOT NULL`) —
  not the strict 6-column filter.
- **Gate 2 starts in SHADOW mode after loading a model.** Set
  `REGIME_GATE2_ENFORCING=true` only after observing ~50 trades. Default `false`.
- **`dump.rdb` is in git but must NOT be staged in commits.** Stage code files
  explicitly. Same for `trades.db.bak.*`.
- **`prev_oi` in `DerivativesFeed` is an instance attribute.** One false-zero
  `oi_delta_pct` on cold start is expected and harmless.
- **LKG sentinel `_lkg=True` in market context dict.** When `_get_market_context()`
  falls back to `regime:features:lkg`, it injects `_lkg=True` into the returned dict.
  `fusion._regime_features()` checks `ctx.get("_lkg", False)` to force `stale=True`.
  The `_lkg` and `_lkg_written_at` keys are never included in the 6-feature dict fed to
  XGBoost — the feature dict is built by explicitly listing the 6 keys. Do not add
  `_lkg` to the feature list or it will corrupt model inputs.
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
- **Coinglass fallback requires `COINGLASS_API_KEY` in `.env`** (`CG-API-KEY` header, v4 API). Without it, the Coinglass fallback logs a WARNING and returns zeros — OKX must recover on its own. Endpoint used: `/api/futures/funding-rate/history` (OKX, BTCUSDT, 8h interval) + `/api/futures/open-interest/exchange-list` (BTC). OI is filtered for `exchange=="OKX"` to stay on the same unit scale as the primary path.
- **Kraken fallback uses spot `BTC/USD` trades** (not a perp). CVD and basis_spread_pct still computed correctly — spot price ≈ BRTI so basis ≈ 0. No API key required.
