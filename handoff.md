# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min
up/down markets). Forecast direction via Kronos + XGBoost regime classifier +
DeepSeek gate, size with fractional Kelly, run 6 pre-trade gates. **Current
focus:** accumulate 500 training-ready rows, then train and deploy the RegimeModel.
The pipeline is fully instrumented — just waiting on data volume.

---

## Current Progress

**As of 2026-05-23 ~14:00 UTC: 132 training-ready rows. System is live (PID 44671).**

- `PAPER_TRADING=true` in `.env`
- **~49 trades/day resolved rate. Expected to hit 500 training-ready rows ~2026-06-01 (~7.5 days).**
- Stats at this handoff: 345 total trades / 132 training-ready rows, 192W / 153L (55.6%), Net P&L: +$21.06
- System is **running** — confirm with `ps aux | grep "[Pp]ython.*main\.py"`
- Latest commit: `a85c349`
- Currently in a **quiet period** — Kronos gating on Gate 2 "Kelly size rounds to 0 contracts" due to elevated volatility (brti_vol ~0.0014, 2x normal). No trades since 13:00 UTC. This is correct self-limiting behavior.

**Go-live thresholds (both must be met):**
- ≥ 500 resolved trades total — for calibrator
- ≥ 500 training-ready rows (`features_stale=0 AND funding_rate IS NOT NULL AND
  outcome IS NOT NULL`) — for regime model training

---

## What Worked (most recent first)

- **Win rate by price script (`scripts/win_rate_by_price.py`, commit `a85c349`).**
  Run anytime to see win rate, net P&L, and avg P&L per trade broken down by entry
  price bucket. Flags: `--bucket 5` (5¢ buckets), `--dir yes/no`, `--min-trades N`.
  Current findings: 0–19¢ = 0% win rate ($-19/trade avg), 20–49¢ = 36–39% but
  positive avg P&L in 20–29¢ bucket due to large payouts, 60+¢ = 67–82%.

- **Entry price floor removed (commit `6f74f82`).** `MIN_ENTRY_PRICE_CENTS=20` was
  added then removed after a blocked 19¢ NO→DOWN trade would have won. Kronos appears
  to have genuine contrarian edge at low prices. Monitor sub-20¢ win rate via the
  script as data accumulates — if it stays at 0% consider re-adding the floor.

- **Per-side position cap (commit `bcd3967`).** Replaced blanket `MAX_POSITIONS_PER_TICKER=3`
  with `MAX_POSITIONS_PER_TICKER_PER_SIDE=2`: YES and NO positions tracked separately.
  Kronos can now flip and enter the opposite direction on the same market if CVD flips
  mid-candle. Added `ticker_direction_count(ticker, direction)` to `PortfolioMonitor` (Redis-backed).

- **`floor_strike` as primary strike source (commit `ee2bc31`).** Kalshi sets
  `floor_strike` to the BRTI average at market open — the canonical resolution
  reference price. Added `> 0` guard to reject unset markets. Previously accepted
  `floor_strike=0`, making Kronos compute P(BTC > $0) ≈ 100%, biasing all signals YES→UP.

- **Circuit breaker drawdown skipped in paper trading (commit `bc9f988`).** Daily
  drawdown check only runs in live mode. In paper mode it was halting data collection
  when paper P&L crossed -$200.

- **Per-ticker cap (Redis-backed) (commit `82375d3`).** `ticker_direction_count`
  reads from `portfolio:open_positions` Redis hash — authoritative across all
  processes. In-memory counts fail when multiple main.py processes are running.

- **Last-Known-Good (LKG) feature fallback.** During exchange outages `regime:features`
  expires. Fix: writes `regime:features:lkg` (TTL=24h). `_get_market_context()` tries
  LKG when primary expired. `fusion._regime_features()` treats `_lkg=True` as stale.

- **CVD oscillation insight.** Today CVD swung -0.747→+0.679→-0.591→+0.373→-0.567→+0.623
  within 2 hours. Kronos (bootstrap mode, Monte Carlo only) doesn't read CVD — it fires
  on price momentum alone. This causes 5-6 consecutive losses when CVD and price diverge.
  The regime model gating on CVD alignment is the long-term fix.

- **Win rate analysis.** YES→UP with negative CVD: 32.3% (well below breakeven).
  Price bucket win rates: <50¢ = 37-44%, 50-65¢ = 56.4%, 65+¢ = 72.4%.
  NO→DOWN all-time: 60.9% vs YES→UP: 51.4% — real directional asymmetry.

- **Coinglass + Kraken fallbacks for derivatives feed.**
- **Loguru hardening (`enqueue=True, catch=True`).**
- **TTL=600s + overlapping 240s refresh for `regime:features`.**
- **`_regime_watchdog` coroutine** (TTL check every 60s, macOS notifications).

---

## What Failed / Avoided

- **Blanket MAX_POSITIONS_PER_TICKER=3.** Prevented CVD flip recovery — when CVD
  flipped mid-candle, the cap blocked the correctly-directioned opposite trade.
  Replaced by per-side cap.
- **20¢ entry price floor (added then removed).** Added to prevent 169x12¢-style trades,
  but removed after a blocked 19¢ trade would have won. Sub-20¢ data too thin (6 trades)
  to make a definitive call. Monitor via `scripts/win_rate_by_price.py`.
- **In-memory position count for per-ticker cap.** Broke when multiple stale main.py
  processes ran simultaneously. Each had its own `_positions` dict, so the cap was
  per-process, not global. Always use Redis-backed count.
- **floor_strike=0 accepted as valid.** Made Kronos compute P(BTC > $0) ≈ 100%.
  Fixed with `if v > 0` guard.
- **Circuit breaker in paper mode.** Tripped at -$200 paper P&L, halting all new
  trades and training data collection. Now disabled in paper mode.
- **Backfilling 200+ pre-instrumentation trades.** Rejected — funding rate/OI/CVD/basis
  history not reliably reconstructable.
- **TTL == refresh interval (both 300s).** Any drift caused key expiry before renewal.
  Fixed via TTL=600s + overlapping refresh.

---

## Files Touched / Created

### This session (2026-05-23)

| File | Change |
|------|--------|
| `main.py` | `_extract_strike`: `floor_strike` primary path with `> 0` guard. Replaced `MAX_POSITIONS_PER_TICKER=3` with `MAX_POSITIONS_PER_TICKER_PER_SIDE=2`. Per-side cap gate in `_process_market`. `MIN_ENTRY_PRICE_CENTS=20` floor added then removed (see What Failed). Circuit breaker drawdown disabled in paper mode. |
| `btc_kalshi_system/portfolio/monitor.py` | Added `ticker_direction_count(ticker, direction) -> int` (Redis-backed with in-memory fallback). |
| `btc_kalshi_system/portfolio/circuit_breaker.py` | Daily drawdown check moved inside `if not self._paper_trading` block. |
| `scripts/monitor_trades.py` | **NEW.** Live SQLite polling monitor (15s interval) — prints new trades, resolutions, running record, P&L, training progress bar. |
| `scripts/win_rate_by_price.py` | **NEW.** Win rate / P&L breakdown by entry price bucket. `--bucket`, `--dir yes/no`, `--min-trades` flags. |

### Prior sessions (2026-05-22 — 2026-05-23)

| File | Change |
|------|--------|
| `btc_kalshi_system/data/derivatives_feed.py` | Coinglass + Kraken fallbacks, LKG key write, TTL=600s, overlapping 240s refresh, `_funding_rate_trend` zero fallback. |
| `btc_kalshi_system/signal/fusion.py` | `TradingSignal` carries `regime_features` + `features_stale`. Gate 2 soft-launch. LKG stale detection. |
| `main.py` | LKG fallback in `_get_market_context`. Loguru `enqueue=True, catch=True`. `_regime_watchdog`. 23-column schema. Idempotent ALTER TABLE. |
| `config.py` | `REGIME_MODEL_PATH`, `REGIME_GATE2_ENFORCING`, `COINGLASS_API_KEY`. |
| `scripts/train_regime.py` | Feature variance gate, 3-fold walk-forward CV, feature importance logging, `--max-rows`. |
| `scripts/regime_health_check.py` | **NEW.** Diagnostic: training progress, staleness rate, feature variance, model health. |
| `scripts/auto_retrain.py` | **NEW.** Cron-driven retraining with emergency/row/time triggers. |

---

## Next Steps

1. **Monitor training-ready row accumulation.** Run `python3 scripts/regime_health_check.py`
   daily. Currently at 132/500 (~7.5 days to go at 49/day). Confirm one process running:
   `ps aux | grep "[Pp]ython.*main\.py"`.

2. **Watch sub-20¢ win rate.** Run `python3 scripts/win_rate_by_price.py` periodically.
   Currently 0/6 wins at 0–19¢ ($-19/trade avg). If this stays at 0% after 20+ trades,
   re-add `MIN_ENTRY_PRICE_CENTS=20` to `main.py`. If it climbs above 35%, Kronos has
   genuine contrarian edge there and the floor should stay removed.

3. **Consider a CVD soft gate (early regime signal).** Add a check in `_process_market`:
   if `cvd_normalized < -0.3` skip YES→UP; if `cvd_normalized > +0.3` skip NO→DOWN.
   This is a preview of Gate 2. Would have prevented most of today's losses. Discuss
   before implementing — it reduces training data diversity.

4. **Add auto_retrain to crontab.** Copy the crontab line from the top of
   `scripts/auto_retrain.py`. Test first with `python3 scripts/auto_retrain.py --dry-run`.

5. **Train the model when ready.** `python3 scripts/train_regime.py --dry-run` previews
   Brier / accuracy. If sane (Brier < 0.25, Kronos agreement > 55%), re-run without
   `--dry-run` to save `models/regime.pkl`. Restart — it auto-loads. Then flip
   `REGIME_GATE2_ENFORCING=true` after ~50 shadow trades.

---

## Context / Gotchas

- **Test suite invariant: 207 pass.** Run from project root: `python3 -m pytest`.
- **Check for stale processes before starting.** `ps aux | grep "[Pp]ython.*main\.py"` —
  if more than one Python process appears, `kill -9` all but the newest. `pgrep -af main.py`
  returns false positives (matches shell wrappers). Use `ps aux` for reliability.
- **Per-side cap is Redis-backed.** `ticker_direction_count` reads `portfolio:open_positions`
  hash directly. Accurate across all processes. Do not revert to in-memory count.
- **No entry price floor currently active.** `MIN_ENTRY_PRICE_CENTS` was added then removed.
  Sub-20¢ trades are allowed. Monitor win rate via `python3 scripts/win_rate_by_price.py`.
- **CVD does NOT influence Kronos decisions in bootstrap mode.** CVD is logged to SQLite
  for regime model training only. Kronos fires on Monte Carlo (BRTI candle price momentum).
  This is why CVD divergence causes multi-trade loss streaks. The regime model (Gate 2) is
  the fix.
- **`floor_strike` is set by Kalshi at market open.** It equals the BRTI average at open.
  For KXBTC15M markets it is always non-zero once the market opens. The `> 0` guard only
  catches markets polled before open. BRTI candle fallback now logs a WARNING when hit.
- **Label = `int(direction == outcome)`, NOT `outcome`.** `outcome` = did trade win.
  For NO→DOWN wins, `outcome=1` but `direction=0`, so label = 0. All evaluation uses this.
- **Two `brti_volatility_1h` implementations exist.** `DerivativesFeed` (Redis ticks) vs
  `fusion._regime_features()` (5-min OHLCV pct_change). Persisted column is fusion version.
  Do not consolidate after model training begins — it invalidates trained models.
- **Kronos preload rule.** Apple Silicon segfault avoidance: preload Kronos in
  `KronosV2.__init__()` before asyncio, `map_location="cpu"`, `set_num_threads(1)` BEFORE
  `from_pretrained`. Do not refactor.
- **Gate 2 starts in SHADOW mode after loading a model.** Set `REGIME_GATE2_ENFORCING=true`
  only after observing ~50 trades. Default `false`.
- **LKG sentinel `_lkg=True` in market context dict.** Never add `_lkg` or `_lkg_written_at`
  to the 6-feature list — it will corrupt XGBoost model inputs.
- **`dump.rdb` and `trades.db.bak.*` must NOT be committed.** Stage code files explicitly.
- **Calibrator is independent.** Uses only `kronos_raw + outcome`, not regime features.
  Hits its 500-sample threshold separately from the regime model.
- **Gate 6 is skipped for `timeframe == "15min"`.**
- **RSA-PSS Kalshi signing, sign path-only.**
- **Coinglass fallback requires `COINGLASS_API_KEY` in `.env`.** Without it, logs WARNING
  and returns zeros. Kraken fallback uses spot BTC/USD — no API key required.
- **`RANGING_SHRINK=0.7`, `_BOOTSTRAP_SHRINK=0.8`, `_UNCERTAINTY_SHRINK=0.5`** in
  `fusion.py`. Do not equate bootstrap and uncertainty shrinks.
- **DeepSeek `NEUTRAL_DEFAULT` on 402, not `SAFE_DEFAULT`.**
