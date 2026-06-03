# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min up/down markets). Forecast direction via Kronos + XGBoost regime classifier + DeepSeek gate, size with fractional Kelly, run 7 pre-trade gates.

**Current focus:** Session 29 complete. WebSocket orderbook cache shipped — `KalshiOrderbookFeed` eliminates ~300ms REST round-trip on every `get_orderbook()` call. 480 tests pass. **Next:** monitor WS hit rate in production logs, then continue accumulating candle_features rows for regime v2 retrain (~June 6–7).

---

## Current Progress

**As of 2026-06-03 session 29: WebSocket orderbook cache. 480 tests pass.**

**Goal:** Cut ~300ms from every trade entry by reading orderbook state from a local WebSocket cache instead of making a REST call on each `get_orderbook()` invocation.

**Changes — session 29:**

| File | Change | Status |
|------|--------|--------|
| `btc_kalshi_system/execution/kalshi_orderbook_feed.py` | **New**: `KalshiOrderbookFeed` — maintains one `OrderbookManager` per active ticker via `pykalshi.AsyncFeed` on the `orderbook_delta` channel. `run()` connects and iterates forever (reconnects handled internally by `AsyncFeed.__aiter__`). `update_subscriptions(tickers)` is thread-safe (called from `asyncio.to_thread`). `get_orderbook(ticker)` returns cached data in `orderbook_fp` format or `None` if: not subscribed, no snapshot yet, or last update > 10s old. `_drain_loop` runs every 1s: detects reconnects (resets `has_snapshot` flags so stale data isn't served), then calls `_apply_drain(feed)` to subscribe/unsubscribe based on diff of `_desired_tickers` vs `_subscribed`. `count_rest_fallback()` called by router when WS returns None. Logs WS vs REST read stats every 100 calls. | ✅ |
| `btc_kalshi_system/execution/router.py` | Added optional `orderbook_feed=None` param to `__init__`. `get_orderbook()` hits WS cache first; falls back to REST with DEBUG log + `count_rest_fallback()` call. No behavioral change when `orderbook_feed=None`. | ✅ |
| `main.py` | Creates `self._orderbook_feed = KalshiOrderbookFeed(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)`. Passes it to `KalshiClientRouter(orderbook_feed=...)`. Adds `self._orderbook_feed.run()` to `asyncio.gather()`. Calls `self._orderbook_feed.update_subscriptions(active_tickers)` in `_run_cycle()` after `_get_active_markets()`, before the market loop. | ✅ |
| `tests/execution/test_kalshi_orderbook_feed.py` | **New**: 23 TDD tests covering: None before snapshot, None when stale (>10s), None for unknown ticker, correct `orderbook_fp` format + ascending order, snapshot+delta quantity update, level removal on zero qty, `_parse_orderbook` compatibility, `update_subscriptions` diffing, thread safety, `_apply_drain` subscribe/unsubscribe/cleanup, reconnect snapshot reset, desired-tickers preserved across reconnect. | ✅ |
| `tests/execution/test_router.py` | 4 new tests: WS cache used when available, REST fallback when WS returns None, `count_rest_fallback` called on fallback, no-feed path unchanged. | ✅ |

**What Worked:**
- `pykalshi.AsyncFeed.__aiter__` handles reconnect internally — no custom retry loop needed in `run()`.
- `_TickerState` dataclass (manager + has_snapshot + last_update_ts) is the cleanest unit of per-ticker state.
- Resetting `has_snapshot=False` on reconnect (via `reconnect_count` delta in drain loop) prevents serving stale data after a WS drop.
- `_apply_drain(feed)` extracted as a sync helper — drain loop calls it, tests can call it directly without async complexity.
- `threading.Lock` on `_books` and `_desired_tickers` — event loop holds it only briefly (dict ops), no blocking risk.

**What Failed / Gotchas:**
- `asyncio.create_task()` in `AsyncFeed.subscribe()` is NOT safe to call from a thread — so `update_subscriptions()` only writes to `_desired_tickers` (thread-safe), and the drain loop (event loop) calls `feed.subscribe()` from within the async context.
- `AsyncFeed` is NOT created with `AsyncFeed(client)` directly — use `client.feed()` which returns `AsyncFeed(self)`.
- The channel name for subscription is `"orderbook_delta"` — both snapshots AND deltas arrive on this channel. Distinguish via `isinstance(msg, OrderbookSnapshotMessage)`.
- `OrderbookManager.yes` / `.no` are plain dicts (no guaranteed order) — must `sorted(..., key=lambda x: float(x[0]))` before returning levels.
- `_subscribed` is event-loop-only (no lock needed); `_desired_tickers` and `_books` are cross-thread (lock required).

**Files Touched / Created:**
- **New**: `btc_kalshi_system/execution/kalshi_orderbook_feed.py`
- **New**: `tests/execution/test_kalshi_orderbook_feed.py`
- **Modified**: `btc_kalshi_system/execution/router.py`
- **Modified**: `main.py`
- **Modified**: `tests/execution/test_router.py`
- **Auto-updated**: `models/calibrator_last_trained.json` (auto-retrain cron fired)

**Next Steps:**
1. **Monitor WS hit rate** in production logs: `grep "WS orderbook cache hit\|falling back to REST" logs/kronos_*.log`. After 100 `get_orderbook()` calls, an INFO line logs `WS=N REST=M`. Target: REST calls only on startup (before first snapshot) and brief reconnect windows.
2. **Verify no REST fallback after ~30s of uptime**: the first `update_subscriptions()` call queues subscriptions; the drain loop fires within 1s; the server sends a snapshot within ~1–2s. After that, WS should serve 100% of reads.
3. **Regime v2 retrain** (~June 6–7): `python3 scripts/train_regime.py --dry-run`. Needs ~670 `candle_features` rows with `kronos_raw_15min` populated.
4. **After regime retrain**: flip `_KRONOS_WEIGHT=0.2` / `_REGIME_WEIGHT=0.8` in `fusion.py`, remove disagreement neutralization block.
5. **Consider wrapping `_orderbook_feed.run()` in an outer retry loop** if production shows it crashing the gather on auth failures — currently `AsyncFeed.__aiter__` handles network reconnects but not auth errors (which would propagate up and cancel all tasks).

**Context / Gotchas:**
- `_parse_orderbook()` in `main.py` is unchanged — the WS output format (`orderbook_fp` with ascending `yes_dollars`/`no_dollars` lists) must exactly match what it expects. Both fetches in `_process_market` (T+0 for feature injection, T+1 for freshness check before order) now go through the WS cache path.
- `_run_cycle()` runs via `asyncio.to_thread` — `update_subscriptions()` must be thread-safe. It is.
- `KalshiOrderbookFeed.run()` is a long-lived coroutine added to `asyncio.gather()`. If it raises (e.g., key file missing), it cancels all other tasks. The `AsyncFeed.__aiter__` handles network errors internally; startup errors (bad key) would propagate. Config must be valid before launch.
- Test helper `_inject_ticker()` directly manipulates `_books` — this is intentional to avoid async complexity in unit tests. All behavior under test is synchronous (state reads/writes).

---

**As of 2026-06-01 session 28: auto_retrain_regime.py built (TDD). Deploy gate weights corrected. 453 tests pass.**

**Changes — session 28:**

| File | Change | Status |
|------|--------|--------|
| `scripts/auto_retrain_regime.py` | **New**: weekly cron (Sun 3am) that trains a candidate RegimeModel in-process and saves only when the candidate strictly beats the deployed model on a shared holdout. Triggers: +200 rows or 14 days elapsed. Min rows: 672. Rolling window: 2000. **Holdout guard**: loads last 2000 qualifying rows sorted by `candle_ts`; trains candidate on rows `[:-100]`; evaluates both candidate and deployed model on the same `[-100:]` rows (Brier score); saves only if `candidate_brier < deployed_brier`. The deployed model was trained on an older window — those 100 rows are genuinely unseen by both, so it's an apples-to-apples out-of-sample comparison at the current data edge. Key functions: `get_qualifying_count`, `load/save_marker`, `brier_score`, `evaluate_deployed_model`, `should_retrain`, `should_deploy`. | ✅ |
| `scripts/train_regime.py` | **Fix deploy gate weights**: was using `0.4 × k15 + 0.6 × regime` (old session 26 weights); corrected to `0.2 × k15 + 0.8 × regime` to match target weight split. | ✅ |
| `tests/test_auto_retrain_regime.py` | 25 new tests written TDD (red before green). All pass. | ✅ |
| crontab | Weekly retrain cron: `0 2 * * 1` (Monday 2am) — model is fresh before Monday volume pickup. Originally Sunday 3am, moved so the weekend's accumulated data trains and deploys before the week's highest-volume session. | ✅ |

---

**As of 2026-06-01 session 27: Phase 1 + Phase 2 infrastructure shipped. 428 tests pass.**

**Changes — session 27:**

| File | Change | Status |
|------|--------|--------|
| `btc_kalshi_system/signal/fusion.py` | **Bootstrap regime-aware shrink**: replaced flat `_BOOTSTRAP_SHRINK=0.8` with regime-aware lookup (`ranging=0.35`, `high_uncertainty=0.50`, `trending_up/down=0.80`). Formula unchanged: `combined = 0.5 + (kronos_cal - 0.5) * shrink`. Reduces Kelly damage in ranging without inverting direction. | ✅ |
| `btc_kalshi_system/models/calibrator.py` | **Direction sanity guard in `fit()`**: before deploying any new model, checks `transform(0.75, regime="trending_up") >= 0.5`. If the calibrator learned to invert momentum signals (as happened with the inverted auto-retrain), it reverts to passthrough instead of deploying. Guards both regime-aware and non-regime-aware fits. | ✅ |
| `btc_kalshi_system/models/regime_model.py` | **`_FEATURE_ORDER` expanded to 28 features**: added `"kronos_raw_15min"` and `"kronos_raw_5min"` at the end. `get_regime()` converts `None` → `float('nan')` for bootstrap rows where Kronos hasn't fired yet (XGBoost treats NaN as missing natively). | ✅ |
| `btc_kalshi_system/signal/fusion.py` | **Fusion engine caches Kronos raw values**: `self._last_kronos_raw_15min` and `self._last_kronos_raw_5min` set on every `get_signal()` call; exposed via `_regime_features()` so the candle logger gets them automatically through `get_features_snapshot()`. | ✅ |
| `main.py` | **`_CANDLE_FEATURES_COLUMN_MIGRATIONS`**: added `("kronos_raw_15min", "REAL DEFAULT NULL")` and `("kronos_raw_5min", "REAL DEFAULT NULL")`. Migrations applied manually to production DB before restart. | ✅ |
| `trades.db` | **Backfill**: 192/193 existing `candle_features` rows updated with Kronos values from `gate_rejections` nearest-timestamp join (±15 min window). 1 row unmatchable (no gate rejection within window). | ✅ |
| Tests | 8 new tests across `test_regime_model.py`, `test_feature_order.py`, `test_main_candle_logger.py`, `test_fusion.py`, `test_fusion_deribit_features.py`. 428 total pass. | ✅ |

**Why fusion caches Kronos (not candle_logger patching):**
The `_regime_features()` method must return exactly the features in `_FEATURE_ORDER` for the `test_feature_order_all_three_match` consistency check to pass. Having the fusion engine cache and expose Kronos values keeps the 3-file contract clean (fusion → regime_model → train_regime all stay in sync). The candle logger requires zero changes.

**System state after session 27:**
- Calibrator: passthrough (forced session 26, direction guard prevents recurrence)
- Bootstrap shrink: `ranging=0.35×`, `high_uncertainty=0.50×`, `trending_up/down=0.80×`
- `candle_features`: 193 rows, 192 with `kronos_raw_15min` populated; ~96/day accumulating
- ETA for 670 rows with Kronos: ~June 6–7 (4–5 days from session 27)
- Service restarted at 2026-06-02 21:35 UTC with new code; first live-logged Kronos row next candle close (~04:30 UTC)
- All-time P&L: -$196.75 (as of session 26)

**Next: regime v2 retrain checklist (~June 6–7)**
```
python3 scripts/train_regime.py --dry-run
```
Deploy if ALL pass:
- [ ] CV Brier < 0.25
- [ ] Accuracy > 55% on holdout
- [ ] `kronos_raw_15min` importance > 0% (model uses Kronos)
- [ ] `kronos_raw_15min` importance < 30% (not just re-learning Kronos)

After deploy:
- [ ] `_KRONOS_WEIGHT = 0.2`, `_REGIME_WEIGHT = 0.8` in `fusion.py`
- [ ] Remove disagreement neutralization block in `fusion.py` (marked "Remove after regime v2 deploys")
- [ ] Restart service

---

**As of 2026-06-02 session 26: Calibrator inversion detected + forced passthrough. Three-phase plan to get into positive EV. 417 tests pass.**

**Calibrator inversion (immediate fix):**

The auto-retrain at 1,302 rows deployed a fully inverted calibration map (neutral regime):
```
k15_raw=0.60 → k15_cal=0.3554
k15_raw=0.80 → k15_cal=0.3443   ← k15_raw=0.80 maps BELOW 0.5
k15_raw=1.00 → k15_cal=0.3327
```
Brier improved (0.2461 → 0.2443) because Kronos is genuinely anti-correlated in ranging: k15_raw > 0.80 has a 38% historical win rate. The calibrator learned a real relationship but applied it universally, inverting every YES signal into a NO trade with no safeguard for the rare trending cases. Forced passthrough immediately via direct `.pkl` edit + service restart.

**Root cause of being underwater:**

Kronos is a momentum signal. The system has operated mostly in ranging market conditions (277/668 trades ranging, 44-47% WR, -$437 drag). In ranging, strong momentum readings precede mean reversion — making high-confidence Kronos calls the worst bets. The calibrator correctly learned this but cannot apply it conditionally without regime context. The regime model is the fix; the calibrator is a symptom.

**Three-phase recovery plan:**

---

**Phase 1 — Near-term (June 2–5): Reduce damage while accumulating data**

| Change | Detail |
|--------|--------|
| **Ranging shrink in `fusion.py` (bootstrap path)** | Replace flat `0.8` bootstrap shrink with regime-aware lookup: `ranging=0.35`, `high_uncertainty=0.50`, `trending_up/down=0.80`. Formula: `combined = 0.5 + (kronos_cal - 0.5) * shrink`. Reduces Kelly on bad-regime signals without inverting direction. Bootstrap path disappears when regime v2 deploys — no cleanup needed. |
| **CVD Gate 7 — enforce as hard gate** | First query shadow rows: `SELECT COUNT(*), AVG(outcome) FROM gate_rejections WHERE shadow=1 AND outcome IS NOT NULL`. If ≥ 20 resolved rows and accuracy ≥ 70% when CVD opposes, flip from shadow to hard gate. |
| **Calibrator direction guard** | Add to `fit()` before any save: `if self.transform(0.75, regime="trending_up") < 0.5: force passthrough, return`. The calibrator may compress; it cannot invert. Prevents recurrence of this session's inversion. |

---

**Phase 2 — Regime v2 as meta-learner (~June 6–7): Regime becomes primary signal**

**Why regime as primary:** The regime model trained on `candle_features` with `btc_direction = close > open` learns to predict BTC direction from Deribit options flow, CVD, funding, fear/greed — and now Kronos itself. With k15_raw as a feature, it answers: *"given Kronos says X and market microstructure says Y, what's the actual direction?"* That's the calibration problem solved with far more context. No circularity: `candle_features` logs every 15-min candle regardless of trading.

| Change | Detail | Status |
|--------|--------|--------|
| **Add `kronos_raw_15min`, `kronos_raw_5min` to `candle_features`** | Schema migration applied; fusion engine caches and exposes Kronos raw values via `_regime_features()` so candle logger writes them automatically. | ✅ Done (session 27) |
| **Backfill 192/193 existing rows** | One-time SQL UPDATE using nearest gate_rejection timestamp within ±15 min window. | ✅ Done (session 27) |
| **Add both to `_FEATURE_ORDER` in `regime_model.py`** | `train_regime.py` auto-syncs via `_FEATURE_COLS = list(_FEATURE_ORDER)`. XGBoost treats NULL→NaN as missing natively. | ✅ Done (session 27) |
| **Fusion weight flip** | `_KRONOS_WEIGHT = 0.2` (was 0.8), `_REGIME_WEIGHT = 0.8` (was 0.2). Regime becomes primary signal; Kronos kept as weak sanity check. If regime v2 performs well after 2 weeks, evaluate going 0.0/1.0. | ⏳ After retrain |
| **Remove disagreement neutralization** | Neutralization block in `fusion.py` (marked "Remove after regime v2 deploys"). | ⏳ After retrain |
| **Retrain gates before deploying** | CV Brier < 0.25 · Accuracy > 55% · `kronos_raw_15min` importance > 0% and < 30% | ⏳ ~June 6–7 |

**candle_features timeline:** 193 rows as of June 2 session 27, 192 with Kronos populated, accumulating ~96/day, need ~670 with `kronos_raw_15min` → ETA June 6–7.

---

**Phase 3 — Post-deploy monitoring (2 weeks after regime v2)**

- **WR > 53%**: regime model working — consider shifting weights to 0.3/0.7 k15/regime
- **WR flat**: investigate feature importances — ranging shrink may be doing the work, not regime model
- **Calibrator retrain**: once 300+ rows accumulate under new 0.2/0.8 signal, run `python3 scripts/train_calibrator.py --dry-run`. Check compression map: `transform(0.80, regime="trending_up")` must be > 0.5 before allowing save.
- **Evaluate going 0.0/1.0**: after 2 weeks of production data under 0.2/0.8, query whether Kronos's 0.2 weight is adding or subtracting. If regime is more confident AND more accurate than Kronos on resolved trades, drop Kronos entirely (`_KRONOS_WEIGHT = 0.0`).

**Current system state (June 1, updated session 28):**
- Calibrator: passthrough (direction guard prevents redeployment of inverted model)
- Bootstrap shrink: regime-aware (`ranging=0.35×`, `high_uncertainty=0.50×`, `trending=0.80×`)
- `candle_features`: 193 rows, 192 with kronos_raw_15min populated (backfill complete)
- All-time P&L: **-$196.75** (668 trades, 51.8% WR)
- Service running with Phase 1 + Phase 2 infrastructure live; 453 tests pass
- `auto_retrain_regime.py` cron active: Sunday 3am weekly retrain with holdout guard

---

**As of 2026-06-01 session 25: P&L fix, Kalshi decoupling, train_regime.py rewrite. System in bootstrap mode. 417 tests pass.**

**Changes — session 25:**

| File | Change | Status |
|------|--------|--------|
| `btc_kalshi_system/portfolio/monitor.py` | **P&L convention restored**: NO branch uses `outcome==1` for WIN (was incorrectly changed to `outcome==0` in a prior session). WIN formula: `+contracts * (100 - fill_price_cents) / 100`. LOSS formula: `-contracts * fill_price_cents / 100`. | ✅ |
| `scripts/monitor_trades.py` | Reverted 4 places (get_stats, get_streak, print_last5, resolution alert) from direction-aware WIN counting back to simple `outcome==1`. | ✅ |
| `trades.db` | All 351 NO trades recalculated with correct `pnl_dollars` via SQL UPDATE. True all-time P&L: **-$192.55** (the previous +$2,000 figure was fabricated by inverted P&L formula). | ✅ |
| `btc_kalshi_system/execution/pretrade_checklist.py` | **Reverted sub-20c NO exemption**: Gate 8/8b now apply unconditionally to all trades. Sub-20c NO setup was 0W/10L (−$146.30) — the prior 32W/0L was based on the inverted win-counting formula. | ✅ |
| `btc_kalshi_system/models/regime_model.py` | **Removed `kalshi_implied_prob` and `kalshi_spread_normalized`** from `_FEATURE_ORDER`. Now 26 features. Kalshi was feature #1 at 19% importance — creating circularity: Kalshi → regime_prob (20% of combined signal) → Gate 5 edge vs Kalshi price → Gate 8 Kalshi consensus block. | ✅ |
| `btc_kalshi_system/signal/fusion.py` | Both Kalshi features still computed (for other uses) but excluded from the returned `_regime_features()` dict. Comment added: `# kalshi_implied_prob and kalshi_spread_normalized intentionally excluded`. | ✅ |
| `scripts/train_regime.py` | **Complete rewrite**: training source changed from `trades` table to `candle_features` table. Label changed from circular `(direction == outcome)` to true `btc_direction` (close > open) computed at every 15-min candle close regardless of whether a trade was placed. `_FEATURE_COLS = list(_FEATURE_ORDER)` auto-syncs. Min rows: 672 (7 days). Removed gate_rejections inclusion and Kronos agreement metric. | ✅ |
| `scripts/regime_health_check.py` | Removed `AND kalshi_implied_prob IS NOT NULL` from `_FRESH_FILTER`. | ✅ |
| `models/regime.pkl` | **Deleted**. System now in bootstrap mode: `combined = 0.5 + (kronos_cal - 0.5) * 0.8`. | ✅ |
| `tests/` (multiple) | Updated count assertions (28→26), removed kalshi entries from feature fixtures, fixed `test_kalshi_spread_*` to assert key absent not value zero. 417 tests pass. | ✅ |

**Why these changes matter:**

*P&L fix:* A prior session inverted the NO branch — `outcome==0` treated as WIN for NO trades. This made 10 losing sub-20c NO trades appear as 10 wins, and reported all-time P&L as +$2,000. True P&L is -$192.55.

*Circularity break:* `kalshi_implied_prob` was the #1 regime feature at 19% importance. This meant Kalshi price already influenced `combined_prob` (via regime at 20% weight), then `combined_prob` determined the edge vs Kalshi price (Gate 5), then Kalshi price blocked trades (Gate 8). The same input appeared at every stage. Removing it makes the regime model a truly independent signal.

*Label fix:* Training on `direction==outcome` ("did Kronos get it right?") only on Kalshi-gate-filtered trade samples is doubly circular — the label depends on gate decisions and the features being predicted. `btc_direction = close > open` is clean ground truth on every 15-min candle.

**Real P&L summary (as of session 25):**
- All-time: **-$192.55** (346W / 321L, 51.9% WR)
- Avg win: $7.02 | Avg loss: $8.17 — negative edge ratio
- Biggest drag: ranging regime (277 trades, 44-47% WR, -$437 total)
- W21 (-$166.84): destroyed by May 25 sub-20c NO losses (-$108) and May 28 high_uncertainty YES collapse (-$78)

**Current system state:**
- Bootstrap mode: `combined = 0.5 + (kronos_cal - 0.5) * 0.8` (Kronos-only signal, 0.8× shrink)
- `candle_features` table: ~172 rows (started May 31), ~143 clean (features_stale=0)
- ETA for 672 rows: ~June 6–7 → then run `python3 scripts/train_regime.py`
- After retrain: remove disagreement neutralization from fusion.py, restore `_REGIME_WEIGHT=0.4`

**Next milestones:**
1. **Regime v2 retrain** (~June 6–7): `python3 scripts/train_regime.py --dry-run` first, then save + restart.
2. **After retrain**: remove disagreement neutralization block in `fusion.py` (comment says "Remove after regime v2 deploys"). Restore `_REGIME_WEIGHT=0.4`.
3. **Gate 4 (rolling edge circuit breaker)**: currently hardcoded `True` in paper mode (`main.py` line 709). Consider enabling with a loose threshold to get signal about edge quality.
4. **Ranging regime gate**: -$437 drag from 277 trades at 44-47% WR. Consider tightening Gate 3 or adding a new gate to reduce sizing in ranging.

---

**As of 2026-05-31 session 24 (part 2): Remove spurious features_stale=0 from calibrator training. Training-ready rows: 883→998. 416 tests pass.**

**Changes — session 24 part 2:**

| File | Change | Status |
|------|--------|--------|
| `scripts/train_calibrator.py` | Removed `AND features_stale = 0` from both `_UNION_QUERY` (trades leg) and `_COUNT_QUERY`. `features_stale` flags bad regime model features, not bad `kronos_raw_15min` or outcome labels. The calibrator only trains on those two columns — stale flags were incorrectly excluding valid training rows. gate_rejections leg was already correct (no filter). Training-ready count: 883 → 998. | ✅ |
| `main.py` | Same fix in `_refit_calibrator()`: removed `AND features_stale=0` from the trades leg of both the k15 primary UNION query and the k5 fallback UNION query. `aged_out=0` and `shadow=0` on gate_rejections untouched. | ✅ |

**Why this matters:** The calibrator's training signal is solely `(kronos_raw_15min, outcome)`. A stale feature set means the regime model had bad inputs — the Kronos MC result and the market outcome are completely independent of Redis feature freshness. Excluding stale rows was throwing away ~115 valid training examples (the gap between 883 and 998).

**Dry-run verification (post-fix):**
```
Training-ready rows: 998 available, using 300
Pre-retrain Brier:  0.2416
Post-retrain Brier: 0.2415  (marginal improvement; main gain is window is now 30% richer)
Compression map (neutral regime):
  0.80 → 0.5807
  1.00 → 0.6416
```

---

**As of 2026-05-31 session 24 (part 1): Calibrator shadow filter fix, Gate 8 high_uncertainty tightening, Gate 11 passthrough warning. 416 tests pass.**

**Changes — session 24 (part 1):**

| File | Change | Status |
|------|--------|--------|
| `main.py` | **Fix double-counting in `_refit_calibrator()`**: Both UNION legs (k15 primary and k5 fallback) against `gate_rejections` now include `AND shadow = 0`. Gate 7 shadow rows (`shadow=1`) are real trades that also exist in `trades.db` — including them counted those outcomes twice in calibrator training. The schema comment already said "Any training query against gate_rejections MUST filter WHERE shadow = 0" but the implementation was missing the filter. | ✅ |
| `btc_kalshi_system/execution/pretrade_checklist.py` | **Gate 8 regime-aware**: `high_uncertainty` now uses flat `gate8_base = 0.05` regardless of signal_confidence. Calibrator compression collapses `calibrated_prob` toward 0.5 in this regime → signal_confidence ≈ 0 → all high_uncertainty trades fell into the lowest tier (0.10), making tiers a no-op. Flat 0.05 ≈ half of Gate 5's 8% floor, weighting Kalshi more heavily (54.9% accuracy vs Kronos 18.4% in losing periods). OI squeeze still applies (÷4 → 0.0125 effective). | ✅ |
| `btc_kalshi_system/execution/pretrade_checklist.py` | **Gate 11 calibrator warning**: 7-line comment above Gate 11 explaining that `kronos_calibrated == kronos_raw` in passthrough mode, and that once the calibrator activates and compresses k15_raw=0.80 → k15_cal≈0.55, Gate 11 will silently deactivate. No logic change. | ✅ |
| `tests/execution/test_pretrade_checklist.py` | **2 new Gate 8 high_uncertainty tests**: `test_gate8_high_uncertainty_uses_flat_005_threshold` (prob=0.89, opposing=0.06 > 0.05 → blocks, even though neutral would use threshold=0.25); `test_gate8_high_uncertainty_same_signal_passes_in_neutral` (control: same signal in neutral → threshold=0.25, opposing=0.06 < 0.25 → passes). 416 tests pass. | ✅ |

**Why Gate 8 flat threshold for high_uncertainty:**
- Calibrator compression forces `calibrated_prob` → ~0.5 in high_uncertainty → `signal_confidence ≈ 0` → always lowest tier (0.10) — tiers were a no-op
- Kalshi accuracy in this regime (54.9%) > Kronos (18.4%) — tighter threshold forces higher Kalshi alignment
- 0.05 ≈ half of Gate 5's 8% edge floor; together they require Kronos edge AND Kalshi near-agreement

**Why the calibrator shadow=0 filter matters:**
- Gate 7 shadow rows (`shadow=1`) are inserted when CVD opposes direction but the trade proceeds. The same trade is also in `trades.db` with a resolved outcome.
- Without `AND shadow=0`, the UNION included both the shadow row and the trades row for the same outcome — double-counting in calibrator training.

**Next milestones:**
1. **Monitor calibrator compression loosening.** Run `python3 scripts/train_calibrator.py --dry-run`. Watch k15_raw=0.80 → k15_cal rising above ~0.65 (`← loosening!`). Currently 0.5859.
2. **Monitor Gate 8 high_uncertainty blocks.** `SELECT COUNT(*), AVG(outcome) FROM gate_rejections WHERE failed_gate=8 AND deepseek_regime='high_uncertainty' AND outcome IS NOT NULL`.
3. **Regime v2 retrain** after 7+ days of `candle_features` (~672 rows, ETA ~June 7).

---

**As of 2026-05-31 session 23 (post-session): Removed high_uncertainty Kelly shrink. Calibrator regime encoding differentiated (ranging=0.3, high_uncertainty=−0.3). Calibrator retrained: Brier 0.2467→0.2449. Compression map added to train_calibrator.py. 411 tests pass.**

**As of 2026-05-31 session 23: Gate 11 + regime weight 0.4→0.2 + disagreement neutralization + router test fixes. 409 tests pass (395 + 14 new/fixed). ✅ Committed d229d1c. ✅ Service restarted.**

**Session 23 (post-session): high_uncertainty Kelly shrink removal, calibrator encoding + retrain**

**Changes — session 23 post-session:**

| File | Change | Status |
|------|--------|--------|
| `btc_kalshi_system/execution/pretrade_checklist.py` | **Removed high_uncertainty Kelly shrink**: the block that halved Kelly dollars + contracts when `deepseek_regime == "high_uncertainty"` is gone. Empirical reason: Gate 5 (regime-aware edge floor) and Gate 3 (high_uncertainty + thin edge) already apply the regime penalty. The Kelly shrink was double-penalizing high_uncertainty signals. Blocked high_uncertainty trades had 68.4% WR vs 45.2% WR for taken trades — the shrink was blocking the stronger signals. Tests updated: `test_high_uncertainty_kelly_same_as_neutral`, `test_high_uncertainty_kelly_contracts_same_as_neutral`, `test_high_uncertainty_strong_k15_no_longer_blocked_by_shrink`. | ✅ |
| `btc_kalshi_system/models/calibrator.py` | **Regime encoding differentiated**: `ranging` 0.0→0.3 (53.7% historical WR, mildly trustworthy); `high_uncertainty` 0.0→−0.3 (45.2% historical WR, mildly penalised). Previously both mapped to 0.0 — the calibrator could not distinguish them, compressing all mid-range signals identically. Effect: k15_raw=0.80 → k15_cal now 0.5859 (was 0.5511); k15_raw=0.90 → 0.6069; k15_raw=1.00 → 0.6270. Brier improved 0.2467→0.2449 after retrain with new encoding. | ✅ |
| `scripts/train_calibrator.py` | **Compression map output**: after every fit, prints `k15_raw → k15_cal` for checkpoints [0.60, 0.70, 0.80, 0.90, 1.00] using `regime="trending_up"`. Flag prints `← loosening!` when k15_cal > 0.65 at k15_raw ≥ 0.80 — key signal that calibrator is starting to trust strong signals. **Metadata auto-write**: `calibrator_last_trained.json` now written automatically after every real save (not on `--dry-run`). Previously required manual write. | ✅ |
| `tests/models/test_calibrator.py` | **2 new calibrator encoding tests**: `test_ranging_and_high_uncertainty_encoded_differently` (asserts `ranging=0.3`, `high_uncertainty=-0.3`, unknown=0.0). `test_transform_regime_unknown_uses_zero_encoding` updated: now asserts `result_hu <= result_unknown <= result_ranging` (unknown=0.0 sits between ranging=0.3 and high_uncertainty=−0.3). Total: 411 tests pass. | ✅ |

**Calibrator compression map after session 23 retrain (trending_up regime):**
```
k15_raw → k15_cal
  0.60 → 0.5345
  0.70 → 0.5614
  0.80 → 0.5859
  0.90 → 0.6069
  1.00 → 0.6270
```
Watch k15_raw=0.80 → k15_cal: if it rises above ~0.65, calibrator is loosening (more data is teaching it to trust strong signals). Currently still well below — system has moderate data; compression will ease as more regime-diverse rows accumulate.

**Why remove the high_uncertainty Kelly shrink:**
- Gate 3 already blocks thin-edge high_uncertainty signals (`edge_from_center < 0.05`)
- Gate 5 already requires 8% edge floor for high_uncertainty (vs regime-specific baseline)
- Gate 8/8b apply Kalshi consensus pressure regardless of regime
- Double-penalizing via Kelly shrink on top of these gates was over-filtering the 68.4% WR blocked category
- With the shrink gone, high_uncertainty signals that clear all gates reach their full Kelly size

**Session 23: Gate 11, regime weight, disagreement neutralization, router test fixes**

**Changes — session 23:**

| File | Change | Status |
|------|--------|--------|
| `btc_kalshi_system/execution/pretrade_checklist.py` | **Gate 11** (new): blocks YES trades where `kronos_calibrated > 0.75` AND `trade_price_cents < 45`. Post-May-26 data shows 15% win rate on 13 trades in this zone. Placed after Gate 2a (price floor), before Kelly computation. Constants `_OVERCONFIDENCE_K_CAL_FLOOR=0.75` and `_OVERCONFIDENCE_MAX_FILL_CENTS=45` defined locally inside `run()`. Only applies to direction=1 (YES). | ✅ |
| `btc_kalshi_system/signal/fusion.py` | **Regime weight 0.4→0.2**: `_KRONOS_WEIGHT=0.8`, `_REGIME_WEIGHT=0.2`. Regime model v1 has circular label (`direction==outcome`; `kalshi_implied_prob` is #1 feature at 19%). Restore to 0.4 after regime v2 retrains. **Disagreement neutralization**: when `kronos_cal` and `regime_prob` are on opposite sides of 0.5, `_regime_in_fusion=0.5` (neutral) is used in the fusion formula instead of raw `regime_prob`. On agreement days regime is fully preserved. Gate 2 warning still logs raw `regime_prob`; `TradingSignal.regime_prob` stores raw value. Remove neutralization after regime v2 validates. | ✅ |
| `tests/execution/test_pretrade_checklist.py` | **4 new Gate 11 tests**: `test_gate11_fires_high_kcal_low_fill_yes`, `test_gate11_does_not_fire_high_fill`, `test_gate11_does_not_fire_low_kcal`, `test_gate11_does_not_fire_no_direction`. **3 existing tests updated**: `test_gate2_depth_capped_to_available` and 2 Gate 8 regression tests moved from 29¢ → 50¢ fills (Gate 11 correctly blocks the 29¢ high-confidence scenario). | ✅ |
| `tests/signal/test_fusion.py` | **5 tests updated** for new weights (0.8/0.2). **1 test updated** for neutralization: `test_gate2_shadow_mode_does_not_block` expected changed from `0.62` → `0.66` (regime neutralized on disagreement). **4 new neutralization tests**: `test_disagreement_neutralization_bullish_kronos_bearish_regime`, `test_disagreement_neutralization_bearish_kronos_bullish_regime`, `test_disagreement_neutralization_does_not_fire_on_agreement_bullish`, `test_disagreement_neutralization_does_not_fire_on_agreement_bearish`. | ✅ |
| `tests/execution/test_router.py` | **2 stale BOTH_FAILED tests fixed**: `test_both_failed_raises_runtime_error` and `test_both_failed_does_not_call_raw` now set `_last_recovery_attempt = time.time()` to suppress the BOTH_FAILED→FALLBACK recovery transition that was silently eating the expected RuntimeError. Pre-existing failures since the session 21 router recovery commit. | ✅ |

**Gate 11 detail:**
- Fires when: `direction == 1 AND kronos_calibrated > 0.75 AND trade_price_cents < 45`
- Uses `signal.kronos_calibrated` (k15-calibrated). With passthrough calibrator = k15_raw; after calibrator activates = compressed value.
- Does NOT apply to direction=0 (NO). NO at 35¢ = YES at 65¢ — market agreeing with Kronos, different dynamics.

**Regime weight + neutralization detail:**
- Gate 2 stays shadow mode (`REGIME_GATE2_ENFORCING=False`) — unchanged.
- Neutralization lives inside the `try:` block only. Bootstrap path (`except NotTrainedError:`) unchanged.
- Effect: k15_cal=0.70, regime_prob=0.18 → `_regime_in_fusion=0.5` → combined=0.8×0.70+0.2×0.50=0.66 (vs 0.596 Task 2 alone, vs 0.522 old 0.6/0.4).
- Minimum k15 to clear Gate 5 at 50¢ fill: 0.6875 (vs 0.75 Task 2 only; vs 0.65 no regime).
- Restore `_REGIME_WEIGHT=0.4` and remove neutralization after regime v2 validates. See Future Roadmap — Phase 1b.

**Next milestones:**
1. **Monitor calibrator compression loosening.** Run `python3 scripts/train_calibrator.py --dry-run` periodically. Watch for k15_raw=0.80 → k15_cal rising above ~0.65 (flag: `← loosening!`). Currently 0.5859 — still compressed.
2. **Monitor Gate 11 rejections.** Query: `SELECT COUNT(*), AVG(trade_price_cents), AVG(k15_calibrated_prob) FROM gate_rejections WHERE failed_gate=11 AND outcome IS NOT NULL`. Confirm win rate validates the 15% historical figure.
3. **Gate 2 enforcement decision.** Run `python3 scripts/regime_confidence_tracker.py` after ~50 shadow trades.
4. **Regime v2 retrain** after 7+ days of `candle_features` (~672 rows, ETA ~June 7). New label: `btc_direction = close > open`. Drop `kalshi_spread_normalized` (circular). Restore `_REGIME_WEIGHT=0.4` after retrain. Remove disagreement neutralization after regime v2 validates. See Future Roadmap — Phase 1b.

---

**As of 2026-05-30 session 22: 15-min candle feature logger + regime confidence tracker. Infrastructure for regime label change. 401 tests pass (395 + 6 new).**

**Session 22: candle feature logger + confidence tracker**

**Changes — session 22:**

| File | Change |
|------|--------|
| `btc_kalshi_system/signal/fusion.py` | **New `get_features_snapshot()` method** on `SignalFusionEngine`: returns `(features_dict, features_stale, deribit_stale)`. Lightweight wrapper around `_regime_features()` — no MC, no calibration, no market context mutation. Safe to call from background loop. |
| `main.py` | **Import `_FEATURE_ORDER`** from `regime_model`; **`_CREATE_CANDLE_FEATURES_TABLE`** SQL constant (schema built dynamically from `_FEATURE_ORDER` to stay in sync with 3-file contract); **`_CANDLE_FEATURES_COLUMN_MIGRATIONS = []`** (empty, same pattern as others); **`candle_features` table creation + migration** called in `__init__` after gate_rejections; **`_candle_logger_loop()`** async coroutine (polls every 30s, logs `btc_direction` + regime features at each 15-min candle close, `INSERT OR IGNORE` on `candle_ts` UNIQUE); coroutine added to `asyncio.gather()` in `run()`. |
| `scripts/regime_confidence_tracker.py` | **New** — 4-section script (modeled on `regime_health_check.py`): overall regime live stats, confidence-stratified accuracy (bins by `ABS(regime_prob - 0.5)`), Gate 2 shadow disagreements detail (high/med/low confidence buckets), candle_features logger health (rows/day, stale counts, date range). `--days N` flag (default 30). |
| `tests/test_main_candle_logger.py` | **New** — 6 TDD tests: table created on init, row written on new candle, no duplicate on same candle, survives exception, `btc_direction=1` when close > open, `btc_direction=0` when close ≤ open. |

**Why 30s poll interval:** The logger fires within 30s of each 15-min close — acceptable for infrastructure that will accumulate ~672 rows/week. The `last_logged_ts` guard prevents re-logging the same candle within a single run; `INSERT OR IGNORE` on `candle_ts UNIQUE` makes it idempotent across restarts.

**candle_features schema:** `id, candle_ts (UNIQUE), btc_direction, logged_at, features_stale, deribit_stale` + all 28 `_FEATURE_ORDER` columns as REAL. Schema is built dynamically from `_FEATURE_ORDER` so it automatically stays in sync with the 3-file feature order contract.

**`btc_direction = 1 if close > open`** — uses the closed 15-min candle (second-to-last row in the 15-min OHLCV at log time). This is the new regime model v2 training label. The `candle_features` table does NOT feed into any existing training pipeline this session — data collection only.

**Restart required:** `launchctl unload && load` to pick up the new `_candle_logger_loop` and `candle_features` table. First rows will appear within 30s of the next 15-min candle close.

**Next milestones:**
0. **Calibrator 883-row retrain** (currently ~826 rows, ~1 day away). Run `python3 scripts/train_calibrator.py --dry-run` before allowing save. See memory `project-calibrator-883-retrain`.
1. **Monitor Gate 2 shadow disagreements.** Run `python3 scripts/regime_confidence_tracker.py` after ~50 trades with `regime_prob IS NOT NULL`. High-confidence calls (> 0.30 from 0.5) should have higher accuracy than low-confidence — if not, dynamic weighting is premature.
2. **After 7+ days of `candle_features`** (~672 rows): retrain regime model with `btc_direction = close > open` label and drop `kalshi_spread_normalized`. See Future Roadmap — Regime Model v2.

---

**As of 2026-05-30 session 21: Regime model trained and live. Bootstrap mode over. Gate 2 shadow mode. live_monitor shows k15raw + k15cal separately. train_regime.py now includes gate_rejections to fix selection bias.**

**Session 21: regime model deployment**

**Changes — session 21:**

| File | Change |
|------|--------|
| `scripts/train_regime.py` | **Gate rejections included by default**: `load_gate_rejections()` parses features JSON blob; merges with trades, sorts by timestamp; 7 missing Deribit+btc_24h_return features filled as NaN (XGBoost handles natively). `--trades-only` flag to revert to old behavior. Feature importances now shown in `--dry-run` mode. |
| `scripts/live_monitor.py` | Gate rejections panel now shows `k15raw` and `k15cal` as separate columns (previously only k15cal, which was misleading in passthrough mode). |
| `models/regime.pkl` | **New** — XGBoost regime model trained on 1099 rows (171 trades + 928 gate rejections). Brier=0.203 ± 0.028, Accuracy=71.3%, Kronos agreement=60%. Top feature: `kalshi_implied_prob` (19%). |

**Why gate rejections in regime model training:**
Placed trades cleared all gates — they skew toward higher-confidence signals, mid-range fill prices, and lower uncertainty. Training on trades-only means the model learns the filtered distribution, not the full market distribution. Gate rejections are ~85% of available labeled data and cover the full signal range.

**Regime model dry-run results:**
- 1099 combined rows (171 trades + 928 gate_rejections)
- CV Brier: 0.203 ± 0.026 (< 0.25 coin flip; std < 0.05 = consistent across time windows)
- CV Accuracy: 71.3%, improving fold-over-fold (66% → 71% → 74%)
- Kronos agreement: 60% — model adds independent signal, not just echoing Kronos
- No single-feature dominance: top feature `kalshi_implied_prob` at 19%
- Deribit features contributing: `iv_rv_spread` #2 at 5.5%

**What changes with regime model live:**
- Signal fusion: was `0.5 + (k15_cal - 0.5) × 0.8` (bootstrap shrink). Now: `0.6 × k15_cal + 0.4 × regime_prob`
- Gate 2 regime enforcement: `REGIME_GATE2_ENFORCING=False` — shadow mode only, logs disagreements but does NOT block trades
- Watch logs for `Gate 2 shadow` lines over ~50 trades, then evaluate whether to flip `REGIME_GATE2_ENFORCING=True`

**Next milestones:**
0. **(Session 22)** 15-min candle feature logger (`candle_features` table) + `scripts/regime_confidence_tracker.py`. Infrastructure for regime label change — data collection only, no retrain.
1. **Calibrator auto-retrain at 883 rows (~1 day, currently 826).** Run `python3 scripts/train_calibrator.py --dry-run` before allowing save.
2. After ~50 trades under regime model: audit Gate 2 shadow disagreements, decide on `REGIME_GATE2_ENFORCING`. Run `python3 scripts/regime_confidence_tracker.py` to see confidence-stratified accuracy.
3. After calibrator deploys: 2-week clean data window → proper gate audit.
4. After 7+ days of `candle_features` data (~672 rows): retrain regime model with new label (`btc_direction = close > open`) and drop `kalshi_spread_normalized`. See Future Roadmap.

---

**As of 2026-05-30 session 20: Train/holdout split + regime-aware calibrator. Calibrator passthrough but ready to deploy with regime as third feature. 395 tests pass.**

**Session 20: train/holdout split + regime-aware calibrator**

**Changes — session 20:**

| File | Change |
|------|--------|
| `btc_kalshi_system/models/calibrator.py` | **Train/holdout split**: `fit()` now holds out newest 20% (min 20 rows) as unseen eval set; only deploys if holdout Brier < passthrough Brier AND < `_prev_brier`; when reverting from passthrough, updates `_prev_brier` to passthrough holdout Brier. **Regime-aware**: `regimes=None` param added to `fit()`; when provided, builds 3-feature matrix `[raw, raw², regime_score]`; `_regime_aware` flag set on deploy; `transform(raw_prob, regime=None)` uses 3-feature vector when `_regime_aware`, 2-feature otherwise (backward compat). `_REGIME_ENCODING` dict: `trending_up=1.0`, `trending_down=-1.0`, `ranging=high_uncertainty=0.0`. `save()`/`load()` persist `regime_aware`. |
| `btc_kalshi_system/signal/fusion.py` | `calibrator.transform()` call now passes `regime=deepseek_regime` |
| `main.py` | All 3 `transform()` calls pass `regime=signal.deepseek_regime`. `_refit_calibrator()`: queries `deepseek_regime` column from both tables; passes `regimes=` array to `fit()` (None for k5 fallback path). |
| `scripts/train_calibrator.py` | `_UNION_QUERY` now selects `deepseek_regime`; passes `regimes=` to `cal.fit()` |
| `tests/models/test_calibrator.py` | 5 new regime-aware tests: `_regime_aware` flag set, false without regimes, transform differs by regime, unknown regime uses 0.0, save/load preserves flag. 22 total tests. |
| `tests/signal/test_fusion_kronos_raw.py` | Updated mock assertion to include `regime="trending_up"` |

**Effective minimum rows (after holdout):**
- `n_holdout = max(20, n // 5)` — at 124 total rows: n_holdout=24, n_train=100 = `_MIN_SAMPLES` → fits
- At 123 rows: n_train=99 → passthrough
- The 883-row guard in auto_retrain is unaffected (it gates total rows, not n_train)

**Why regime encoding is `±1`/`0`:**
- `trending_up=+1.0`, `trending_down=-1.0`: logistic can learn that same k15_raw has opposite implications in these regimes
- `ranging=high_uncertainty=0.0`: neutral; calibrator treats them identically unless data says otherwise
- Unknown/None regimes also map to 0.0 via safe `dict.get()` lookup

**Commit:** `12e3a24`

---

**As of 2026-05-30 session 19: Calibrator system overhaul + training pipeline expansion + Gate 9 removal. Calibrator currently passthrough (identity). _MIN_ROWS=883 in auto_retrain. 390 tests pass.**

**Session 19: calibrator improvements + training expansion + Gate 9 removal**

**Key findings driving changes:**
- k5 0.80+ bucket: avg calibrated prob 92%, actual win rate 52.4% — severe overconfidence (historical, from isotonic/passthrough era)
- k15 0.80+ bucket: avg k15 93%, actual win rate 39.4% — inverted signal at high confidence
- trending_down regime: k15 averaging 76.9% confidence, 28.6% win rate — k15 is an inverted signal in this regime
- k5 and k15 never disagree in direction (k15 only evaluated on trades that already passed k5 gates — structural, not informational)
- Calibrator trained on trades-only had selection bias: never saw low-confidence/rejected signals

**Changes — session 19:**

| File | Change |
|------|--------|
| `btc_kalshi_system/models/calibrator.py` | `_MIN_SAMPLES` lowered 300→100 — allows calibrator to fit on 129 available k15 rows (was blocking activation) |
| `btc_kalshi_system/signal/fusion.py` | No change — `trending_down` shrink was added then reverted; only `high_uncertainty` and `ranging` have shrink cases. |
| `tests/models/test_calibrator.py` | Updated 3 tests that hardcoded 300-sample threshold to use 100 (direct consequence of _MIN_SAMPLES change) |
| `scripts/auto_retrain_calibrator.py` | **New** — cron-driven calibrator auto-retrain. Three triggers: emergency (Brier > 0.25 on last 50 k15 rows), row-based (+50 combined rows), time-based (7 days). `_MIN_ROWS=883`. Marker at `models/calibrator_last_trained.json`. |
| `scripts/train_calibrator.py` | Training query replaced with UNION of trades + gate_rejections (eliminates selection bias). `min_rows` guard uses `total_available` count not windowed count. `AND failed_gate != 9` removed (redundant with `shadow=0`). |
| `main.py` | Gate 9 (edge-flip shadow, ~50 lines) removed from `_process_market`. `flip_price_cents` column retained for schema backward compatibility with existing rows. |
| `models/calibrator.pkl` | Forced to passthrough (identity). Current state: `passthrough=True`, `n_samples=300` (stale). |
| `models/calibrator_last_trained.json` | Written manually: `trained_at_rows=130`, timestamp `2026-05-30T12:45:22Z`. |
| crontab | Added `0 */2 * * *` entry for `auto_retrain_calibrator.py` |

**Calibrator history this session (in order):**
1. **Retrained on 129 k15 rows (trades only):** Brier 0.2708→0.2463. Inversion crossover at raw=0.57 — raw=0.90 mapped to cal=0.43. System was fading essentially all bullish signals above 57% conviction (the crossover stripping YES bets).
2. **Training expanded to UNION trades + gate_rejections (680 combined rows):** Curve collapsed to constant ~0.57–0.58 for all inputs — all directional gradient removed. Gate 5 rejected most trades due to Kelly=0 (combined ≈ 0.54–0.55).
3. **Forced to passthrough:** `state["passthrough"] = True` written directly to `.pkl`. Identity transform restores full directional signal. This was the correct fix — not enough regime diversity in 680 rows to train a stable logistic curve.
4. **`_MIN_ROWS` raised to 883:** 683 combined rows as of 2026-05-30 + 200 genuinely new observations required before next retrain. At ~50 k15-ready rows/day, ETA ~4 days.

**Current calibrator state:**
- `passthrough=True` → `calibrator.transform(raw) = raw` (identity, no transformation)
- Emergency trigger dormant while passthrough is active (no Brier baseline to compare)
- Marker file: `trained_at_rows=130` reflects the last non-passthrough k15-only fit
- `_prev_brier` staleness gap: after reverting to passthrough, `_prev_brier` inside the calibrator reflects the old 129-row k15 fit (Brier≈0.24), NOT the passthrough baseline. When 883-row retrain fires, run `--dry-run` first and manually compare fitted Brier vs passthrough Brier before allowing save. See memory: `project-calibrator-883-retrain`.

**Gate 9 removal:**
Gate 9 (edge-flip shadow) logged potential NO trades when Gate 2 "Kelly rounds to 0" fired on a YES signal. Used `shadow=2`, `failed_gate=9` in `gate_rejections`. Removed because:
- `shadow=2` rows were already excluded from training via `WHERE shadow=0` — no training impact
- No live trade impact (shadow-only)
- Removes ~50 lines of complexity from `_process_market`
- `flip_price_cents` column retained in DB schema for backward compatibility with existing rows
- `AND failed_gate != 9` clause was redundant with `shadow=0` filter and removed from all queries

**Commits:** `bc583b5`, `97fa222`, `b05896a`, `a6683e2`, `57dc3f8`, `bc482df`, `81112aa`, `ca9f363`, `f220981`

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

**26-feature `_FEATURE_ORDER`** (identical in `regime_model.py`, `train_regime.py`, and `fusion._regime_features()` dict keys — mismatch silently corrupts training). `kalshi_implied_prob` and `kalshi_spread_normalized` removed in session 25 to break Kalshi circularity.
```
funding_rate, funding_rate_trend, oi_delta_pct, cvd_normalized, basis_spread_pct,
brti_volatility_1h, cvd_velocity, cvd_acceleration, brti_momentum_5min,
brti_momentum_15min, candle_progress, hour_sin, hour_cos,
funding_window_proximity, trend_slope_1h, trend_r2_1h, hourly_sr_proximity,
range_breakout_flag, tape_speed_tpm, large_print_direction,
atm_iv, iv_rv_spread, pcr_oi, term_structure_slope, skew_25d,
btc_24h_return
```

**Feature sources:**
| Feature | Source |
|---------|--------|
| Features 1–6 | `derivatives_feed.py` → Redis `regime:features` |
| `cvd_velocity`, `cvd_acceleration` | Redis sorted set `regime:cvd_history` |
| `brti_momentum_*`, `candle_progress`, `hour_*`, `trend_*`, `hourly_sr_proximity`, `range_breakout_flag` | `fusion._regime_features()` from OHLCV |
| `tape_speed_tpm` | `store.get_raw_ticks(60)` |
| `large_print_direction` | `derivatives_feed.py` fetch_trades (net dir from prints > 2× avg size) |
| `funding_window_proximity` | UTC time proximity to 00/08/16h funding |
| `atm_iv`, `pcr_oi`, `term_structure_slope`, `skew_25d` | `deribit_options_feed.py` → Redis `options:features` |
| `iv_rv_spread` | Derived in `_get_market_context()`: `atm_iv − brti_volatility_1h` |
| `btc_24h_return` | `fusion._regime_features()` from 1h OHLCV (26 candles); defaults 0.0 when < 25 candles |
| `kalshi_implied_prob`, `kalshi_spread_normalized` | **Excluded from model** (session 25) — computed but not in feature dict. Removes Kalshi circularity. |

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

**Immediate (next session):**

0. **Build 15-min candle feature logger + confidence tracker.** See `docs/session-22-prompt.md` for the full implementation spec. Two deliverables: `candle_features` table in `trades.db` (background coroutine logging regime features at every 15-min close) + `scripts/regime_confidence_tracker.py` (confidence-stratified Gate 2 shadow analysis). No retrain — infrastructure only.

1. **Watch for 883 combined k15-ready rows — manual retrain check required (currently 826, ~1 day away).** When `python3 scripts/auto_retrain_calibrator.py --dry-run` shows row trigger firing AND min-rows guard passes, do NOT let it run automatically. Run `python3 scripts/train_calibrator.py --dry-run` first and manually compare fitted Brier vs passthrough Brier on the same 300-row window. Only allow save if fitted Brier < passthrough Brier by ≥0.005. See memory `project-calibrator-883-retrain` for exact SQL. **Note:** train/holdout split is now live — the auto-revert guard will correctly catch degraded fits. `_prev_brier` is still potentially stale (reflects 129-row k15 fit baseline, not passthrough) — manual verification is still required once. First deployment will use regime as third feature (`_regime_aware=True`).
   - The in-memory 25-resolution refit is also running continuously. Last attempt (07:10 2026-05-30) missed passthrough by only 0.0006 (Brier 0.2433 vs baseline 0.2427). Watch `logs/launchd_stderr.log` for `Calibrator refit: passthrough=False` — that means it auto-deployed a non-passthrough model.

**This week:**

2. **Monitor Gate 5 ranging filter impact.** Track: `SELECT deepseek_regime, failed_gate, COUNT(*) FROM gate_rejections WHERE failed_gate=5 AND DATE(timestamp) >= date('now','-7 days') GROUP BY deepseek_regime`. Win rate on ranging trades passing Gate 5 should move from 47% toward 60%.

3. **Investigate high_uncertainty oversizing.** −$0.84/trade at 51% win rate means Kelly is ignoring something. Check whether high_uncertainty fills are systematically at unfavorable prices (spread eating edge). Query: `SELECT AVG(fill_price_cents), AVG(pnl_dollars) FROM trades WHERE deepseek_regime='high_uncertainty' AND outcome IS NOT NULL`.

4. **After ~50 Gate 2 shadow trades: run `python3 scripts/regime_confidence_tracker.py`.** Check whether high-confidence regime calls (ABS(regime_prob - 0.5) > 0.2) have meaningfully higher accuracy. If not, dynamic weighting is premature — wait for more data. If yes, see Future Roadmap for dynamic weighting plan.

**Medium-term (~1–2 weeks after candle_features has 7+ days of data):**

5. **Retrain regime model with new label + drop `kalshi_spread_normalized`.** Prerequisites: `SELECT COUNT(*) FROM candle_features` ≥ 672 (7 days × 96/day). See Future Roadmap — Regime Model v2 section for full spec. Do both changes in one retrain.

6. **Segmented calibrators after regime model v2 is deployed.** Full separate calibrators per regime (Phase 3b). Higher accuracy ceiling than the current regime-as-feature model, but needs ~500 rows per regime. Ranging will hit that first (~August). trending_up/down may never have enough rows — those stay on global.

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

- **`_RANGING_SHRINK=0.7`, `_BOOTSTRAP_SHRINK=0.8`, `_UNCERTAINTY_SHRINK=0.5`** — do not equate. Only `high_uncertainty` and `ranging` have regime-specific shrinks; `trending_down` and `trending_up` fall through with no adjustment.

- **Calibrator is currently passthrough (identity).** `calibrator.transform(raw) = raw`. Emergency trigger in `auto_retrain_calibrator.py` is dormant while passthrough (no Brier baseline). `n_samples=300` in the `.pkl` is stale from the last non-passthrough fit — ignore it. Next retrain at 883 combined rows requires manual Brier comparison first. When it deploys, `_regime_aware=True` and `transform()` will use 3-feature vector.

- **`_prev_brier` staleness gap at next retrain.** After the passthrough revert, `Calibrator._prev_brier` reflects the old 129-row k15 Brier (~0.24), NOT the passthrough baseline. The train/holdout split (session 20) fixes this going forward: when `fit()` reverts to passthrough, it now updates `_prev_brier = passthrough_holdout_brier`. But the currently serialized `.pkl` still has the stale 0.24 value. Must manually compare fitted holdout Brier vs passthrough holdout Brier at 883-row retrain. See memory `project-calibrator-883-retrain`.

- **Regime-aware calibrator: `_regime_aware` flag gates the feature at `transform()`.** Old `.pkl` files load with `_regime_aware=False` (backward compat default in `load()`). When it first deploys with `regimes=`, `_regime_aware=True` is saved. All 3 transform() call sites in `main.py`, fusion.py, and train_calibrator.py already pass `regime=`. Unknown/None regimes encode to 0.0 — same as ranging.

- **Gate 9 (edge-flip shadow) is removed.** `failed_gate=9` and `shadow=2` rows still exist in the DB from the shadow period — do not mistake them for active data. `flip_price_cents` column retained in `gate_rejections` and `trades` schema for backward compatibility; it is no longer populated by any live code.

- **`gate_rejections` training filter: `WHERE shadow=0` already excludes all Gate 9 rows (which used `shadow=2`).** The old `AND failed_gate != 9` clause was redundant and has been removed from all queries in `train_calibrator.py` and `auto_retrain_calibrator.py`.

- **DeepSeek returns `NEUTRAL_DEFAULT` on 402, not `SAFE_DEFAULT`.**

- **DerivativesFeed re-resolves on ANY exception** (commit `229b88b`). Prior to this fix, only 403/Forbidden triggered failover; timeouts/resets silently kept a dead session alive. If the feed goes quiet again, check `redis-cli ttl regime:features` — TTL of -2 means it expired and feed is down. Restart main.py to recover.

- **Multi-source derivatives feed (session 7):** `_fetch_funding_and_oi()` now queries OKX, Hyperliquid, and Kraken Futures in parallel; averages whichever succeed. `okx_partial=True` only when ALL three fail. Hyperliquid reports 1h funding (multiply × 8 for 8h equiv). Kraken reports annualized funding (divide by 1095 for 8h equiv). `_prev_oi` is now a dict `{"okx": float, "hyperliquid": float, "kraken_futures": float}` — each source tracks its own prev OI for delta computation. `_coinglass_funding_and_oi` is preserved but no longer in the main fallback chain; it can be called directly for debugging.

- **Restart procedure (launchd):** `launchctl unload ~/Library/LaunchAgents/com.kronos.v2.plist && launchctl load ~/Library/LaunchAgents/com.kronos.v2.plist` — launchd will restart automatically on crash. Logs in `logs/launchd_stdout.log` and `logs/launchd_stderr.log`. Old manual procedure (direct python3) is superseded.

- **Feed health check:** `redis-cli ttl regime:features` should return 400–600. If -2, feed is down. `redis-cli get regime:features:lkg` shows LKG age via `_lkg_written_at` field.

- **Disagreement neutralization is a temporary patch for regime v1's circular label.** When `kronos_cal` and `regime_prob` are on opposite sides of 0.5, `_regime_in_fusion = 0.5` (neutral) is used instead of `regime_prob` in the fusion formula. This prevents bearish regime noise from dragging combined below Gate 5 threshold on bullish days. The raw `regime_prob` is still logged to `TradingSignal.regime_prob` and in the Gate 2 disagreement warning — only the fusion computation uses the neutralized value. **Remove this block after regime v2 deploys** (new label makes disagreements genuinely informative). The exact code to remove is the `_kronos_bullish != _regime_bullish` block in `fusion.py`'s `try:` branch.

- **`candle_features` table is data-collection only.** It does NOT feed into any existing training pipeline. It logs regime features + `btc_direction` at every 15-min candle close to enable the future regime label change. The `btc_direction` column there is `1 if close > open` on the BRTI 15-min candle — independent of any Kronos signal. Do not use it for calibrator training (calibrator uses `kronos_raw_15min` + outcome, not `btc_direction`).

- **`candle_features.candle_ts` is the open timestamp of the just-closed candle** (second-to-last row in 15-min OHLCV at log time), not the current time. The `UNIQUE` constraint on `candle_ts` makes writes idempotent — `INSERT OR IGNORE` is required.

- **Regime model v1 label is circular.** `direction == outcome` = "did Kronos get it right?" Training features include `kalshi_implied_prob` at 19% importance, meaning the model largely learned "when does Kalshi agree with Kronos?" This is partially circular — Gate 8 already captures that. Do NOT treat the v1 regime model as a fully independent signal. The v2 retrain (new label: `btc_direction = close > open`) fixes this by making the model a BTC direction predictor, not a Kronos-success predictor.

- **`kalshi_spread_normalized` is a dead feature in training.** It has near-zero variance (range 0.00–0.02¢) in the training dataset, giving it 0.000 importance in the v1 model. It IS populated in live inference (0.01–0.02), but this has no effect since the model ignores it. Drop it from `_FEATURE_ORDER` in the v2 retrain (3-file contract: `regime_model.py`, `train_regime.py`, `fusion._regime_features()`). The feature column stays in the DB schemas for backward compatibility.

- **Dynamic weighting and hybrid Kelly are gated on real disagreement data.** Do NOT implement before `python3 scripts/regime_confidence_tracker.py` shows 50+ Gate 2 shadow trades AND high-confidence regime calls have meaningfully higher accuracy than low-confidence ones. Implementing before that has no empirical basis.

---

## Future Roadmap

Ordered by data requirements. Each item has a clear trigger condition before implementation begins.

### Phase 1 — Calibrator infrastructure COMPLETE; waiting for data (2026-05-30)

**What's been fixed (all complete):**
- ✅ Label bug fixed: `y_up = int(direction == outcome)` instead of raw `outcome` — NO trades were training backwards before
- ✅ Training signal switched from k5 → k15: calibrator now maps `kronos_raw_15min` to calibrated prob (15-min horizon matches 15-min markets)
- ✅ Switched from isotonic → quadratic logistic [raw, raw²]: can learn inverted-U shapes; isotonic was forced monotone-increasing
- ✅ Calibrator persistence added: survived restarts (was resetting to passthrough on every launchd restart before)
- ✅ Rolling 300-row window: no longer blending all historical regimes
- ✅ Training expanded to UNION trades + gate_rejections: selection bias eliminated (was only seeing high-confidence placed trades)
- ✅ `_MIN_SAMPLES` lowered 300→100; `auto_retrain_calibrator.py` added (cron `0 */2 * * *`; row/time/emergency triggers)
- ✅ Gate 9 (edge-flip shadow) removed
- ✅ Train/holdout split: `fit()` holds out newest 20% (min 20); deploys only if holdout Brier < passthrough AND < `_prev_brier`
- ✅ Regime-aware calibrator: `[raw, raw², regime_score]` feature vector; `trending_up=1.0`, `trending_down=-1.0`, `ranging=0.0`; first deployment activates at 883 rows

**Current state:**
- **Calibrator currently passthrough (identity)** — 826 combined rows as of 2026-05-30, need 883 before external retrain (~1 day). At ~50 k15-ready rows/day, ETA ~1 day.
- In-memory 25-resolution refit running continuously; last attempt (07:10 2026-05-30) missed passthrough threshold by 0.0006 (Brier 0.2433 vs baseline 0.2427)
- Emergency trigger dormant (no Brier baseline while passthrough)

**What's still missing:**
1. **More data** — regime diversity is the #1 blocker. Dataset is ~80% ranging/high_uncertainty. Trending_up sessions like 2026-05-30 are filling the gap. First deployment at 883 rows.

**Before next retrain:** manually compare fitted Brier vs passthrough Brier. See memory `project-calibrator-883-retrain`. Do NOT rely on `_prev_brier` auto-revert guard (stale from 129-row fit).

### Phase 1b — Regime model v2 with clean label (~1–2 weeks)

**Timeline:**
- **ETA June 6–7:** `candle_features` hits 672 rows (7 days × 96/day). Logger started 2026-05-30.
- **Day of training:** run updated `train_regime.py`, evaluate holdout (target Brier < 0.25, accuracy > 60%).
- **Days 1–4 shadow:** keep Gate 2 shadow mode, watch whether new model's disagreements actually predict BTC direction.
- **Day 5+ if clean:** restore `_REGIME_WEIGHT = 0.4`, remove disagreement neutralization from `fusion.py`, consider Gate 2 enforcement.

**Why `candle_features` first:** Without the session 22 logger, retraining uses only ~1099 trade/rejection rows — same data, different label, marginal improvement. With 7 days of `candle_features`, you get ~672 independent rows (one per 15-min candle close), regardless of trade frequency. That's the 10× multiplier.

**Context:** Regime model v1 was trained with `direction == outcome` — "did Kronos get it right?" Two problems:
1. **Circular:** `kalshi_implied_prob` at 19% importance → model learned "when does Kalshi agree with Kronos?" Gate 8 already captures this. The regime model's disagreements are largely noise — confirmed by Gate 2 shadow data (39 disagreements at 76% win rate on 2026-05-30).
2. **Noisy label:** 50/50 at dataset level. The model can't learn much from a coin-flip label on 1099 rows.

**New label:** `btc_direction = 1 if 15-min candle close > open else 0`. Clean ground truth from `candle_features.btc_direction`. The regime model becomes a genuine BTC direction predictor.

**Retrain spec (both changes in one retrain):**
1. **New training source:** `SELECT * FROM candle_features WHERE btc_direction IS NOT NULL`. Trigger: `SELECT COUNT(*) FROM candle_features` ≥ 672.
2. **Drop `kalshi_spread_normalized`:** Zero importance in v1 (near-zero variance, 0.00–0.02¢). Remove from `_FEATURE_ORDER` in all 3 files (`regime_model.py`, `train_regime.py`, `fusion._regime_features()`). Model shrinks 27→26 features. `test_feature_order` will catch any missed file. Column stays in DB schemas.
3. **Run `python3 scripts/train_regime.py --dry-run` first.** Expect: `kalshi_implied_prob` feature importance drops significantly (no longer circular).
4. Deploy shadow mode. Gate 2 disagreement rate may increase — the model is now genuinely independent of Kronos. That's expected and fine.

**After regime v2 deploys and validates:**
- Remove disagreement neutralization from `fusion.py` (the `_kronos_bullish != _regime_bullish` block)
- Set `_KRONOS_WEIGHT = 0.2`, `_REGIME_WEIGHT = 0.8` (regime primary; see session 27 decision)
- Update test `test_gate2_shadow_mode_does_not_block` back to `expected = 0.6 * 0.70 + 0.4 * 0.30`
- Consider Gate 2 enforcement after 50+ shadow validation trades

**What NOT to do yet:**
- Do NOT add `kronos_raw_15min`/`kronos_raw` as regime features. Not enough k15 rows per regime.
- Do NOT implement dynamic weighting or hybrid Kelly until confidence-stratified accuracy data exists (~50+ Gate 2 shadow trades with regime v2).

### Phase 2 — Dynamic weighting + hybrid Kelly (after 50+ Gate 2 shadow trades)

**Trigger:** `python3 scripts/regime_confidence_tracker.py` shows high-confidence regime calls (ABS(regime_prob - 0.5) > 0.2) have meaningfully higher accuracy than low-confidence calls.

**Dynamic weighting** (replaces flat 60/40 fusion split):
- Scale regime weight by `ABS(regime_prob - 0.5)`. At conf=0.9 → go 50/50; at conf=0.1 → go 75/25 (regime nearly ignored).
- Formula: `regime_weight = 0.4 * min(ABS(regime_prob - 0.5) / 0.3, 1.0)` (scales linearly from 0 at no-confidence to 0.4 at full confidence). Adjust constants after seeing real confidence distribution.
- Change is in `fusion.py` signal computation only. Single-line change, but requires validation.
- Do NOT implement until confidence-stratified accuracy data confirms the premise.

**Hybrid Kelly on disagreements** (reduces downside on uncertain disagreements):
- When Gate 2 is shadow mode and `regime_prob` disagrees with Kronos direction AND `ABS(regime_prob - 0.5) > 0.2` (high confidence): cut Kelly to 50%.
- Implementation: add `regime_confidence = ABS(signal.regime_prob - 0.5)` check in `pretrade_checklist.py`; if disagreement AND confidence > 0.2, multiply `kelly_dollars *= 0.5` before the sizing chain.
- This is the Gate 10 planning item already noted in the handoff. Don't implement before Gate 2 has 50+ shadow trades.
- **Note:** Gate 2 enforcing (`REGIME_GATE2_ENFORCING=True`) is a hard block. Hybrid Kelly is a softer version — block only at high confidence, reduce at medium, pass at low. Use hybrid Kelly first; flip to enforcing only if hybrid Kelly data shows high-confidence disagreements are strong negative-EV.

### Phase 3 — Regime model trains (~early July, after Phase 1b retrain)
**Trigger:** `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL AND features_stale=0 AND deribit_stale=0 AND atm_iv IS NOT NULL` ≥ 500.

- This is the *deribit-clean* retrain — Phase 1b uses the new label but may still include rows with `deribit_stale=1`. This phase uses the full 27-feature (or 26 after dropping `kalshi_spread_normalized`) clean dataset.
- Run `python3 scripts/train_regime.py --dry-run`. Check Brier < 0.25 and feature importances make sense (funding_rate, cvd_normalized, brti_momentum should rank high; `kalshi_implied_prob` should be lower than 19% with the new clean label).
- Deploy in shadow mode (REGIME_GATE2_ENFORCING=false). Observe ~50 trades. Gate 2 disagreement rate should be < 35% — if higher, the model is noisy and needs more data before enforcing.
- Once enforcing, the 0.4 regime weight in fusion.py becomes real. Expect calibrated_prob distribution to widen slightly.
- Add `kronos_raw_15min` and `kronos_raw` as features 29–30 in the NEXT retrain after this one (not this one — too soon, not enough k15 rows per regime to be meaningful).

### Phase 3 — Regime-aware calibration (~June–August)

**Phase 3a — Regime as input feature** ✅ COMPLETE (session 20, 2026-05-30)
- Feature vector extended from `[raw, raw²]` to `[raw, raw², regime_score]`
- Encoding: `trending_up=1.0`, `trending_down=-1.0`, `ranging=high_uncertainty=0.0`
- `_regime_aware=True` flag set on deploy; `transform(p, regime=X)` dispatches correctly
- All callers updated: `fusion.py`, `main.py` (3 sites), `_refit_calibrator()`, `train_calibrator.py`
- Will activate on first non-passthrough deployment (883+ rows)

**Phase 3b — Fully segmented calibrators** (~August, after Phase 3a deployed and evaluated)
**Trigger:** ≥500 k15_raw rows per regime for ranging.
- Train a separate logistic [raw, raw²] calibrator on ranging-only rows
- In `fusion.py`, route `calibrator.transform()` through a `SegmentedCalibrator` wrapper that checks `deepseek_regime` and dispatches to the right model, falling back to global for regimes without enough data
- Expect the ranging calibrator to learn a more aggressive compression (closer to 0.5) than the global one
- high_uncertainty: segmented calibrator only if ≥400 k15 rows accumulate. Otherwise keep on global
- trending_up/down: probably never enough rows — keep on global or Phase 3a regime-feature model

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
| Candle features logger health | `python3 scripts/regime_confidence_tracker.py` | Check rows logged last 24h ≈ 96; check `btc_direction` split is not wildly skewed |
| Regime confidence stratification | `python3 scripts/regime_confidence_tracker.py` | After 50+ trades with regime_prob: high-confidence calls (>0.70) should have higher accuracy than low-confidence (<0.60) — if not, dynamic weighting is premature |
