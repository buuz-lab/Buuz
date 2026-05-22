# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi. The system trades **KXBTC15M** (`KXBTC15M-*`) 15-minute up/down markets. It forecasts direction using the Kronos time-series model + XGBoost regime classifier + DeepSeek LLM gate, sizes positions with fractional Kelly, and runs 6 pre-trade gates. The immediate goal is to accumulate 500+ resolved paper trades so the calibrator and edge tracker cross their thresholds, then flip to live trading.

---

## Current Progress

**As of this session: `trades.db` shows 202 total / 201 resolved / 1 open. 125W / 76L (62%). Net P&L: +$280.92.**

- `PAPER_TRADING=true` in `.env`
- System is healthy, running as PID 19229 on `logs/kronos_2026-05-21_20-05-07_232084.log`
- All gate blockers from prior sessions are fixed and confirmed in code
- **New fix this session**: `_RANGING_SHRINK = 0.7` added to `fusion.py` — when DeepSeek reports `ranging` regime, the combined signal is compressed 30% toward 0.5 (reducing kelly size). Applied in both the trained-regime path and the bootstrap `NotTrainedError` path.
- **RegimeModel is untrained** — always raises `NotTrainedError`. System runs in bootstrap mode (Kronos-only, 0.8× shrink). See Gotchas for details.
- Test suite: **197 pass, 0 fail**
- Candle count: 402 completed 5-min candles loaded from Redis on last startup — well above the 400 target

**Bootstrap counters needed before going live:**
- `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL` → need ≥ 500 resolved (currently 201)
- `edge_tracker.current_edge() > 0` over last 50 trades

---

## What Worked

- **Kronos preload in `KronosV2.__init__()` before asyncio**: The only reliable fix for the Apple Silicon segfault. Never load it inside `asyncio.to_thread()` or any async context on Apple Silicon.
- **`map_location="cpu"` in both `from_pretrained()` calls** and **`torch.set_num_threads(1)` BEFORE `from_pretrained()`**: Triple fix for Apple Silicon / PyTorch Accelerate segfaults.
- **`asyncio.to_thread(self._run_cycle)`**: Keeps 5-min blocking CPU cycle off the event loop so WebSocket feeds aren't starved.
- **RSA-PSS signing** (not PKCS1v15) for Kalshi auth; sign path-only, strip query string at `?`.
- **DeepSeek 402 → `NEUTRAL_DEFAULT`**: `suppress=False`, `regime=ranging`. Using `high_uncertainty` as the 402 fallback was silently shrinking all signals and killing Gate 5.
- **Direction-aware pricing**: Gate 5 and Kelly use `win_prob = 1 - calibrated_prob` and `trade_price = 100 - bid_cents` for "no" trades.
- **`_BOOTSTRAP_SHRINK = 0.8`**: The `NotTrainedError` path in `fusion.py` uses 0.8 instead of 0.5. An untrained regime ≠ high market uncertainty.
- **`_RANGING_SHRINK = 0.7` (this session)**: When DeepSeek signals `ranging`, compress the combined signal 30% toward 0.5. Applied in both the trained and bootstrap paths. Motivation: in ranging markets, momentum signals flip frequently causing consecutive losses at full size.
- **DeepSeek suppress_trading prompt**: Explicit rules — `suppress=true` ONLY for extraordinary events (FOMC in <30 min, exchange hacks, flash crashes), NOT for calm/ranging/low-volatility markets.
- **Decimal strike parsing**: `_extract_strike` uses `try: float(part[1:])` instead of `.isdigit()` so KXBTCD decimal strikes parse correctly.
- **24h position age-out**: `_check_resolutions` calls `monitor.remove_position()` for any open position older than 24h.
- **Gate 6 skip for KXBTC15M**: Gate 6 is skipped entirely when `signal.timeframe == "15min"`.
- **15-min BRTI reference price**: `_get_15min_reference_price()` walks the 15-min OHLCV to the last completed candle. Do not revert to `composite_price` for 15-min markets.
- **Dedup market list by ticker**: `_get_active_markets` deduplicates on ticker so the same market isn't processed twice per cycle.

---

## What Failed (avoid repeating)

- **Running Kronos inside `asyncio.to_thread()` before preloading**: Segfaults on Apple Silicon.
- **`torch.set_num_threads(1)` after `from_pretrained()`**: Race happens during load, not inference. Must come first.
- **`SAFE_DEFAULT` (high_uncertainty) as DeepSeek fallback for billing errors**: Shrinks all signals 50%, killing Gate 5.
- **`.isdigit()` for float strike parsing**: Returns False for decimal strings.
- **`_UNCERTAINTY_SHRINK = 0.5` in the `NotTrainedError` path**: Too aggressive — signals never cleared Gate 5 during bootstrap.
- **Gate 5 using `calibrated_prob - ask_price` for "no" trades**: "no" trades could never pass.
- **Old Kalshi `orderbook` format**: Kalshi now returns `orderbook_fp`. Old parser returned (0, 0, 0).
- **Gate 6 blocking all KXBTC15M markets**: Distance was always 0 → every 15-min market rejected.
- **Using 5-min close as threshold for KXBTC15M**: Introduces directional error mid-window. Fixed: `_get_15min_reference_price()`.
- **KXBTCD in market loop**: P ≈ 0, Kelly = 0, Gate 2 always rejects. Removed from `_get_active_markets()`.
- **Bybit for derivatives**: HTTP 403 CloudFront geo-block for US. OKX is the primary fallback.
- **PKCS1v15 signing**: Kalshi requires RSA-PSS.
- **Ranging regime with no signal shrink (this session)**: Full-size kelly in ranging markets caused a -$76 drawdown from ATH ($357 → $280) over ~30 trades as momentum signals flipped repeatedly. Fixed with `_RANGING_SHRINK = 0.7`.

---

## Files Touched / Created This Session

| File | Change |
|------|--------|
| `btc_kalshi_system/signal/fusion.py` | Added `_RANGING_SHRINK = 0.7`; apply it in both the trained-regime path and the bootstrap `NotTrainedError` path when `deepseek_regime == "ranging"` |
| `handoff.md` | This file |

---

## Next Steps

1. **Train the RegimeModel** — 201 resolved trades is enough to start. The model (`btc_kalshi_system/models/regime_model.py`) is XGBoost binary classification: features are `funding_rate`, `funding_rate_trend`, `oi_delta_pct`, `cvd_normalized`, `basis_spread_pct`, `brti_volatility_1h`; label is `outcome` (1=WIN/UP, 0=LOSS/DOWN). **Problem**: these features are not persisted in `trades.db` — only `kronos_raw` and `outcome` are stored. Need to either (a) add feature columns to the trades table and backfill from Redis, or (b) design a separate training pipeline. Once trained, call `regime_model.save(path)` and load it at startup in `main.py`. This unlocks Gate 2 (direction agreement check) and the full 60/40 Kronos+regime signal blend.

2. **Continue accumulating paper trades toward 500 resolved** — currently at 201. At ~15 min per trade cycle, expect to hit 500 in ~2-3 more days of continuous operation.

3. **Evaluate ranging shrink impact** — the `_RANGING_SHRINK = 0.7` was just deployed. Monitor whether kelly sizes decrease noticeably during the next ranging period and whether the win rate stabilizes. If 0.7 is still too permissive, consider tightening to 0.65.

4. **Fix Gate 4 for live trading** — Gate 4 (`edge_above_threshold`) is bypassed in paper mode. Before going live, verify the edge tracker is accumulating correctly and will actually gate live trades.

5. **Go live** — once resolved ≥ 500 AND `edge_tracker.current_edge() > 0` over last 50 trades: set `PAPER_TRADING=false` in `.env` and restart.

---

## Context / Gotchas

- **Kronos MUST be preloaded in `KronosV2.__init__()`** — never inside `asyncio.to_thread()` or any async context on Apple Silicon. Most important rule.
- **`torch 2.x` on Apple Silicon** — MPS causes segfaults. `map_location="cpu"` + `set_num_threads(1)` + preload-before-asyncio is the triple fix.
- **KXBTC15M threshold = last completed 15-min BRTI candle close** — do NOT use `composite_price` or 5-min close for 15-min markets.
- **Gate 6 is skipped for `timeframe == "15min"`** — do not re-add.
- **RegimeModel is permanently in `NotTrainedError` bootstrap mode** — it was always this way. The XGBoost model exists in code but has never been trained or loaded. `regime_prob` is `nan` in every DB row. Gate 2 (Kronos ↔ regime agreement) is always bypassed. The trained-regime code path in `fusion.py` is dead code until training is implemented.
- **Ranging shrink is now active in bootstrap path** — `_RANGING_SHRINK = 0.7` applies when `deepseek_regime == "ranging"` even while regime model is untrained. This is the current production behavior.
- **Direction-aware pricing is critical** — for "no" trades: `win_prob = 1 - calibrated_prob`, `fill_price_cents = 100 - best_bid_cents`. Do not revert.
- **Kalshi orderbook format is `orderbook_fp`** — `yes_dollars`/`no_dollars` are ascending `[price_str, qty_str]` pairs. Best bid = `list[-1][0]`.
- **DeepSeek suppress_trading is a hard gate** — if True, ALL markets in the cycle are blocked. Falls back to `NEUTRAL_DEFAULT` on 402 (not SAFE_DEFAULT). Check credits at platform.deepseek.com.
- **"No signal (gated out)" = Gate 1 or Gate 2 in `fusion.get_signal()`**. "checklist failed" = Gates 1–6 in `PreTradeChecklist.run()`.
- **`watch` not available on macOS** — use `while true; do <command>; sleep 30; done`.
- **`dump.rdb`** holds Redis tick history — do not delete it.
- **Kalshi key**: `KALSHI_KEY_ID` in `.env`, private key at `./keys/kalshi_private.key`. Both gitignored.
- **Test suite**: `python3 -m pytest` from project root. Expected: **197 pass, 0 fail**.
- **RSA-PSS signing** (not PKCS1v15) for Kalshi; sign path-only; base URL `https://api.elections.kalshi.com`.
- **Hedged positions are normal** — Kronos sometimes enters both YES and NO on the same strike in consecutive cycles when the signal is near 0.5. This is not a bug; it's a reflection of low conviction in ranging conditions.
