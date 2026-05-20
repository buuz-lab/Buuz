# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi. The system trades **KXBTC 15-minute and hourly up/down markets** (`KXBTC15M-*`, `KXBTCD-*`). It forecasts direction using the Kronos time-series model + XGBoost regime classifier + DeepSeek LLM gate, sizes positions with fractional Kelly, and runs 6 pre-trade gates. The immediate goal is to accumulate 500+ resolved paper trades so the calibrator and edge tracker cross their thresholds, then flip to live trading.

---

## Current Progress

**All known gate blockers are fixed. trades.db was at 0 trades when the last session ended — the Gate 6 fix (skip for KXBTC15M) was the final blocker and was committed/pushed. After restarting, KXBTC15M paper trades should begin flowing.**

- `PAPER_TRADING=true` in `.env`
- Redis live with 7200+ ticks and 400+ completed 5-min candles
- Kronos model loaded cleanly to CPU before asyncio starts
- DeepSeek working: `suppress=False, regime=ranging`
- Gate 6 now skipped for `timeframe == "15min"` — this was the last blocker
- KXBTCD markets will always fail Kelly Gate 2 (correct behavior — wrong instrument for 5-min Kronos, see Gotchas)
- Orderbook parser updated for new Kalshi `orderbook_fp` format
- **Verify fills after restart**: `while true; do sqlite3 ~/Kronos\ V2/trades.db "SELECT COUNT(*) total, SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) resolved FROM trades;"; sleep 30; done`

**Bootstrap counters needed before going live:**
- Calibrator: `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL` → need ≥ 500
- Edge tracker: rolling window of last 50 trades with positive realized edge (`edge_tracker.current_edge() > 0`)

---

## What Worked

- **Kronos preload in `KronosV2.__init__()` before asyncio**: The only reliable fix for the Apple Silicon segfault. Never load it inside `asyncio.to_thread()` or any async context on Apple Silicon.
- **`map_location="cpu"` in both `from_pretrained()` calls** and **`torch.set_num_threads(1)` BEFORE `from_pretrained()`**: Triple fix for Apple Silicon / PyTorch Accelerate segfaults.
- **`asyncio.to_thread(self._run_cycle)`**: Keeps 5-min blocking CPU cycle off the event loop so WebSocket feeds aren't starved.
- **RSA-PSS signing** (not PKCS1v15) for Kalshi auth; sign path-only, strip query string at `?`.
- **DeepSeek 402 → `NEUTRAL_DEFAULT`**: `suppress=False`, `regime=ranging`. Using `high_uncertainty` as the 402 fallback was silently shrinking all signals and killing Gate 5.
- **Direction-aware pricing (this session)**: Gate 5 and Kelly now use `win_prob = 1 - calibrated_prob` and `trade_price = 100 - bid_cents` for "no" trades. Previously "no" trades always had negative signal_edge and could never pass Gate 5.
- **`_BOOTSTRAP_SHRINK = 0.8` (this session)**: The `NotTrainedError` path in `fusion.py` now uses 0.8 instead of 0.5 (`_UNCERTAINTY_SHRINK`). An untrained regime ≠ high market uncertainty; 0.5 was killing Gate 5 during bootstrap.
- **DeepSeek prompt key name fix (this session)**: `_build_prompt` was passing `funding_trend`/`oi_delta`/`basis_spread` but feature store writes `funding_rate_trend`/`oi_delta_pct`/`basis_spread_pct`. The mismatches caused DeepSeek to receive `n/a` for most fields and suppress all trading.
- **DeepSeek suppress_trading prompt clarification (this session)**: Added explicit rules — `suppress=true` ONLY for extraordinary events (FOMC in <30 min, exchange hacks, flash crashes), NOT for calm/ranging/low-volatility markets.
- **Decimal strike parsing fix (this session)**: `_extract_strike` used `.isdigit()` which returns False for `"73749.99"`. Replaced with `try: float(part[1:])` so KXBTCD decimal strikes parse correctly instead of falling back to composite_price.
- **24h position age-out (this session)**: `_check_resolutions` now calls `monitor.remove_position()` for any open position older than 24h. Outcome stays `NULL` in SQLite so it doesn't bias the calibrator.
- **Gate 6 skip for KXBTC15M (this session)**: For 15-min up/down markets, `_extract_strike` falls back to `composite_price` (there is no explicit strike field). Distance = 0 < $150, so Gate 6 was rejecting every single KXBTC15M market unconditionally. Fixed by skipping Gate 6 entirely when `signal.timeframe == "15min"`. This was the final blocker preventing any paper trades.
- **`orderbook_fp` parser rewrite (this session)**: Kalshi changed their orderbook API format. Old: `orderbook.yes` / `orderbook.no` — integer cents, descending. New: `orderbook_fp.yes_dollars` / `orderbook_fp.no_dollars` — dollar-string pairs `[price, qty]`, ascending. Old parser returned (0, 0, 0) for every market, causing Gate 2 to fail with "Insufficient depth." Fixed `_parse_orderbook` to detect `orderbook_fp` first:
  ```python
  book_fp = orderbook.get("orderbook_fp")
  if book_fp:
      yes_bids = book_fp.get("yes_dollars", [])   # ascending dollar strings
      no_bids = book_fp.get("no_dollars", [])
      best_yes_bid = float(yes_bids[-1][0]) if yes_bids else 0.0
      best_no_bid = float(no_bids[-1][0]) if no_bids else 0.0
      best_bid_cents = round(best_yes_bid * 100)
      best_ask_cents = round((1.0 - best_no_bid) * 100)
      available_contracts = int(float(no_bids[-1][1])) if no_bids else 0
  ```
- **KXBTCD instrument mismatch confirmed (this session)**: Kronos predicts 5-min BTC close. KXBTCD markets have strikes $79K–$83K+ while BTC was ~$76,700. `P(close > $79K in 5 min) ≈ 0.0` → Kelly = 0 → Gate 2 fails with "Kelly size rounds to 0 contracts." This is **correct behavior, not a bug**. KXBTCD is the wrong instrument for a 5-min forecaster. Focus exclusively on KXBTC15M markets.
- **CircuitBreaker `paper_trading` constructor injection**: `check()` was reading `config.PAPER_TRADING` at call time, so `_check_rolling_edge` and `_check_calibrator` were skipped whenever `.env` had `PAPER_TRADING=true`. Added `paper_trading: bool | None = None` to `__init__`; resolves once at construction (`self._paper_trading`). Tests pass `paper_trading=False` explicitly so live-mode checks always run regardless of env. `main.py` passes no argument and continues to read `config.PAPER_TRADING`.

---

## What Failed (avoid repeating)

- **Running Kronos inside `asyncio.to_thread()` before preloading**: Segfaults on Apple Silicon due to PyTorch Accelerate + macOS kqueue race. The `preload()` call in `__init__` is the fix.
- **`torch.set_num_threads(1)` after `from_pretrained()`**: Race happens during load, not inference. Must come first.
- **`device="cpu"` in `KronosPredictor` alone without `map_location="cpu"`**: MPS is chosen by `from_pretrained()` before `__init__` can call `.to("cpu")`, causing segfault.
- **`SAFE_DEFAULT` (high_uncertainty) as DeepSeek fallback for billing errors**: Shrinks all signals 50% toward 0.5, killing Gate 5. Use `NEUTRAL_DEFAULT`.
- **`.isdigit()` for float strike parsing**: Returns False for decimal strings like `"73749.99"`. Always use `try: float(part[1:])`.
- **Wrong key names in `_build_prompt`**: Feature store writes `funding_rate_trend`, `oi_delta_pct`, `basis_spread_pct` but the prompt was asking for `funding_trend`, `oi_delta`, `basis_spread`. DeepSeek saw `n/a` and suppressed trading.
- **Ambiguous DeepSeek suppress_trading prompt**: Without explicit guidance, DeepSeek interpreted "calm/ranging/low-volatility" as a reason to suppress. Must explicitly state that suppress is reserved for extraordinary events only.
- **`_UNCERTAINTY_SHRINK = 0.5` in the `NotTrainedError` path**: Too aggressive for bootstrap. Combined signals never cleared Gate 5. Use `_BOOTSTRAP_SHRINK = 0.8` instead.
- **Gate 5 using `calibrated_prob - ask_price` for "no" trades**: P(up) is <0.5 for down signals but ask is >0.5, so edge was always negative — "no" trades could never pass. Fixed to use `win_prob - market_price` where both are direction-adjusted.
- **Old Kalshi `orderbook` format assumed in `_parse_orderbook`**: Kalshi now returns `orderbook_fp` with dollar-string pairs in ascending order, not integer cents in descending order. The old parser returned (0, 0, 0) for every market, silently killing Gate 2.
- **Using 5-min close as threshold for KXBTC15M markets**: KXBTC15M resolves relative to the last completed 15-min BRTI candle close, not the live 5-min close. If BTC drifts $300 during the current 15-min window, using spot/5-min close makes Kronos compute the wrong P(direction). Fixed: `_get_15min_reference_price()` walks the 15-min OHLCV to the most recent completed candle. Do not revert this.
- **Gate 6 blocking all KXBTC15M markets**: `_extract_strike` has no explicit strike field for up/down markets and falls back to `composite_price`. Distance from strike to composite = 0, which is always < $150 → every single 15-min market rejected. Skipping Gate 6 for `timeframe == "15min"` is the fix.
- **Assuming KXBTCD markets are tradeable with Kronos**: Kronos predicts 5-min ahead. KXBTCD strikes are set relative to BTC's daily target — when BTC is at $76,700 and the nearest strike is $79,000, `P(close > $79K in 5 min) = 0` and Kelly = 0. These markets will never pass Gate 2 and should be ignored.
- **Bybit for derivatives**: HTTP 403 CloudFront geo-block for US users. OKX is the primary fallback.
- **PKCS1v15 signing**: Kalshi requires RSA-PSS. Auth fails with 401.
- **Missing `from loguru import logger` in fusion.py**: Adding logger calls without the import crashes the signal engine silently.

---

## Files Touched / Created This Session

| File | Change |
|------|--------|
| `btc_kalshi_system/execution/pretrade_checklist.py` | Gate 2 (Kelly) and Gate 5 now direction-aware: "no" trades use `win_prob = 1 - calibrated_prob` and `trade_price = 100 - bid_cents`; **Gate 6 now skipped for `timeframe == "15min"`** (final blocker fix) |
| `btc_kalshi_system/signal/fusion.py` | `_BOOTSTRAP_SHRINK = 0.8` for `NotTrainedError` path (was `_UNCERTAINTY_SHRINK = 0.5`); added `from loguru import logger`; Gate 1 now logs suppress reason + notes at WARNING |
| `btc_kalshi_system/models/deepseek_parser.py` | Fixed `_build_prompt` key names (`funding_rate_trend`, `oi_delta_pct`, `basis_spread_pct`); rewrote suppress_trading prompt rules to clarify suppress is for extraordinary events only |
| `main.py` | `fill_price_cents` is direction-aware (`best_ask_cents` for yes, `100 - best_bid_cents` for no); 24h age-out in `_check_resolutions` via `monitor.remove_position()`; fixed decimal strike parsing (`.isdigit()` → `try: float()`); **`_parse_orderbook` fully rewritten to handle new `orderbook_fp` format** |
| `btc_kalshi_system/portfolio/circuit_breaker.py` | Added `paper_trading: bool \| None = None` to `__init__`; resolved to `self._paper_trading` at construction; `check()` now uses `self._paper_trading` instead of `config.PAPER_TRADING` |
| `tests/portfolio/test_circuit_breaker.py` | Added `paper_trading: bool = False` to `make_breaker`; passed through to `CircuitBreaker`; fixes 2 pre-existing test failures |
| `handoff.md` | This file |

---

## Next Steps

1. **Restart the system and confirm KXBTC15M paper trades are flowing** — Gate 6 was the last blocker; it was pushed but a restart is required to pick it up. Run: `while true; do sqlite3 ~/Kronos\ V2/trades.db "SELECT COUNT(*) total, SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) resolved FROM trades;"; sleep 30; done`. Expect `total` to climb within 1–2 cycles (10–15 min). Resolved count lags by ~15 min (KXBTC15M market duration).

2. **Diagnose if no fills appear after 30 min** — run `grep "checklist failed\|No signal" ~/Kronos\ V2/logs/kronos_*.log | tail -20`. If all failures are Gate 2 Kelly on KXBTCD — that's expected (wrong instrument). If Gate 5 failures dominate, Kronos edge is too thin; check `grep "signal_edge" logs`. If Gate 1 suppress is firing, check DeepSeek key and credits at `platform.deepseek.com`.

3. **Ignore KXBTCD market failures** — KXBTCD will always fail Kelly Gate 2. `P(5-min close > $79K strike when BTC = $76.7K) = 0`. This is correct. The system should be configured to skip KXBTCD series entirely if possible, or accept that all KXBTCD attempts fail at Gate 2.

4. **Train the RegimeModel at ~100 resolved trades** — once `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL` ≥ 100, fit the XGBoost model using resolved trade features. The `RegimeModel.save(path)` call is the trigger. The system handles `NotTrainedError` gracefully so this is not blocking.

5. **Add DeepSeek credits if they run out** — key is in `.env` as `DEEPSEEK_API_KEY`. System reverts to `NEUTRAL_DEFAULT` on 402 (suppress=False, ranging). Check at platform.deepseek.com. Note: DeepSeek cache is 15 minutes, so a credit top-up takes up to 15 min to take effect without a restart.

6. **Go live** — once `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL` ≥ 500 AND `edge_tracker.current_edge() > 0` over last 50 trades: set `PAPER_TRADING=false` in `.env` and restart.

---

## Context / Gotchas

- **Kronos MUST be preloaded in `KronosV2.__init__()`** — never inside `asyncio.to_thread()` or any async context on Apple Silicon. This is the single most important rule.
- **`torch 2.x` on Apple Silicon** — MPS causes segfaults in the Kronos inference path. `map_location="cpu"` + `set_num_threads(1)` + preload-before-asyncio is the triple fix.
- **"No signal (gated out)" in logs = Gate 1 or Gate 2 in `fusion.get_signal()`** — NOT a checklist failure. If all markets show this, check DeepSeek suppress first: `grep "Gate 1\|suppress=" ~/Kronos\ V2/logs/kronos_*.log | tail -5`.
- **"checklist failed" in logs = Gates 1–6 in `PreTradeChecklist.run()`** — the trade reached the checklist but was rejected. Grep for `[gate N]` to see which gate.
- **DeepSeek suppress_trading is a hard gate** — if True, ALL markets in the cycle are blocked. Always log what DeepSeek returns; the log line in `fusion.get_signal()` now does this at DEBUG level.
- **Direction-aware pricing is critical** — for "no" trades: `win_prob = 1 - calibrated_prob`, `fill_price_cents = 100 - best_bid_cents`. If this reverts, "no" trades will always fail Gate 5 with negative edge.
- **KXBTCD decimal strikes** — tickers like `KXBTCD-26MAY1922-T73749.99`. The `T` prefix parser uses `try: float(part[1:])` now. If you see markets falling back to composite_price as strike, the ticker format probably changed.
- **KXBTCD markets always fail Gate 2** — this is expected and correct. Kronos forecasts 5-min BTC close; KXBTCD strikes are far above/below spot (e.g., $79K strike when BTC = $76.7K). P ≈ 0, Kelly = 0, Gate 2 rejects. Do not debug these failures; consider filtering KXBTCD out of the market loop.
- **Kalshi orderbook format is `orderbook_fp`** — `yes_dollars` and `no_dollars` are lists of `[price_str, qty_str]` pairs in **ascending** order. Best bid = `list[-1][0]`. Old `orderbook.yes/no` format (descending integer cents) is no longer returned. If the parser starts getting empty books again, check whether the format changed again.
- **Gate 6 is skipped for `timeframe == "15min"`** — do not add strike proximity checks back for KXBTC15M. The "strike" for these markets is the last completed 15-min BRTI candle close (see below).
- **KXBTC15M threshold = last completed 15-min BRTI candle close** — KXBTC15M resolves "yes" if BRTI at resolution > BRTI at market open. "Market open" = close of the previous 15-min candle. `_extract_strike()` calls `_get_15min_reference_price()` for 15-min markets, which walks the 15-min OHLCV backwards to the most recent completed candle. Using the live 5-min close as threshold was wrong: mid-window drift (e.g. BTC up $300 in the first 10 min) would cause Kronos to ask the wrong directional question. Do NOT revert to `composite_price` for 15-min markets.
- **`watch` not available on macOS** — use `while true; do <command>; sleep 30; done` instead.
- **`_check_resolutions()` polls Kalshi for `market.status == "finalized"` and reads `market.result`** (`"yes"` or `"no"`). If field names change, resolutions stop working and `resolved` count freezes. The 24h age-out is the safety net.
- **Edge tracker needs 50 resolved trades before `is_above_threshold()` can return True** — Gate 4 is bypassed in paper mode (`edge_above_threshold = True if config.PAPER_TRADING`), so this doesn't block bootstrap.
- **Kalshi key**: ID in `.env` as `KALSHI_KEY_ID`, private key at `./keys/kalshi_private.key`. Both gitignored.
- **`trades.db`** in project root — source of truth for bootstrap progress. `dump.rdb` holds Redis tick history; do not delete it.
- **Test suite**: `python3 -m pytest` from project root — run after any model/gate change. **Expected baseline: 197 pass, 0 fail.**
- **RSA-PSS signing** (not PKCS1v15) for Kalshi; sign path-only, strip query string at `?`; base URL `https://api.elections.kalshi.com`.
