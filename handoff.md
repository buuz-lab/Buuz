# KronosV2 — Agent Handoff

## Goal

Bootstrap a live BTC prediction-market trading system on Kalshi. The system trades **KXBTC 15-minute and hourly up/down markets** (e.g. `KXBTC15M-*`, `KXBTCH-*`) — not the daily close-above-strike markets. It forecasts direction using the Kronos time-series model + XGBoost regime classifier + DeepSeek LLM gate, sizes positions with fractional Kelly, and runs 6 pre-trade gates. The immediate goal is to accumulate 500+ resolved paper trades so the calibrator and edge tracker cross their thresholds, then flip to live trading.

---

## Current Progress

**The system now starts and runs without crashing.** All prior segfaults and silent crashes are resolved.

- `trades.db` exists and has been initialized (schema created, at least one cycle ran)
- `PAPER_TRADING=true` in `.env`
- Redis is live with 7201 ticks and 65+ completed 5-min candles
- pykalshi auth works (RSA-PSS signing confirmed)
- Kronos model loads cleanly to CPU before asyncio starts
- DeepSeek is in neutral fallback mode (402 — no credits, not blocking trading)
- Bybit is geo-blocked (403 CloudFront); OKX failover is wired in
- Claude Code applied market-type changes: system now queries 15-min and hourly KXBTC markets, infers timeframe from ticker, and uses per-market close_time for blackout logic
- `trades.db` size: 12KB — schema is live, bootstrap in progress

**Bootstrap counters needed before going live:**
- Calibrator: `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL` → need ≥ 500
- Edge tracker: rolling window of last 30 trades with positive realized edge

---

## What Worked

- **Kronos preload in `__init__` before asyncio**: The only reliable fix for the Apple Silicon segfault. Loading the model inside `asyncio.to_thread()` while WebSocket feeds run concurrently causes a segfault because PyTorch's Accelerate-framework initialization races with macOS kqueue. Loading it in `KronosV2.__init__()` — before `asyncio.run()` — is single-threaded and safe.
- **`map_location="cpu"` in both `from_pretrained()` calls**: Forces weights to land on CPU during `torch.load()` inside `PyTorchModelHubMixin`, before `KronosPredictor` can do device selection.
- **`torch.set_num_threads(1)` BEFORE `from_pretrained()`**: Must precede all torch ops. Calling it after still races with Accelerate's internal thread pool init.
- **`asyncio.to_thread(self._run_cycle)`** in `_main_loop`: Keeps the 5-min blocking CPU cycle off the event loop thread so WebSocket feeds aren't starved.
- **RSA-PSS signing** (not PKCS1v15) for Kalshi auth; sign path-only, strip query string at `?`
- **pykalshi** installed via `pip install git+https://github.com/ArshKA/pykalshi`
- **DeepSeek 402 → `NEUTRAL_DEFAULT`** (regime=`ranging`, suppress=False): Using `high_uncertainty` as the 402 fallback was silently shrinking all signals by 50% and killing Gate 5 (signal edge vs spread), preventing any paper trades from being placed.
- **`fakeredis` injection** for tests: `store._redis = fakeredis.FakeRedis()` after `__new__`
- **Frozen candle hashes** in Redis (`brti:candles:{tf}`) with no TTL — monotonically non-decreasing candle count

---

## What Failed (avoid repeating)

- **Running Kronos inference in `asyncio.to_thread()` before preloading**: Even with `map_location="cpu"` and `device="cpu"`, calling `from_pretrained()` in a worker thread while asyncio WebSocket feeds run in the main thread segfaults on Apple Silicon (macOS kqueue + PyTorch Accelerate conflict). The isolation test script passed; main.py crashed. Root cause: concurrent C-extension initialization, not inference.
- **Calling `torch.set_num_threads(1)` after `from_pretrained()`**: Looked correct but the race happens during the load itself, not just inference.
- **`device="cpu"` in `KronosPredictor` alone, without `map_location="cpu"`**: `PyTorchModelHubMixin.from_pretrained()` defaults to MPS on Apple Silicon before `KronosPredictor.__init__` gets to call `.to("cpu")`. The move triggers a segfault.
- **`SAFE_DEFAULT` (high_uncertainty) as the DeepSeek fallback for billing errors**: Every failed DeepSeek call shrunk all Kronos signals by 50% toward 0.5, causing Gate 5 (signal edge vs spread check) to fail on nearly every market, so no paper trades were placed.
- **Bybit for derivatives**: HTTP 403 from AWS CloudFront geo-block for US users. OKX is now the primary fallback.
- **PKCS1v15 signing** — Kalshi requires RSA-PSS; auth fails with 401
- **Whole-DataFrame Redis writes for OHLCV** — had a TTL and caused candle count to fluctuate

---

## Files Touched / Created This Session

| File | Change |
|------|--------|
| `main.py` | Exception handler in `main()` to route crashes through loguru; `asyncio.to_thread` for `_run_cycle`; `self._kronos.preload()` in `KronosV2.__init__()`; fixed double market-context fetch; `PAPER_TRADING` set |
| `.env` | `PAPER_TRADING=true` |
| `btc_kalshi_system/models/kronos_engine.py` | `preload()` method; `torch.set_num_threads(1)` before loads; `map_location="cpu"` on both `from_pretrained()` calls |
| `btc_kalshi_system/models/deepseek_parser.py` | Added `NEUTRAL_DEFAULT`; 402 → neutral; no-key → neutral; other errors → `SAFE_DEFAULT` |
| `btc_kalshi_system/execution/router.py` | `logger.critical` before re-raise in `KalshiRawClient` init failure |
| `btc_kalshi_system/data/derivatives_feed.py` | Exchange preference list (OKX → Bybit); lazy exchange resolution; 403 mid-session failover |
| `btc_kalshi_system/signal/fusion.py` | (Claude Code) — comment clarifying Kronos threshold = current price for up/down markets |
| `tests/models/test_deepseek_parser.py` | Updated no-key test to expect `NEUTRAL_DEFAULT` |
| `scripts/test_kronos_cpu.py` | NEW — isolation smoke test for Kronos CPU inference |
| `scripts/bootstrap_progress.py` | NEW (Claude Code) — prints calibrator/edge tracker progress bars |
| `scripts/discover_kalshi_markets.py` | NEW (Claude Code) — probes Kalshi API for live market field names |
| `CLAUDE_CODE_FIX_PROMPT.md` | NEW — prompt used for Claude Code session (now obsolete, can delete) |

**Claude Code also modified** (applied via separate session from `CLAUDE_CODE_FIX_PROMPT.md`):
- `main.py`: `_get_active_markets()` now queries correct series for 15-min/hourly markets; `_extract_strike()` handles up/down reference price; `timeframe` derived from market type; `_is_in_blackout()` replaced with per-market `_market_is_in_blackout(market)`; removed `RESOLUTION_TIMES_EDT` hardcoded list
- `btc_kalshi_system/signal/fusion.py`: Minor clarifying comment

---

## Next Steps

1. **Confirm bootstrap is running** — in a terminal: `watch -n 30 'sqlite3 ~/Kronos\ V2/trades.db "SELECT COUNT(*) total, SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) resolved FROM trades;"'`. You should see `resolved` climbing every 15 min (15-min markets) and every hour (hourly markets). If it stays at 0 after 30 min, check logs for gate failures.

2. **Diagnose if no fills appear** — check what the 6 gates are rejecting: `grep "checklist failed" logs/kronos_*.log | tail -20`. Gate 5 (signal edge vs spread) and Gate 6 (strike proximity) are the most common early rejects. Gate 5 fails when `calibrated_prob - ask_price <= spread + 0.005`; if Kronos signals are weak (≤ 55% with uncalibrated model), this trips often. Acceptable during bootstrap — it means the system is being conservative, not broken.

3. **Add DeepSeek credits** — go to https://platform.deepseek.com and add $5–10. The LLM context gate will activate automatically on the next 15-min refresh cycle. Without it, the `ranging` fallback is used (fine for bootstrap, but reduces signal quality).

4. **Train the RegimeModel** — once ~100+ resolved trades exist, fit the XGBoost model using resolved trade features. Currently using Kronos-only with 0.5× shrink (conservative fallback). The `RegimeModel.save(path)` call is the trigger. The system handles `NotTrainedError` gracefully so this is not blocking.

5. **Go live** — once `SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL` ≥ 500 AND `edge_tracker.current_edge() > 0` over 30+ trades: set `PAPER_TRADING=false` in `.env` and restart.

---

## Context / Gotchas

- **Kronos MUST be preloaded in `KronosV2.__init__()`** — never load it inside `asyncio.to_thread()` or any async context on Apple Silicon. The `preload()` method handles the safe loading path. This is the single most important rule in this codebase for macOS.
- **torch 2.12.0 on Apple Silicon** — MPS is available but causes segfaults in the Kronos inference path. `map_location="cpu"` + `set_num_threads(1)` + preload-before-asyncio is the triple fix that works.
- **DeepSeek credits** — key `sk-41a0ee36dfc042168a1bdfe3f200e5cd` is active but has $0 balance. System trades without it (neutral fallback). Add credits at platform.deepseek.com.
- **Bybit geo-blocked** for US users (HTTP 403 CloudFront). OKX is now the primary derivatives exchange; if OKX also blocks, the feed silently writes zero-valued regime features and the system continues.
- **RSA-PSS signing** (not PKCS1v15) for Kalshi auth; sign path-only, strip query string at `?`; base URL `https://api.elections.kalshi.com`
- **15-min market bootstrap math**: KXBTC 15-min markets resolve every 15 minutes. With ~10–30 active markets at any time, expect 10–30 resolved paper trades per 15-min window, or ~960–2880 per day. Bootstrap to 500 should take less than 1 day if Gate 5 passes consistently.
- **`candle_count()` vs `len(get_ohlcv())`** — use `store.candle_count("5min")` for the monotonic metric.
- **Resolution detection** — `_check_resolutions()` polls Kalshi for `market.status == "finalized"` and reads `market.result` (`"yes"` or `"no"`). Field names could change; if resolution stops working, add a 24-hour age-out fallback.
- **EDT offset** — the per-market blackout now uses each market's own `close_time` from the Kalshi API (UTC), so the old hardcoded EDT offset issue is gone.
- **Test suite**: `python3 -m pytest` from project root — 197 tests, all green. Run after any model change.
- **Credentials** — Kalshi key ID: `9c53fcaf-13aa-46d6-ae4b-0a8f1363ede2`; private key at `./keys/kalshi_private.key`. Both gitignored. Never commit.
- **`trades.db`** in project root — source of truth for bootstrap progress. `dump.rdb` holds Redis tick history; deleting it resets candle history (don't delete).
